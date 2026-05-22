"""PyTorch MLP upgrade path for M2HSG stage-1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

from .evaluation import evaluate_predictions
from .features import LinearFeatureBuilder
from .io_utils import read_jsonl, write_json
from .linear_baseline import _compute_relation_balanced_weights


@dataclass
class MLPTrainConfig:
    supervision_dir: Path
    output_dir: Path
    seed: int = 42
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    rel_emb_dim: int = 16
    hidden_dim: int = 128
    max_text_features: int = 5000
    svd_dim: int = 512


def _load_split(supervision_dir: Path, split: str) -> List[Dict[str, Any]]:
    return read_jsonl(supervision_dir / f"{split}.supervision.jsonl")


def _heuristic_tier_to_label(row: Dict[str, Any]) -> int:
    st = str(row.get("source_type", "weak")).strip().lower()
    if st == "hard_negative":
        return 0
    tier = str(row.get("heuristic_tier", row.get("base_tier", "weak"))).strip().lower()
    if st == "repaired":
        tier = str(row.get("base_tier", tier)).strip().lower() or tier
    return 2 if tier == "strong" else 1 if tier == "medium" else 0


class RelationAwareMLP:
    """Relation-aware MLP with lazy torch import.

    Defined at module level so it can be imported by rerank.py.
    The actual nn.Module is created lazily when torch is available.
    """

    @staticmethod
    def build(input_dim: int, rel_count: int, rel_emb_dim: int, hidden_dim: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.rel_emb = nn.Embedding(rel_count, rel_emb_dim)
                self.fc1 = nn.Linear(input_dim + rel_emb_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.edge_head = nn.Linear(hidden_dim, 1)
                self.tier_head = nn.Linear(hidden_dim, 3)

            def forward(self, x, rel_idx):
                emb = self.rel_emb(rel_idx)
                z = torch.cat([x, emb], dim=1)
                z = F.gelu(self.fc1(z))
                z = F.dropout(z, p=0.1, training=self.training)
                z = F.gelu(self.fc2(z))
                edge_logit = self.edge_head(z).squeeze(1)
                tier_logits = self.tier_head(z)
                return edge_logit, tier_logits

        return _Impl()


class NewRelationAwareMLP:
    """Paper-aligned MLP: L_edge (pairwise BCE) + L_hyper (hyperedge coherence BCE).

    Eq.16: L_edge = BCE(edge_logit, a_ij)
    Eq.17: L_hyper = BCE(hyperedge_logit, b_k)
    Eq.18: L_graph = L_edge + L_hyper
    """

    @staticmethod
    def build(input_dim: int, rel_count: int, rel_emb_dim: int, hidden_dim: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.rel_emb = nn.Embedding(rel_count, rel_emb_dim)
                self.fc1 = nn.Linear(input_dim + rel_emb_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.edge_head = nn.Linear(hidden_dim, 1)
                self.hyperedge_head = nn.Linear(hidden_dim, 1)

            def forward(self, x, rel_idx):
                emb = self.rel_emb(rel_idx)
                z = torch.cat([x, emb], dim=1)
                z = F.gelu(self.fc1(z))
                z = F.dropout(z, p=0.1, training=self.training)
                z = F.gelu(self.fc2(z))
                edge_logit = self.edge_head(z).squeeze(1)
                hyperedge_logit = self.hyperedge_head(z).squeeze(1)
                return edge_logit, hyperedge_logit

        return _Impl()


def train_mlp(cfg: MLPTrainConfig) -> Dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:  # pragma: no cover
        report = {
            "status": "skipped_torch_missing",
            "reason": str(e),
            "supervision_dir": str(cfg.supervision_dir.resolve()),
        }
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(cfg.output_dir / "mlp_training_report.json", report)
        return report

    class _RowsDataset(Dataset):
        def __init__(self, x: np.ndarray, rel_idx: np.ndarray, tier: np.ndarray, edge: np.ndarray, w: np.ndarray):
            self.x = torch.from_numpy(x).float()
            self.rel_idx = torch.from_numpy(rel_idx).long()
            self.tier = torch.from_numpy(tier).long()
            self.edge = torch.from_numpy(edge).float()
            self.w = torch.from_numpy(w).float()

        def __len__(self) -> int:
            return self.x.shape[0]

        def __getitem__(self, idx: int) -> Tuple[Any, ...]:
            return self.x[idx], self.rel_idx[idx], self.tier[idx], self.edge[idx], self.w[idx]

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    supervision_dir = cfg.supervision_dir.resolve()
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_split(supervision_dir, "train")
    dev_rows = _load_split(supervision_dir, "dev")
    test_rows = _load_split(supervision_dir, "test")

    fit_rows = [x for x in train_rows if bool(x.get("include_in_loss"))]
    if not fit_rows:
        raise RuntimeError("no train rows with include_in_loss=true")

    fb = LinearFeatureBuilder(max_text_features=cfg.max_text_features).fit(fit_rows)
    x_train_sparse = fb.transform(fit_rows)
    x_dev_sparse = fb.transform(dev_rows)
    x_test_sparse = fb.transform(test_rows)

    n_comp = min(cfg.svd_dim, max(32, x_train_sparse.shape[0] - 1), max(32, x_train_sparse.shape[1] - 1))
    svd = TruncatedSVD(n_components=int(n_comp), random_state=int(cfg.seed))
    x_train = svd.fit_transform(x_train_sparse).astype(np.float32)
    x_dev = svd.transform(x_dev_sparse).astype(np.float32)
    x_test = svd.transform(x_test_sparse).astype(np.float32)

    relations = sorted({str(x.get("relation", "")) for x in train_rows + dev_rows + test_rows})
    rel_to_idx = {r: i for i, r in enumerate(relations)}
    rel_train = np.array([rel_to_idx[str(x.get("relation", ""))] for x in fit_rows], dtype=np.int64)
    rel_dev = np.array([rel_to_idx[str(x.get("relation", ""))] for x in dev_rows], dtype=np.int64)
    rel_test = np.array([rel_to_idx[str(x.get("relation", ""))] for x in test_rows], dtype=np.int64)

    tier_train = np.array([int(x.get("tier_label", 0)) for x in fit_rows], dtype=np.int64)
    tier_dev = np.array([int(x.get("tier_label", 0)) for x in dev_rows], dtype=np.int64)
    tier_test = np.array([int(x.get("tier_label", 0)) for x in test_rows], dtype=np.int64)
    edge_train = np.array([1.0 if int(x.get("tier_label", 0)) in {1, 2} else 0.0 for x in fit_rows], dtype=np.float32)
    edge_dev = np.array([1.0 if int(x.get("tier_label", 0)) in {1, 2} else 0.0 for x in dev_rows], dtype=np.float32)
    edge_test = np.array([1.0 if int(x.get("tier_label", 0)) in {1, 2} else 0.0 for x in test_rows], dtype=np.float32)
    w_train = np.array([float(x.get("sample_weight", 1.0)) for x in fit_rows], dtype=np.float32)
    w_train = _compute_relation_balanced_weights(fit_rows, w_train)

    train_ds = _RowsDataset(x_train, rel_train, tier_train, edge_train, w_train)
    train_dl = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RelationAwareMLP.build(
        input_dim=x_train.shape[1],
        rel_count=max(1, len(rel_to_idx)),
        rel_emb_dim=int(cfg.rel_emb_dim),
        hidden_dim=int(cfg.hidden_dim),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)

    best_dev = -1.0
    best_state = None
    train_trace = []

    for epoch in range(int(cfg.epochs)):
        model.train()
        total_loss = 0.0
        for xb, rb, tb, eb, wb in train_dl:
            xb = xb.to(device)
            rb = rb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)
            wb = wb.to(device)

            edge_logit, tier_logits = model(xb, rb)
            edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_logit, eb, reduction="none")
            tier_loss = torch.nn.functional.cross_entropy(tier_logits, tb, reduction="none")
            loss = ((0.5 * edge_loss + 0.5 * tier_loss) * wb).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            dev_x = torch.from_numpy(x_dev).float().to(device)
            dev_r = torch.from_numpy(rel_dev).long().to(device)
            dev_edge_logit, dev_tier_logits = model(dev_x, dev_r)
            dev_edge_prob = torch.sigmoid(dev_edge_logit).cpu().numpy().tolist()
            dev_tier_prob = torch.softmax(dev_tier_logits, dim=1).cpu().numpy()
            dev_tier_pred = np.argmax(dev_tier_prob, axis=1).astype(np.int64).tolist()
            heuristic_edge = [float(r.get("heuristic_score", 0.0)) if str(r.get("source_type", "")).lower() != "hard_negative" else 0.0 for r in dev_rows]
            heuristic_tier = [_heuristic_tier_to_label(r) for r in dev_rows]
            dev_metrics = evaluate_predictions(
                rows=dev_rows,
                learned_edge_scores=dev_edge_prob,
                learned_tier_labels=dev_tier_pred,
                heuristic_edge_scores=heuristic_edge,
                heuristic_tier_labels=heuristic_tier,
            )
            score = float(dev_metrics["learned_edge_pr_auc"] + dev_metrics["learned_tier_macro_f1"])
            train_trace.append(
                {
                    "epoch": epoch + 1,
                    "loss": round(total_loss / max(1, len(train_dl)), 6),
                    "dev_learned_edge_pr_auc": round(float(dev_metrics["learned_edge_pr_auc"]), 6),
                    "dev_learned_tier_macro_f1": round(float(dev_metrics["learned_tier_macro_f1"]), 6),
                }
            )
            if score > best_dev:
                best_dev = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    def _eval(rows: List[Dict[str, Any]], x: np.ndarray, rel: np.ndarray) -> Dict[str, Any]:
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x).float().to(device)
            rt = torch.from_numpy(rel).long().to(device)
            edge_logit, tier_logits = model(xt, rt)
            edge_prob = torch.sigmoid(edge_logit).cpu().numpy()
            tier_prob = torch.softmax(tier_logits, dim=1).cpu().numpy()
            tier_pred = np.argmax(tier_prob, axis=1).astype(np.int64)
        heuristic_edge = [float(r.get("heuristic_score", 0.0)) if str(r.get("source_type", "")).lower() != "hard_negative" else 0.0 for r in rows]
        heuristic_tier = [_heuristic_tier_to_label(r) for r in rows]
        metrics = evaluate_predictions(
            rows=rows,
            learned_edge_scores=edge_prob.tolist(),
            learned_tier_labels=tier_pred.tolist(),
            heuristic_edge_scores=heuristic_edge,
            heuristic_tier_labels=heuristic_tier,
        )
        return {
            "metrics": metrics,
            "edge_scores": edge_prob.tolist(),
            "tier_pred": tier_pred.tolist(),
            "tier_probs": tier_prob.tolist(),
        }

    dev_eval = _eval(dev_rows, x_dev, rel_dev)
    test_eval = _eval(test_rows, x_test, rel_test)

    state_path = output_dir / "mlp_model.pt"
    torch.save(model.state_dict(), state_path)
    joblib.dump({"feature_builder": fb, "svd": svd, "rel_to_idx": rel_to_idx}, output_dir / "mlp_featurizer.joblib")

    report = {
        "status": "ok",
        "seed": int(cfg.seed),
        "supervision_dir": str(supervision_dir),
        "state_path": str(state_path.resolve()),
        "featurizer_path": str((output_dir / "mlp_featurizer.joblib").resolve()),
        "train_rows_total": len(train_rows),
        "train_rows_loss": len(fit_rows),
        "dev": {"n_rows": len(dev_rows), "metrics": dev_eval["metrics"]},
        "test": {"n_rows": len(test_rows), "metrics": test_eval["metrics"]},
        "train_trace": train_trace,
    }
    write_json(output_dir / "mlp_training_report.json", report)
    return report
