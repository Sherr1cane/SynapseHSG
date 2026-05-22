"""Offline rerank for stage-1 candidate triples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json

import joblib
import numpy as np

from .frozen.rules import compute_heuristic_score_tier, infer_source_type, provenance_modality, provenance_quality
from .io_utils import read_json, read_jsonl, write_json
from .linear_baseline import load_linear_model

TIER_TO_LABEL = {"weak": 0, "medium": 1, "strong": 2}
LABEL_TO_TIER = {0: "weak", 1: "medium", 2: "strong"}


@dataclass
class RerankConfig:
    run_dir: Path
    model_path: Path
    output_dir: Path
    keep_threshold: float = 0.5
    model_type: str = "linear"  # "linear" or "mlp"


def _safe_text(x: Any) -> str:
    return str(x or "").strip()


def _modality_summary(provenance: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    counts = {"audio": 0, "slide": 0, "paper": 0}
    stats = {
        "audio": {"quality_max": 0.0, "duration_sum": 0.0},
        "slide": {"quality_max": 0.0, "has_chart_hint": 0.0},
        "paper": {"quality_max": 0.0, "repaired_count": 0.0, "similarity_max": 0.0},
    }
    for p in provenance:
        if not isinstance(p, dict):
            continue
        m = provenance_modality(p)
        if m not in counts:
            continue
        counts[m] += 1
        q = provenance_quality(m, p)
        stats[m]["quality_max"] = max(float(stats[m]["quality_max"]), float(q))
        if m == "audio":
            try:
                st = float(p.get("start_time"))
                et = float(p.get("end_time"))
                if et > st:
                    stats[m]["duration_sum"] += max(0.0, et - st)
            except Exception:
                pass
        elif m == "slide":
            if _safe_text(p.get("chart_type")) and _safe_text(p.get("chart_type")) != "none":
                stats[m]["has_chart_hint"] = 1.0
        elif m == "paper":
            if bool(p.get("is_repaired")):
                stats[m]["repaired_count"] += 1.0
            try:
                sim = float(p.get("similarity_score", 0.0))
                stats[m]["similarity_max"] = max(float(stats[m]["similarity_max"]), sim)
            except Exception:
                pass
    return counts, stats


def _triple_to_infer_sample(sample_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    head = row.get("head", {}) if isinstance(row.get("head", {}), dict) else {}
    tail = row.get("tail", {}) if isinstance(row.get("tail", {}), dict) else {}
    provenance = row.get("provenance", []) if isinstance(row.get("provenance", []), list) else []
    mcounts, mstats = _modality_summary([p for p in provenance if isinstance(p, dict)])
    heuristic_score, heuristic_tier, repaired, _ = compute_heuristic_score_tier(row)
    source_type, base_tier = infer_source_type(row, hard_negative=False)
    tier = _safe_text(row.get("tier", "weak")).lower()
    if tier not in {"weak", "medium", "strong"}:
        tier = "weak"
    tier_label = TIER_TO_LABEL.get(base_tier or tier, 0)

    return {
        "sample_id": sample_id,
        "head_id": _safe_text(head.get("id")),
        "head_type": _safe_text(head.get("type")),
        "head_content": _safe_text(head.get("content")),
        "relation": _safe_text(row.get("relation")),
        "tail_id": _safe_text(tail.get("id")),
        "tail_type": _safe_text(tail.get("type")),
        "tail_content": _safe_text(tail.get("content")),
        "is_repaired": bool(row.get("is_repaired")) or repaired,
        "heuristic_score": float(row.get("provenance_score", heuristic_score) or heuristic_score),
        "heuristic_tier": heuristic_tier,
        "tier_label": int(tier_label),
        "source_type": source_type,
        "base_tier": base_tier,
        "negative_type": None,
        "modality_mask": {
            "audio": 1 if mcounts["audio"] > 0 else 0,
            "visual": 1 if mcounts["slide"] > 0 else 0,
            "paper": 1 if mcounts["paper"] > 0 else 0,
        },
        "audio_repr": {
            "count": mcounts["audio"],
            "quality_max": round(float(mstats["audio"]["quality_max"]), 4),
            "duration_sum": round(float(mstats["audio"]["duration_sum"]), 4),
        },
        "visual_repr": {
            "count": mcounts["slide"],
            "quality_max": round(float(mstats["slide"]["quality_max"]), 4),
            "has_chart_hint": float(mstats["slide"]["has_chart_hint"]),
        },
        "paper_repr": {
            "count": mcounts["paper"],
            "quality_max": round(float(mstats["paper"]["quality_max"]), 4),
            "repaired_count": int(mstats["paper"]["repaired_count"]),
            "similarity_max": round(float(mstats["paper"]["similarity_max"]), 4),
        },
    }


def _predict_linear(rows: List[Dict[str, Any]], model_path: Path) -> Tuple[List[float], List[int], List[Dict[str, float]]]:
    fb, model = load_linear_model(model_path)
    x = fb.transform(rows)
    probs = model.predict_proba(x)
    classes = {int(c): i for i, c in enumerate(model.classes_.tolist())}
    idx_w = classes.get(0)
    idx_m = classes.get(1)
    idx_s = classes.get(2)

    edge_scores: List[float] = []
    tier_labels: List[int] = []
    tier_probs: List[Dict[str, float]] = []
    for row in probs:
        p_w = float(row[idx_w]) if idx_w is not None else 0.0
        p_m = float(row[idx_m]) if idx_m is not None else 0.0
        p_s = float(row[idx_s]) if idx_s is not None else 0.0
        edge_scores.append(p_m + p_s)
        tier_probs.append({"weak": p_w, "medium": p_m, "strong": p_s})
        tier_labels.append(int(model.classes_[int(row.argmax())]))
    return edge_scores, tier_labels, tier_probs


def _predict_mlp(rows: List[Dict[str, Any]], model_path: Path) -> Tuple[List[float], List[int], List[Dict[str, float]]]:
    """Predict with trained MLP model (state_dict) + featurizer bundle."""
    import torch
    from .mlp import RelationAwareMLP

    bundle_dir = model_path.parent
    featurizer_path = bundle_dir / "mlp_featurizer.joblib"
    if not featurizer_path.exists():
        # Try sibling path convention
        featurizer_path = model_path.parent / "mlp_featurizer.joblib"
    if not featurizer_path.exists():
        raise FileNotFoundError(f"MLP featurizer not found at {featurizer_path}")

    feat_bundle = joblib.load(featurizer_path)
    fb = feat_bundle["feature_builder"]
    svd = feat_bundle["svd"]
    rel_to_idx = feat_bundle["rel_to_idx"]
    idx_to_rel = {v: k for k, v in rel_to_idx.items()}
    rel_count = max(1, len(rel_to_idx))

    x_sparse = fb.transform(rows)
    x_dense = svd.transform(x_sparse).astype(np.float32)
    rel_idx = np.array([rel_to_idx.get(str(r.get("relation", "")), 0) for r in rows], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Infer architecture from state_dict
    state = torch.load(model_path, map_location=device, weights_only=True)
    emb_weight = state.get("rel_emb.weight")
    rel_emb_dim = emb_weight.shape[1] if emb_weight is not None else 16
    fc1_weight = state.get("fc1.weight")
    hidden_dim = fc1_weight.shape[0] if fc1_weight is not None else 128
    input_dim = x_dense.shape[1]

    model = RelationAwareMLP.build(
        input_dim=input_dim,
        rel_count=rel_count,
        rel_emb_dim=rel_emb_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(x_dense).float().to(device)
        rt = torch.from_numpy(rel_idx).long().to(device)
        edge_logit, tier_logits = model(xt, rt)
        edge_prob = torch.sigmoid(edge_logit).cpu().numpy()
        tier_prob = torch.softmax(tier_logits, dim=1).cpu().numpy()
        tier_pred = np.argmax(tier_prob, axis=1).astype(np.int64)

    LABEL_MAP = {0: "weak", 1: "medium", 2: "strong"}
    edge_scores: List[float] = []
    tier_labels: List[int] = []
    tier_probs_list: List[Dict[str, float]] = []
    n_classes = tier_prob.shape[1]
    for i in range(len(rows)):
        edge_scores.append(float(edge_prob[i]))
        tier_labels.append(int(tier_pred[i]))
        probs_dict = {}
        for c in range(min(n_classes, 3)):
            probs_dict[LABEL_MAP.get(c, str(c))] = float(tier_prob[i, c])
        tier_probs_list.append(probs_dict)
    return edge_scores, tier_labels, tier_probs_list


def _predict_mlp_v2(rows, model_path):
    """Predict with MLP v2 (edge + hyperedge heads)."""
    import torch
    from .mlp import NewRelationAwareMLP

    bundle_dir = model_path.parent
    featurizer_path = bundle_dir / "mlp_featurizer.joblib"
    if not featurizer_path.exists():
        featurizer_path = model_path.parent / "mlp_featurizer.joblib"
    if not featurizer_path.exists():
        raise FileNotFoundError(f"MLP featurizer not found at {featurizer_path}")

    feat_bundle = joblib.load(featurizer_path)
    fb = feat_bundle["feature_builder"]
    svd = feat_bundle["svd"]
    rel_to_idx = feat_bundle["rel_to_idx"]
    rel_count = max(1, len(rel_to_idx))

    x_sparse = fb.transform(rows)
    x_dense = svd.transform(x_sparse).astype(np.float32)
    rel_idx = np.array([rel_to_idx.get(str(r.get("relation", "")), 0) for r in rows], dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(model_path, map_location=device, weights_only=True)
    emb_weight = state.get("rel_emb.weight")
    rel_emb_dim = emb_weight.shape[1] if emb_weight is not None else 16
    fc1_weight = state.get("fc1.weight")
    hidden_dim = fc1_weight.shape[0] if fc1_weight is not None else 128
    input_dim = x_dense.shape[1]

    model = NewRelationAwareMLP.build(
        input_dim=input_dim,
        rel_count=rel_count,
        rel_emb_dim=rel_emb_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        xt = torch.from_numpy(x_dense).float().to(device)
        rt = torch.from_numpy(rel_idx).long().to(device)
        edge_logit, hyper_logit = model(xt, rt)
        edge_prob = torch.sigmoid(edge_logit).cpu().numpy()
        hyper_prob = torch.sigmoid(hyper_logit).cpu().numpy()

    # Combined score: edge quality + hyperedge coherence
    edge_scores = (0.5 * edge_prob + 0.5 * hyper_prob).tolist()
    # For compatibility: tier_labels from edge threshold
    tier_labels = (edge_prob >= 0.5).astype(int).tolist()
    tier_probs_list = [
        {"edge": round(float(edge_prob[i]), 6), "hyperedge": round(float(hyper_prob[i]), 6)}
        for i in range(len(rows))
    ]
    return edge_scores, tier_labels, tier_probs_list


def rerank_offline(cfg: RerankConfig) -> Dict[str, Any]:
    run_dir = cfg.run_dir.resolve()
    model_path = cfg.model_path.resolve()
    out_dir = cfg.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    predict_fn = (_predict_mlp_v2 if cfg.model_type == "mlp_v2"
                  else _predict_mlp if cfg.model_type == "mlp"
                  else _predict_linear)

    splits = read_json(run_dir / "splits.json")
    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "model_type": cfg.model_type,
        "keep_threshold": float(cfg.keep_threshold),
        "splits": {},
        "total_rows": 0,
        "kept_rows": 0,
    }

    for split in ["train", "dev", "test"]:
        split_stats = {"rows": 0, "kept": 0, "sessions": {}}
        for sample_id in splits.get(split, []) or []:
            conf, sess = str(sample_id).split("/", 1)
            src_path = run_dir / "sessions" / conf / sess / "kg_triples.jsonl"
            if not src_path.exists():
                continue
            triples = read_jsonl(src_path)
            infer_rows = [_triple_to_infer_sample(str(sample_id), r) for r in triples]
            if infer_rows:
                edge_scores, tier_labels, tier_probs = predict_fn(infer_rows, model_path)
            else:
                edge_scores, tier_labels, tier_probs = [], [], []

            out_rows: List[Dict[str, Any]] = []
            kept = 0
            for raw, inf, edge, tier_label, probs in zip(triples, infer_rows, edge_scores, tier_labels, tier_probs):
                keep = bool(float(edge) >= float(cfg.keep_threshold))
                if keep:
                    kept += 1
                out_rows.append(
                    {
                        "head": raw.get("head"),
                        "relation": raw.get("relation"),
                        "tail": raw.get("tail"),
                        "tier": raw.get("tier"),
                        "provenance_score": raw.get("provenance_score"),
                        "is_repaired": raw.get("is_repaired"),
                        "learned_edge_score": round(float(edge), 6),
                        "learned_tier_probs": {k: round(float(v), 6) for k, v in probs.items()},
                        "learned_tier": LABEL_TO_TIER.get(int(tier_label), "weak"),
                        "heuristic_score": round(float(inf.get("heuristic_score", 0.0)), 6),
                        "score_delta": round(float(edge) - float(inf.get("heuristic_score", 0.0)), 6),
                        "keep_decision": keep,
                        "source_type": inf.get("source_type"),
                        "base_tier": inf.get("base_tier"),
                        "negative_type": inf.get("negative_type"),
                    }
                )

            dst = out_dir / "sessions" / conf / sess / "kg_triples.rerank.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("w", encoding="utf-8") as f:
                for row in out_rows:
                    f.write(json.dumps(row, ensure_ascii=False))
                    f.write("\n")

            split_stats["rows"] += len(out_rows)
            split_stats["kept"] += kept
            split_stats["sessions"][str(sample_id)] = {
                "rows": len(out_rows),
                "kept": kept,
                "output_path": str(dst),
            }

        report["splits"][split] = split_stats
        report["total_rows"] += split_stats["rows"]
        report["kept_rows"] += split_stats["kept"]

    write_json(out_dir / "rerank_report.json", report)
    return report

