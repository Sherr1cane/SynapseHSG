"""Relation-aware linear baseline for M2HSG stage-1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from .evaluation import evaluate_predictions, summarize_seed_metrics
from .features import LinearFeatureBuilder
from .io_utils import read_jsonl, write_json

LABEL_TO_TIER = {0: "weak", 1: "medium", 2: "strong"}


def _compute_relation_balanced_weights(rows: List[Dict[str, Any]], base_weights: np.ndarray) -> np.ndarray:
    """Inverse-frequency relation-balanced sample weights.

    Up-weights rare relations (compares, measured_by) by computing a
    multiplier inversely proportional to relation frequency among
    supervised rows, normalized so mean multiplier = 1.0.
    """
    from collections import Counter
    rels = [str(r.get("relation", "")) for r in rows]
    counts = Counter(rels)
    if not counts:
        return base_weights
    n = len(rels)
    n_rel = len(counts)
    rel_weight = {r: n / (n_rel * c) for r, c in counts.items()}
    out = np.array(
        [float(base_weights[i]) * rel_weight.get(rels[i], 1.0) for i in range(n)],
        dtype=np.float32,
    )
    return out


@dataclass
class LinearTrainConfig:
    supervision_dir: Path
    output_dir: Path
    seeds: List[int]
    max_iter: int = 2000
    c: float = 1.0
    max_text_features: int = 5000


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


def _edge_score_from_probs(classes: np.ndarray, probs: np.ndarray) -> List[float]:
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    idx_m = class_to_idx.get(1)
    idx_s = class_to_idx.get(2)
    out = []
    for row in probs:
        val = 0.0
        if idx_m is not None:
            val += float(row[idx_m])
        if idx_s is not None:
            val += float(row[idx_s])
        out.append(val)
    return out


def _tier_label_from_probs(classes: np.ndarray, probs: np.ndarray) -> List[int]:
    out = []
    for row in probs:
        idx = int(np.argmax(row))
        out.append(int(classes[idx]))
    return out


def _pred_probs_as_dict(classes: np.ndarray, probs: np.ndarray) -> List[Dict[str, float]]:
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    idx_w = class_to_idx.get(0)
    idx_m = class_to_idx.get(1)
    idx_s = class_to_idx.get(2)
    out = []
    for row in probs:
        out.append(
            {
                "weak": float(row[idx_w]) if idx_w is not None else 0.0,
                "medium": float(row[idx_m]) if idx_m is not None else 0.0,
                "strong": float(row[idx_s]) if idx_s is not None else 0.0,
            }
        )
    return out


def _fit_one_seed(
    *,
    seed: int,
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    out_dir: Path,
    max_iter: int,
    c: float,
    max_text_features: int,
) -> Dict[str, Any]:
    fit_rows = [x for x in train_rows if bool(x.get("include_in_loss"))]
    if not fit_rows:
        raise RuntimeError("no train rows with include_in_loss=true")

    fb = LinearFeatureBuilder(max_text_features=max_text_features).fit(fit_rows)
    x_train = fb.transform(fit_rows)
    y_train = np.array([int(x.get("tier_label", 0)) for x in fit_rows], dtype=np.int64)
    w_train = np.array([float(x.get("sample_weight", 1.0)) for x in fit_rows], dtype=np.float32)
    w_train = _compute_relation_balanced_weights(fit_rows, w_train)

    model = LogisticRegression(
        random_state=seed,
        max_iter=max_iter,
        C=c,
        solver="saga",
    )
    model.fit(x_train, y_train, sample_weight=w_train)

    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_path = seed_dir / "linear_model.joblib"
    joblib.dump({"feature_builder": fb, "model": model, "seed": seed}, model_path)

    reports: Dict[str, Dict[str, Any]] = {}
    for split_name, rows in [("dev", dev_rows), ("test", test_rows)]:
        x = fb.transform(rows)
        probs = model.predict_proba(x)
        learned_edge = _edge_score_from_probs(model.classes_, probs)
        learned_tier = _tier_label_from_probs(model.classes_, probs)
        learned_probs = _pred_probs_as_dict(model.classes_, probs)

        heuristic_edge = [float(r.get("heuristic_score", 0.0)) if str(r.get("source_type", "")).lower() != "hard_negative" else 0.0 for r in rows]
        heuristic_tier = [_heuristic_tier_to_label(r) for r in rows]

        metrics_all_rows = evaluate_predictions(
            rows=rows,
            learned_edge_scores=learned_edge,
            learned_tier_labels=learned_tier,
            heuristic_edge_scores=heuristic_edge,
            heuristic_tier_labels=heuristic_tier,
        )
        supervised_idx = [i for i, r in enumerate(rows) if bool(r.get("include_in_loss"))]
        rows_supervised = [rows[i] for i in supervised_idx]
        learned_edge_supervised = [learned_edge[i] for i in supervised_idx]
        learned_tier_supervised = [learned_tier[i] for i in supervised_idx]
        heuristic_edge_supervised = [heuristic_edge[i] for i in supervised_idx]
        heuristic_tier_supervised = [heuristic_tier[i] for i in supervised_idx]
        metrics_supervised = evaluate_predictions(
            rows=rows_supervised,
            learned_edge_scores=learned_edge_supervised,
            learned_tier_labels=learned_tier_supervised,
            heuristic_edge_scores=heuristic_edge_supervised,
            heuristic_tier_labels=heuristic_tier_supervised,
        )
        reports[split_name] = {
            "metrics": metrics_all_rows,
            "metrics_all_rows": metrics_all_rows,
            "metrics_supervised": metrics_supervised,
            "n_rows": len(rows),
            "n_rows_all_rows": len(rows),
            "n_rows_supervised": len(rows_supervised),
        }

        pred_rows = []
        for row, edge_score, tier_label, prob in zip(rows, learned_edge, learned_tier, learned_probs):
            pred_rows.append(
                {
                    "sample_key": row.get("sample_key"),
                    "sample_id": row.get("sample_id"),
                    "split": split_name,
                    "source_type": row.get("source_type"),
                    "negative_type": row.get("negative_type"),
                    "head_id": row.get("head_id"),
                    "relation": row.get("relation"),
                    "tail_id": row.get("tail_id"),
                    "learned_edge_score": round(float(edge_score), 6),
                    "learned_tier_probs": {k: round(float(v), 6) for k, v in prob.items()},
                    "learned_tier": LABEL_TO_TIER.get(int(tier_label), "weak"),
                    "heuristic_score": round(float(row.get("heuristic_score", 0.0)), 6),
                    "heuristic_tier": row.get("heuristic_tier"),
                }
            )
        with (seed_dir / f"{split_name}.predictions.jsonl").open("w", encoding="utf-8") as f:
            for pr in pred_rows:
                f.write(json.dumps(pr, ensure_ascii=False))
                f.write("\n")

    seed_report = {
        "seed": seed,
        "model_path": str(model_path),
        "train_rows_total": len(train_rows),
        "train_rows_loss": len(fit_rows),
        "dev": reports["dev"],
        "test": reports["test"],
    }
    write_json(seed_dir / "seed_report.json", seed_report)
    return seed_report


def train_linear_baseline(cfg: LinearTrainConfig) -> Dict[str, Any]:
    supervision_dir = cfg.supervision_dir.resolve()
    out_dir = cfg.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_split(supervision_dir, "train")
    dev_rows = _load_split(supervision_dir, "dev")
    test_rows = _load_split(supervision_dir, "test")

    fallback = {
        "train_from_dev": False,
        "train_from_test": False,
        "dev_from_train": False,
        "dev_from_test": False,
        "test_from_train": False,
        "test_from_dev": False,
    }
    if not train_rows and dev_rows:
        train_rows = list(dev_rows)
        fallback["train_from_dev"] = True
    if not train_rows and test_rows:
        train_rows = list(test_rows)
        fallback["train_from_test"] = True
    if not dev_rows and train_rows:
        dev_rows = list(train_rows)
        fallback["dev_from_train"] = True
    if not dev_rows and test_rows:
        dev_rows = list(test_rows)
        fallback["dev_from_test"] = True
    if not test_rows and train_rows:
        test_rows = list(train_rows)
        fallback["test_from_train"] = True
    if not test_rows and dev_rows:
        test_rows = list(dev_rows)
        fallback["test_from_dev"] = True

    seed_reports: List[Dict[str, Any]] = []
    for seed in cfg.seeds:
        seed_reports.append(
            _fit_one_seed(
                seed=int(seed),
                train_rows=train_rows,
                dev_rows=dev_rows,
                test_rows=test_rows,
                out_dir=out_dir,
                max_iter=cfg.max_iter,
                c=cfg.c,
                max_text_features=cfg.max_text_features,
            )
        )

    dev_metric_rows_all = [x["dev"]["metrics_all_rows"] for x in seed_reports]
    test_metric_rows_all = [x["test"]["metrics_all_rows"] for x in seed_reports]
    dev_metric_rows_supervised = [x["dev"]["metrics_supervised"] for x in seed_reports]
    test_metric_rows_supervised = [x["test"]["metrics_supervised"] for x in seed_reports]
    dev_summary_all = summarize_seed_metrics(dev_metric_rows_all)
    test_summary_all = summarize_seed_metrics(test_metric_rows_all)
    dev_summary_supervised = summarize_seed_metrics(dev_metric_rows_supervised)
    test_summary_supervised = summarize_seed_metrics(test_metric_rows_supervised)

    # Select best seed by dev (edge + tier)
    def score_seed(row: Dict[str, Any]) -> float:
        m = row["dev"]["metrics_supervised"]
        return float(m["learned_edge_pr_auc"]) + float(m["learned_tier_macro_f1"])

    best = sorted(seed_reports, key=score_seed, reverse=True)[0]
    best_model_path = Path(best["model_path"])
    model_bundle = joblib.load(best_model_path)
    joblib.dump(model_bundle, out_dir / "linear_best.joblib")

    report = {
        "supervision_dir": str(supervision_dir),
        "seeds": cfg.seeds,
        "split_fallback": fallback,
        "seed_reports": seed_reports,
        "dev_summary": dev_summary_supervised,
        "test_summary": test_summary_supervised,
        "dev_summary_supervised": dev_summary_supervised,
        "test_summary_supervised": test_summary_supervised,
        "dev_summary_all_rows": dev_summary_all,
        "test_summary_all_rows": test_summary_all,
        "best_seed": int(best["seed"]),
        "best_model_path": str((out_dir / "linear_best.joblib").resolve()),
    }
    write_json(out_dir / "linear_training_report.json", report)
    return report


def load_linear_model(model_path: Path) -> Tuple[Any, Any]:
    bundle = joblib.load(model_path)
    return bundle["feature_builder"], bundle["model"]
