"""Evaluation and gate logic for M2HSG stage-1."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List

from sklearn.metrics import average_precision_score, f1_score

TIER_ORDER = ["weak", "medium", "strong"]
TIER_TO_LABEL = {"weak": 0, "medium": 1, "strong": 2}


def _safe_div(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def edge_binary_label(row: Dict[str, Any]) -> int:
    st = str(row.get("source_type", "weak")).strip().lower()
    if st in {"strong", "medium", "repaired"}:
        base_tier = str(row.get("base_tier", "")).strip().lower()
        if st == "repaired" and base_tier:
            return 1 if base_tier in {"strong", "medium"} else 0
        return 1
    return 0


def tier_true_label(row: Dict[str, Any]) -> int:
    return int(row.get("tier_label", 0))


def evaluate_predictions(
    rows: List[Dict[str, Any]],
    learned_edge_scores: List[float],
    learned_tier_labels: List[int],
    heuristic_edge_scores: List[float],
    heuristic_tier_labels: List[int],
) -> Dict[str, float]:
    y_edge = [edge_binary_label(x) for x in rows]
    y_tier = [tier_true_label(x) for x in rows]

    learned_edge_pr = float(average_precision_score(y_edge, learned_edge_scores)) if len(set(y_edge)) > 1 else 0.0
    heuristic_edge_pr = float(average_precision_score(y_edge, heuristic_edge_scores)) if len(set(y_edge)) > 1 else 0.0

    learned_tier_f1 = float(f1_score(y_tier, learned_tier_labels, average="macro"))
    heuristic_tier_f1 = float(f1_score(y_tier, heuristic_tier_labels, average="macro"))

    return {
        "learned_edge_pr_auc": learned_edge_pr,
        "heuristic_edge_pr_auc": heuristic_edge_pr,
        "learned_tier_macro_f1": learned_tier_f1,
        "heuristic_tier_macro_f1": heuristic_tier_f1,
        "edge_pr_auc_rel_impr": _safe_div(learned_edge_pr - heuristic_edge_pr, max(1e-8, abs(heuristic_edge_pr))),
        "tier_macro_f1_rel_impr": _safe_div(learned_tier_f1 - heuristic_tier_f1, max(1e-8, abs(heuristic_tier_f1))),
    }


def summarize_seed_metrics(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not seed_rows:
        return {"count": 0, "mean": {}, "std": {}}

    keys = [
        "learned_edge_pr_auc",
        "heuristic_edge_pr_auc",
        "learned_tier_macro_f1",
        "heuristic_tier_macro_f1",
        "edge_pr_auc_rel_impr",
        "tier_macro_f1_rel_impr",
    ]
    mean_map = {k: mean(float(x[k]) for x in seed_rows) for k in keys}
    std_map = {k: pstdev(float(x[k]) for x in seed_rows) for k in keys}
    return {"count": len(seed_rows), "mean": mean_map, "std": std_map}


def compute_silver_gate_noninferiority(summary: Dict[str, Any], tolerance: float = 0.005) -> Dict[str, Any]:
    m = summary.get("mean", {}) if isinstance(summary.get("mean", {}), dict) else {}
    learned_edge = float(m.get("learned_edge_pr_auc", 0.0))
    heuristic_edge = float(m.get("heuristic_edge_pr_auc", 0.0))
    learned_tier = float(m.get("learned_tier_macro_f1", 0.0))
    heuristic_tier = float(m.get("heuristic_tier_macro_f1", 0.0))

    edge_margin = learned_edge - heuristic_edge
    tier_margin = learned_tier - heuristic_tier
    eps = 1e-12
    passed = edge_margin >= (-tolerance - eps) and tier_margin >= (-tolerance - eps)
    return {
        "tolerance": tolerance,
        "edge_margin": edge_margin,
        "tier_margin": tier_margin,
        "learned_edge_pr_auc_mean": learned_edge,
        "heuristic_edge_pr_auc_mean": heuristic_edge,
        "learned_tier_macro_f1_mean": learned_tier,
        "heuristic_tier_macro_f1_mean": heuristic_tier,
        "pass": passed,
        "rule": "silver_non_inferiority_both_metrics",
    }


def gate_from_linear_report(linear_report: Dict[str, Any], tolerance: float = 0.005) -> Dict[str, Any]:
    dev_supervised = (
        linear_report.get("dev_summary_supervised", {})
        if isinstance(linear_report.get("dev_summary_supervised", {}), dict)
        else {}
    )
    dev_all_rows = (
        linear_report.get("dev_summary_all_rows", {})
        if isinstance(linear_report.get("dev_summary_all_rows", {}), dict)
        else {}
    )

    # Backward compatibility for old reports that only expose dev_summary.
    if not dev_supervised:
        dev_supervised = linear_report.get("dev_summary", {}) if isinstance(linear_report.get("dev_summary", {}), dict) else {}
    if not dev_all_rows:
        dev_all_rows = linear_report.get("dev_summary", {}) if isinstance(linear_report.get("dev_summary", {}), dict) else {}

    gate_supervised = compute_silver_gate_noninferiority(dev_supervised, tolerance=tolerance)
    gate_all_rows = compute_silver_gate_noninferiority(dev_all_rows, tolerance=tolerance)
    return {
        "decision_basis": "include_in_loss_only",
        "gate": gate_supervised,
        "gate_margin_supervised": gate_supervised,
        "gate_margin_all_rows": gate_all_rows,
        "dev_summary_supervised": dev_supervised,
        "dev_summary_all_rows": dev_all_rows,
    }
