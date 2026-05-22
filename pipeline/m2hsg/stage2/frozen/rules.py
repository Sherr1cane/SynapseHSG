"""Frozen relation and heuristic scoring rules.

Self-contained module for M2HSG pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_ENTITY_TYPES = {
    "Claim",
    "Method",
    "Dataset",
    "Metric",
    "Result",
    "Limitation",
    "Slide",
    "VisualRegion",
    "PaperChunk",
    "Figure",
    "Table",
    "Utterance",
    "ProsodyEvent",
}

RELATION_CONSTRAINTS = {
    "supported_by": ({"Claim"}, {"Result", "Figure", "Table", "VisualRegion", "PaperChunk"}),
    "compares": ({"Method", "Result"}, {"Method", "Result"}),
    "measured_by": ({"Result"}, {"Metric"}),
    "aligned_to_slide": ({"Utterance", "Claim", "Method", "Result"}, {"Slide", "VisualRegion"}),
    "grounded_in_paper": ({"Claim", "Method", "Result", "Slide", "VisualRegion"}, {"PaperChunk", "Figure", "Table"}),
    "emphasizes": ({"ProsodyEvent"}, {"Claim", "Method", "Result", "Metric"}),
    "referenced_by": ({"Figure", "Table", "VisualRegion", "PaperChunk"}, {"Utterance", "PaperChunk", "Claim", "Method"}),
}

RELATION_EVIDENCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "aligned_to_slide": {"audio": 0.2, "slide": 0.8, "paper": 0.0},
    "emphasizes": {"audio": 0.8, "slide": 0.1, "paper": 0.1},
    "supported_by": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "compares": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "measured_by": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "referenced_by": {"audio": 0.0, "slide": 0.4, "paper": 0.6},
    "grounded_in_paper": {"audio": 0.0, "slide": 0.2, "paper": 0.8},
}

TIER_STRONG_MIN = 0.75
TIER_MEDIUM_MIN = 0.45


def to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def dedupe_provenance_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for p in items:
        if not isinstance(p, dict):
            continue
        try:
            key = json.dumps(p, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def relation_allowed(h_type: str, rel: str, t_type: str) -> bool:
    if rel not in RELATION_CONSTRAINTS:
        return False
    hs, ts = RELATION_CONSTRAINTS[rel]
    return h_type in hs and t_type in ts


def provenance_modality(p: Dict[str, Any]) -> Optional[str]:
    if not isinstance(p, dict):
        return None
    st = str(p.get("source_type", "")).strip().lower()
    if st == "audio":
        return "audio"
    if st in {"slide", "slide_region", "visual_region"}:
        return "slide"
    if st in {"paper", "paper_chunk", "figure", "table"}:
        return "paper"
    if p.get("slide_id") is not None or p.get("region_id") is not None:
        return "slide"
    if p.get("chunk_id") is not None or p.get("page") is not None:
        return "paper"
    return None


def provenance_quality(modality: str, p: Dict[str, Any]) -> float:
    if modality == "audio":
        st = to_float_or_none(p.get("start_time"))
        et = to_float_or_none(p.get("end_time"))
        return 1.0 if (st is not None and et is not None and et > st) else 0.6

    if modality == "slide":
        slide_id = str(p.get("slide_id", "")).strip()
        region_id = str(p.get("region_id", "")).strip()
        bbox = p.get("bbox")
        has_bbox = isinstance(bbox, list) and len(bbox) == 4
        if region_id and has_bbox:
            return 1.0
        if slide_id:
            return 0.6
        return 0.0

    if modality == "paper":
        if bool(p.get("is_repaired")) and str(p.get("repair_method", "")).strip() == "dense_retrieval":
            return 0.9
        chunk_id = str(p.get("chunk_id", "")).strip()
        page = p.get("page")
        bbox = p.get("bbox")
        has_anchor = (page is not None) or (isinstance(bbox, list) and len(bbox) == 4)
        if chunk_id and has_anchor:
            return 1.0
        if str(p.get("source_type", "")).strip() in {"figure", "table"} and has_anchor:
            return 1.0
        if chunk_id or has_anchor:
            return 0.6
        return 0.0
    return 0.0


def compute_heuristic_score_tier(row: Dict[str, Any]) -> Tuple[float, str, bool, List[str]]:
    """Compute frozen V1 provenance score + tier.

    Returns:
      (score, tier, repaired, repaired_granularity_list)
    """
    prov = row.get("provenance", [])
    if not isinstance(prov, list):
        prov = []
    prov = dedupe_provenance_items([p for p in prov if isinstance(p, dict)])

    rel = str(row.get("relation", "")).strip()
    weights = RELATION_EVIDENCE_WEIGHTS.get(rel, {"audio": 0.0, "slide": 0.0, "paper": 0.0})
    qmax = {"audio": 0.0, "slide": 0.0, "paper": 0.0}
    repaired_grans: List[str] = []
    for p in prov:
        m = provenance_modality(p)
        if m not in qmax:
            continue
        q = provenance_quality(m, p)
        if q > qmax[m]:
            qmax[m] = q
        if bool(p.get("is_repaired")):
            g = str(p.get("repaired_granularity", "")).strip()
            if g:
                repaired_grans.append(g)

    score = 0.0
    for m in ["audio", "slide", "paper"]:
        w = float(weights.get(m, 0.0))
        present = 1.0 if qmax[m] > 0 else 0.0
        score += w * present * qmax[m]
    score = max(0.0, min(1.0, score))

    tier = "weak"
    if score >= TIER_STRONG_MIN:
        tier = "strong"
    elif score >= TIER_MEDIUM_MIN:
        tier = "medium"

    repaired = bool(row.get("is_repaired")) or any(bool(p.get("is_repaired")) for p in prov)
    if repaired and "page_level" in set(repaired_grans) and tier == "strong":
        tier = "medium"

    return round(float(score), 4), tier, repaired, repaired_grans


def infer_source_type(row: Dict[str, Any], hard_negative: bool = False) -> Tuple[str, Optional[str]]:
    """Infer source_type for training/rerank diagnostics.

    Returns:
      (source_type, base_tier)
    """
    if hard_negative:
        return "hard_negative", None

    tier = str(row.get("tier", "weak")).strip().lower()
    if tier not in {"strong", "medium", "weak"}:
        tier = "weak"

    repaired = bool(row.get("is_repaired"))
    if repaired:
        return "repaired", tier
    return tier, None

