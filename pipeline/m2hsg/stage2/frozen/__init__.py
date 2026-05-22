"""Frozen heuristic snapshot for M2HSG.

Self-contained rule module — no dependency on build scripts.
"""

from .rules import (
    ALLOWED_ENTITY_TYPES,
    RELATION_CONSTRAINTS,
    RELATION_EVIDENCE_WEIGHTS,
    TIER_MEDIUM_MIN,
    TIER_STRONG_MIN,
    compute_heuristic_score_tier,
    provenance_modality,
    provenance_quality,
    relation_allowed,
)

__all__ = [
    "ALLOWED_ENTITY_TYPES",
    "RELATION_CONSTRAINTS",
    "RELATION_EVIDENCE_WEIGHTS",
    "TIER_MEDIUM_MIN",
    "TIER_STRONG_MIN",
    "compute_heuristic_score_tier",
    "provenance_modality",
    "provenance_quality",
    "relation_allowed",
]

