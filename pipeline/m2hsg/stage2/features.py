"""Feature extraction and sparse design matrix builders."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

AUDIO_GROUP = [0, 1, 2]
VISUAL_GROUP = [3, 4, 5]
PAPER_GROUP = [6, 7, 8, 9]
MASK_GROUP = [10, 11, 12]
REPAIRED_GROUP = [13]
NUMERIC_DIM = 14

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def text_token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def lexical_jaccard(a: str, b: str) -> float:
    sa = text_token_set(a)
    sb = text_token_set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1.0, len(sa | sb))


def parse_numeric_features(sample: Dict[str, Any]) -> np.ndarray:
    audio = sample.get("audio_repr", {}) if isinstance(sample.get("audio_repr", {}), dict) else {}
    visual = sample.get("visual_repr", {}) if isinstance(sample.get("visual_repr", {}), dict) else {}
    paper = sample.get("paper_repr", {}) if isinstance(sample.get("paper_repr", {}), dict) else {}
    mask = sample.get("modality_mask", {}) if isinstance(sample.get("modality_mask", {}), dict) else {}

    return np.array(
        [
            float(audio.get("count", 0.0)),
            float(audio.get("quality_max", 0.0)),
            float(audio.get("duration_sum", 0.0)),
            float(visual.get("count", 0.0)),
            float(visual.get("quality_max", 0.0)),
            float(visual.get("has_chart_hint", 0.0)),
            float(paper.get("count", 0.0)),
            float(paper.get("quality_max", 0.0)),
            float(paper.get("repaired_count", 0.0)),
            float(paper.get("similarity_max", 0.0)),
            float(mask.get("audio", 0.0)),
            float(mask.get("visual", 0.0)),
            float(mask.get("paper", 0.0)),
            1.0 if bool(sample.get("is_repaired")) else 0.0,
        ],
        dtype=np.float32,
    )


def _cross_group(num: np.ndarray, rel_sparse: csr_matrix, feat_indices: Sequence[int]) -> csr_matrix:
    cols = []
    for ridx in range(rel_sparse.shape[1]):
        rcol = rel_sparse[:, ridx]
        for findex in feat_indices:
            vals = num[:, findex].reshape(-1, 1)
            cols.append(rcol.multiply(vals))
    if not cols:
        return csr_matrix((num.shape[0], 0), dtype=np.float32)
    return hstack(cols, format="csr")


class LinearFeatureBuilder:
    """Build relation-aware sparse features for linear baseline."""

    def __init__(self, max_text_features: int = 5000) -> None:
        self.head_vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tail_vectorizer = TfidfVectorizer(
            max_features=max_text_features,
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.rel_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        self._fitted = False

    def fit(self, samples: List[Dict[str, Any]]) -> "LinearFeatureBuilder":
        head_texts = [self._head_text(x) for x in samples]
        tail_texts = [self._tail_text(x) for x in samples]
        relations = np.array([[str(x.get("relation", ""))] for x in samples], dtype=object)
        self.head_vectorizer.fit(head_texts)
        self.tail_vectorizer.fit(tail_texts)
        self.rel_encoder.fit(relations)
        self._fitted = True
        return self

    def transform(self, samples: List[Dict[str, Any]]) -> csr_matrix:
        if not self._fitted:
            raise RuntimeError("LinearFeatureBuilder is not fitted")
        head_texts = [self._head_text(x) for x in samples]
        tail_texts = [self._tail_text(x) for x in samples]
        relations = np.array([[str(x.get("relation", ""))] for x in samples], dtype=object)
        num = np.vstack([parse_numeric_features(x) for x in samples]).astype(np.float32)

        head_x = self.head_vectorizer.transform(head_texts).astype(np.float32)
        tail_x = self.tail_vectorizer.transform(tail_texts).astype(np.float32)
        rel_x = self.rel_encoder.transform(relations).tocsr()
        num_x = csr_matrix(num)

        cross_audio = _cross_group(num, rel_x, AUDIO_GROUP)
        cross_visual = _cross_group(num, rel_x, VISUAL_GROUP)
        cross_paper = _cross_group(num, rel_x, PAPER_GROUP)
        cross_mask = _cross_group(num, rel_x, MASK_GROUP)
        cross_repaired = _cross_group(num, rel_x, REPAIRED_GROUP)

        return hstack(
            [
                head_x,
                tail_x,
                rel_x,
                num_x,
                cross_audio,
                cross_visual,
                cross_paper,
                cross_mask,
                cross_repaired,
            ],
            format="csr",
            dtype=np.float32,
        )

    @staticmethod
    def _head_text(sample: Dict[str, Any]) -> str:
        h_type = str(sample.get("head_type", ""))
        h = str(sample.get("head_content", ""))
        return f"{h_type} {h}".strip()

    @staticmethod
    def _tail_text(sample: Dict[str, Any]) -> str:
        t_type = str(sample.get("tail_type", ""))
        t = str(sample.get("tail_content", ""))
        return f"{t_type} {t}".strip()

