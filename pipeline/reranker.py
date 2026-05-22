#!/usr/bin/env python3
"""Semantic reranker with learned MLP (Eq.17-18) or heuristic fallback.

Modes (ablation):
- Default: channel_weight × FAISS similarity × type_bonus (heuristic)
- use_learned_reranker + mlp_mode=filter: heuristic ranking, MLP as threshold filter
- use_learned_reranker + mlp_mode=bonus: heuristic * (1 + beta * mlp_score)
- use_learned_reranker + mlp_mode=blend: alpha * mlp + (1-alpha) * heuristic
- use_hyperedge=True: hyperedge aggregation + modality_bonus
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from constructor import Triple


class LearnableReranker:
    """Load precomputed MLP edge scores from cache file."""

    def __init__(self, cache_path: str = "dataset/output/mlp_edge_scores.json"):
        self._score_cache = {}
        self._load_cache(cache_path)

    def _load_cache(self, cache_path: str):
        if not os.path.exists(cache_path):
            print(f"  WARNING: MLP score cache not found at {cache_path}")
            print(f"  Run: python pipeline/precompute_mlp_scores.py")
            return
        t0 = time.time()
        with open(cache_path) as f:
            raw = json.load(f)
        for key_str, score in raw.items():
            parts = key_str.split("|")
            if len(parts) == 3:
                self._score_cache[tuple(parts)] = score
        print(f"  Loaded {len(self._score_cache)} cached MLP scores from {cache_path} ({time.time()-t0:.1f}s)")

    def predict(self, triples: List[Triple]) -> List[float]:
        if not triples:
            return []
        return [self._score_cache.get((t.head_id, t.relation, t.tail_id), 0.5) for t in triples]


class SemanticReranker:
    """Rerank triples by combining channel weights with FAISS similarity."""

    def __init__(self, type_bonus: float = 1.2, modality_bonus: float = 0.1,
                 learned_reranker: Optional[LearnableReranker] = None,
                 mlp_mode: str = "blend"):
        self.type_bonus = type_bonus
        self.modality_bonus = modality_bonus
        self.learned = learned_reranker
        self.mlp_mode = mlp_mode

    def rerank(
        self,
        triples_with_scores: List[Tuple[Triple, str, float]],
        channels: List[Dict],
        decomposition: dict,
        use_hyperedge: bool = False,
    ) -> List[Tuple[Triple, float]]:
        ch_weights = {ch["name"]: ch["weight"] for ch in channels}
        target_unit = decomposition.get("target_semantic_unit", "")

        # Deduplicate triples, keep best channel similarity per (triple, channel)
        triple_channel_best: Dict[Tuple, Dict[str, float]] = defaultdict(lambda: {})
        for triple, ch_name, sim in triples_with_scores:
            key = (triple.head_id, triple.relation, triple.tail_id)
            prev = triple_channel_best[key].get(ch_name, 0.0)
            if sim > prev:
                triple_channel_best[key][ch_name] = sim

        triple_map = {}
        for triple, ch_name, sim in triples_with_scores:
            key = (triple.head_id, triple.relation, triple.tail_id)
            triple_map[key] = triple

        # Heuristic scores (always computed)
        heuristic_scores: Dict[Tuple, float] = {}
        for key, ch_sims in triple_channel_best.items():
            triple = triple_map[key]
            total = sum(ch_weights.get(cn, 0.1) * s for cn, s in ch_sims.items())
            if target_unit and triple.head_type == target_unit:
                total *= self.type_bonus
            heuristic_scores[key] = total

        # Apply MLP scores if available
        if self.learned is not None:
            unique_triples = [triple_map[k] for k in triple_channel_best]
            mlp_scores = self.learned.predict(unique_triples)
            key_list = list(triple_channel_best.keys())

            if self.mlp_mode == "filter":
                mlp_threshold = 0.3
                scored = []
                for key, mlp_s in zip(key_list, mlp_scores):
                    if mlp_s >= mlp_threshold:
                        scored.append((triple_map[key], heuristic_scores[key]))
                if not scored:
                    scored = [(triple_map[k], s) for k, s in sorted(
                        heuristic_scores.items(), key=lambda x: x[1], reverse=True)[:5]]
            elif self.mlp_mode == "bonus":
                beta = 0.2
                scored = []
                for key, mlp_s in zip(key_list, mlp_scores):
                    heur_s = heuristic_scores[key]
                    combined = heur_s * (1 + beta * mlp_s)
                    scored.append((triple_map[key], combined))
            else:  # blend
                alpha = 0.6
                scored = []
                for key, mlp_s in zip(key_list, mlp_scores):
                    heur_s = heuristic_scores[key]
                    combined = alpha * mlp_s + (1 - alpha) * heur_s
                    scored.append((triple_map[key], combined))
        else:
            scored = [(triple_map[k], s) for k, s in heuristic_scores.items()]

        if not use_hyperedge:
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        # Hyperedge-aware reranking
        center_groups: Dict[str, List[Tuple[Triple, float]]] = defaultdict(list)
        for triple, score in scored:
            center_groups[triple.head_id].append((triple, score))

        center_scores = {}
        for center_id, members in center_groups.items():
            base = sum(s for _, s in members)
            tail_types = set(t.tail_type for t, _ in members)
            bonus = 1.0 + self.modality_bonus * max(len(tail_types) - 1, 0)
            center_scores[center_id] = base * bonus

        sorted_centers = sorted(center_scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        seen = set()
        for center_id, he_score in sorted_centers:
            members = center_groups[center_id]
            members.sort(key=lambda x: x[1], reverse=True)
            for triple, _ in members:
                key = (triple.head_id, triple.relation, triple.tail_id)
                if key not in seen:
                    seen.add(key)
                    result.append((triple, he_score))

        return result
