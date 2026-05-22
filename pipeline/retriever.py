#!/usr/bin/env python3
"""Session-scoped multi-channel retriever.

Key optimizations:
1. Session-scoped: only search within the target paper's nodes
2. Similarity scores preserved for downstream reranking
3. O(K) node-to-triple expansion via HSG inverted index
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from constructor import HSG, Triple, BGEEmbeddingClient


class SessionScopedRetriever:
    """Multi-channel retrieval scoped to a single session."""

    def __init__(self, hsg: HSG, embedding_client: BGEEmbeddingClient):
        self.hsg = hsg
        self.embedding_client = embedding_client

    def _faiss_search(self, query_embedding: np.ndarray,
                      top_k: int = 500) -> List[Tuple[str, float]]:
        if self.hsg.embedding_index is None:
            return []
        scores, indices = self.hsg.embedding_index.search(
            query_embedding.reshape(1, -1).astype("float32"), top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                nid = self.hsg.idx_to_node_id.get(str(idx))
                if nid:
                    results.append((nid, float(score)))
        return results

    def retrieve_channel(self, session_id: str, query: str,
                         top_k: int = 50) -> List[Tuple[Triple, str, float]]:
        """Single-channel retrieval within a session.

        Returns list of (triple, query_text, faiss_similarity).
        """
        query_emb = self.embedding_client.encode(query)
        if query_emb is None:
            return []

        # Global search with BGE-M3 (pure, no GNN concat)
        raw = self._faiss_search(query_emb, top_k=1000)
        prefix = session_id + "/"
        session_hits = [(nid, s) for nid, s in raw if nid.startswith(prefix)]
        session_hits = session_hits[:top_k]

        if not session_hits:
            return []

        node_ids = [nid for nid, _ in session_hits]
        node_scores = {nid: s for nid, s in session_hits}

        # GNN re-scoring: boost nodes whose GNN embedding is close to neighbors
        if self.hsg.gnn_dim > 0:
            gnn_boost = self._gnn_rescore(query_emb, node_ids)
            for nid in node_ids:
                bge_sim = node_scores[nid]
                gnn_b = gnn_boost.get(nid, 0.0)
                # Weighted sum: BGE-M3 is primary, GNN is bonus
                node_scores[nid] = bge_sim * (1.0 + 0.15 * gnn_b)
            # Re-sort by combined score
            session_hits = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            node_ids = [nid for nid, _ in session_hits]

        triples = self.hsg.expand_to_triples(node_ids, max_triples=300)

        results = []
        for t in triples:
            best_sim = max(node_scores.get(t.head_id, 0.0),
                           node_scores.get(t.tail_id, 0.0))
            results.append((t, query, best_sim))
        return results

    def _gnn_rescore(self, query_emb: np.ndarray,
                     node_ids: List[str]) -> Dict[str, float]:
        """Compute GNN-based bonus for each node.

        For each retrieved node, compute how similar its GNN embedding is
        to the average GNN embedding of all retrieved nodes (proxy for
        query's graph context). Nodes with GNN embeddings close to the
        cluster center get a bonus.
        """
        gnn_embs = []
        gnn_nids = []
        for nid in node_ids:
            emb = self.hsg.gnn_embeddings.get(nid)
            if emb:
                gnn_embs.append(emb)
                gnn_nids.append(nid)

        if not gnn_embs:
            return {}

        gnn_arr = np.array(gnn_embs, dtype=np.float32)
        # Cluster center = mean of top-K retrieved nodes' GNN embeddings
        center = gnn_arr.mean(axis=0)
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-8:
            return {}

        center = center / center_norm
        # Normalize each GNN embedding
        norms = np.linalg.norm(gnn_arr, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        gnn_arr_norm = gnn_arr / norms

        # Cosine similarity to center
        sims = (gnn_arr_norm @ center).tolist()

        return {nid: sim for nid, sim in zip(gnn_nids, sims)}

    def retrieve_multi_channel(self, session_id: str,
                               channels: List[Dict]) -> List[Tuple[Triple, str, float]]:
        """Multi-channel retrieval, merging all channel results."""
        all_results = []
        seen = set()

        for ch in channels:
            hits = self.retrieve_channel(
                session_id, ch["query"], top_k=ch.get("top_k", 50)
            )
            for triple, _, sim in hits:
                key = (triple, ch["name"])
                if key not in seen:
                    all_results.append((triple, ch["name"], sim))
                    seen.add(key)
            print(f"  Channel [{ch['name']}]: {len(hits)} triples")

        return all_results


def get_retrieval_channels(question: str, decomposition: dict) -> List[Dict]:
    """Build retrieval channels from decomposition output."""
    channels = []
    raw = decomposition.get("constraints", {})
    constraints = raw if isinstance(raw, dict) else {}

    channels.append({
        "name": "question", "query": question,
        "weight": 0.3, "top_k": 50,
    })

    for key in ["baseline_constraint", "metric_constraint",
                "experiment_setting", "section_constraint"]:
        val = constraints.get(key)
        if val:
            channels.append({
                "name": key, "query": str(val),
                "weight": 0.25, "top_k": 30,
            })

    unit = decomposition.get("target_semantic_unit")
    if unit:
        channels.append({
            "name": "target_unit", "query": str(unit),
            "weight": 0.15, "top_k": 20,
        })

    return channels
