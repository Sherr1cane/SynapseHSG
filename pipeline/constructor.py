#!/usr/bin/env python3
"""HSG data model, embedding client, and subgraph builder.

Fully self-contained — no imports from stage2/scripts/.

Key improvements over the old implementation:
- Inverted index for O(K) node-to-triple lookup (was O(N*M))
- Session-scoped node index built at load time
- No duplicate decomposer calls (decomposition passed in, not re-derived)
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


# ---------------------------------------------------------------------------
# Embedding client (self-contained copy)
# ---------------------------------------------------------------------------

class BGEEmbeddingClient:
    """BGE-M3 Embedding API client."""

    def __init__(self, api_base: str, api_key: str, model: str = "bge-m3"):
        import requests
        self.requests = requests
        self.api_url = f"{api_base.rstrip('/')}/v1/embeddings"
        self.api_key = api_key
        self.model = model
        self.cache = {}

    def encode(self, text: str) -> Optional[np.ndarray]:
        if text in self.cache:
            return self.cache[text]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            resp = self.requests.post(
                self.api_url, headers=headers,
                json={"model": self.model, "input": text[:4000]},
                timeout=30,
            )
            if resp.status_code == 200:
                vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self.cache[text] = vec
                return vec
        except Exception as e:
            print(f"  Embedding error: {e}")
        return None

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            for t in batch:
                batch_results.append(self.encode(t))
            results.extend(batch_results)
        return results


# ---------------------------------------------------------------------------
# Triple data model
# ---------------------------------------------------------------------------

@dataclass
class Triple:
    head_id: str
    head_type: str
    head_content: str
    relation: str
    tail_id: str
    tail_type: str
    tail_content: str
    weight: float
    provenance: list
    tier: str

    def __hash__(self):
        return hash((self.head_id, self.relation, self.tail_id))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.head_id == other.head_id and
                self.relation == other.relation and
                self.tail_id == other.tail_id)


@dataclass
class Hyperedge:
    center_node_id: str
    center_node_type: str
    center_content: str
    member_triples: list
    tail_types: set
    relations: set
    modality_count: int
    evidence_summary: dict  # {"paper_chunks": N, "slides": N, "visual_regions": N, "utterances": N, "prosody_events": N}


# ---------------------------------------------------------------------------
# HSG — Hypergraph Semantic Graph
# ---------------------------------------------------------------------------

class HSG:
    """HSG with inverted index for fast node-to-triple lookup."""

    def __init__(self, data_root: str, embedding_dir: str = None,
                 gnn_embeddings_path: str = None):
        self.triples: List[Triple] = []
        self.triples_by_relation: Dict[str, List[Triple]] = defaultdict(list)
        self.triples_by_node: Dict[str, List[Triple]] = defaultdict(list)
        self.session_ids: set = set()
        self.session_node_ids: Dict[str, set] = defaultdict(set)
        self.evidence_links: Dict[str, dict] = {}  # node_id -> evidence dict

        # Embedding index
        self.embedding_index = None
        self.node_id_to_idx: Dict[str, int] = {}
        self.idx_to_node_id: Dict[str, str] = {}
        self.node_contents: Dict[str, str] = {}
        self.gnn_embeddings: Dict[str, list] = {}  # node_id -> 128-dim

        self._load_triples(data_root)
        self._load_evidence_links(data_root)
        if embedding_dir and os.path.exists(embedding_dir):
            self._load_embeddings(embedding_dir)
        if gnn_embeddings_path and os.path.exists(gnn_embeddings_path):
            self._load_gnn_embeddings(gnn_embeddings_path)

    # -- Loading -----------------------------------------------------------

    def _load_triples(self, data_root: str):
        print(f"Loading HSG from {data_root}...")
        triple_files = self._find_triple_files(data_root)
        print(f"Found {len(triple_files)} kg_triples files")

        for path in triple_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            td = json.loads(line)
                            t = self._parse_triple(td)
                            self._add_triple(t)
                        except Exception:
                            continue
            except Exception as e:
                print(f"  Warning: {path}: {e}")

        print(f"Loaded {len(self.triples)} triples from {len(self.session_ids)} sessions")

    @staticmethod
    def _find_triple_files(data_root: str) -> List[str]:
        files = []
        for root, _, names in os.walk(data_root):
            for name in names:
                if name == "kg_triples.jsonl":
                    files.append(os.path.join(root, name))
        if not files:
            for root, _, names in os.walk(data_root):
                for name in names:
                    if name == "kg_triples_trainable.jsonl":
                        files.append(os.path.join(root, name))
        return files

    @staticmethod
    def _parse_triple(td: dict) -> Triple:
        return Triple(
            head_id=td["head"]["id"],
            head_type=td["head"].get("type", "Claim"),
            head_content=td["head"]["content"],
            relation=td["relation"],
            tail_id=td["tail"]["id"],
            tail_type=td["tail"].get("type", "Claim"),
            tail_content=td["tail"]["content"],
            weight=td.get("weight", 1.0),
            provenance=td.get("provenance", []),
            tier=td.get("tier", "weak"),
        )

    def _add_triple(self, t: Triple):
        self.triples.append(t)
        self.triples_by_relation[t.relation].append(t)
        self.triples_by_node[t.head_id].append(t)
        self.triples_by_node[t.tail_id].append(t)
        for nid in (t.head_id, t.tail_id):
            parts = nid.split("/")
            if len(parts) >= 2:
                sid = "/".join(parts[:2])
                self.session_ids.add(sid)
                self.session_node_ids[sid].add(nid)

    # -- Evidence links / hyperedges ---------------------------------------

    def _load_evidence_links(self, data_root: str):
        link_files = []
        for root, _, names in os.walk(data_root):
            for name in names:
                if name == "evidence_links.json":
                    link_files.append(os.path.join(root, name))
        if not link_files:
            return
        count = 0
        for path in link_files:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for item in data.get("node_evidence", []):
                    nid = item.get("node_id", "")
                    if nid:
                        self.evidence_links[nid] = item
                        count += 1
            except Exception:
                continue
        print(f"Loaded {count} evidence links from {len(link_files)} files")

    def get_hyperedge(self, node_id: str) -> Optional[Hyperedge]:
        ev = self.evidence_links.get(node_id)
        if not ev:
            return None
        triples = self.triples_by_node.get(node_id, [])
        tail_types = set()
        relations = set()
        for t in triples:
            tail_types.add(t.tail_type)
            relations.add(t.relation)
        evidence_summary = {}
        for key in ("paper_chunks", "slides", "visual_regions", "utterances", "prosody_events"):
            val = ev.get(key)
            evidence_summary[key] = len(val) if isinstance(val, list) else 0
        modality_count = sum(1 for v in evidence_summary.values() if v > 0)
        return Hyperedge(
            center_node_id=node_id,
            center_node_type=ev.get("node_type", ""),
            center_content=ev.get("content", ""),
            member_triples=triples,
            tail_types=tail_types,
            relations=relations,
            modality_count=modality_count,
            evidence_summary=evidence_summary,
        )

    # -- Embeddings --------------------------------------------------------

    def _load_embeddings(self, embedding_dir: str):
        try:
            import faiss
        except ImportError:
            print("  Warning: faiss not installed, skipping embeddings")
            return

        print(f"Loading embeddings from {embedding_dir}...")
        index_path = os.path.join(embedding_dir, "node_embeddings.faissindex")
        if os.path.exists(index_path):
            self.embedding_index = faiss.read_index(index_path)
            print(f"  FAISS index: {self.embedding_index.ntotal} vectors")
        else:
            npy = os.path.join(embedding_dir, "node_embeddings.npy")
            if os.path.exists(npy):
                arr = np.load(npy)
                self.embedding_index = faiss.IndexFlatIP(arr.shape[1])
                self.embedding_index.add(arr.astype("float32"))
                print(f"  FAISS from numpy: {self.embedding_index.ntotal} vectors")

        for filename, attr in [
            ("node_id_to_index.json", "node_id_to_idx"),
            ("index_to_node_id.json", "idx_to_node_id"),
            ("node_contents.json", "node_contents"),
        ]:
            p = os.path.join(embedding_dir, filename)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    setattr(self, attr, json.load(f))
        print(f"  {len(self.node_id_to_idx)} node mappings")

    def _load_gnn_embeddings(self, gnn_path: str):
        """Load precomputed GNN embeddings for re-scoring."""
        print(f"Loading GNN embeddings from {gnn_path}...")
        t0 = time.time()
        with open(gnn_path) as f:
            raw = json.load(f)
        for nid, emb in raw.items():
            self.gnn_embeddings[nid] = emb
        print(f"  Loaded {len(self.gnn_embeddings)} GNN embeddings ({time.time()-t0:.1f}s)")

    @property
    def gnn_dim(self):
        if not self.gnn_embeddings:
            return 0
        return len(next(iter(self.gnn_embeddings.values())))

    # -- Retrieval helpers -------------------------------------------------

    def expand_to_triples(self, node_ids: List[str],
                          max_triples: int = 200) -> List[Triple]:
        """O(K) expansion via inverted index."""
        triples = []
        seen = set()
        for nid in node_ids:
            for t in self.triples_by_node.get(nid, []):
                if t not in seen:
                    triples.append(t)
                    seen.add(t)
                if len(triples) >= max_triples:
                    return triples
        return triples

    def get_session_triples(self, session_id: str) -> List[Triple]:
        """Return all triples belonging to a session."""
        node_ids = self.session_node_ids.get(session_id, set())
        if not node_ids:
            return []
        triples = []
        seen = set()
        for nid in node_ids:
            for t in self.triples_by_node.get(nid, []):
                if t not in seen:
                    triples.append(t)
                    seen.add(t)
        return triples

    def expand_hyperedge_anchors(self, node_ids: List[str],
                                 max_extra_triples: int = 15) -> List[Triple]:
        """Hyperedge anchor expansion: for each node, pull in cross-modal triples.

        For each node in node_ids, look up its evidence_links to find related
        nodes (slides, utterances, paper_chunks sharing the same hyperedge).
        Then expand those nodes to triples and return the extras.
        """
        extra = []
        seen = set()
        existing = set(node_ids)

        for nid in node_ids:
            ev = self.evidence_links.get(nid)
            if not ev:
                continue

            related = set()
            for chunk in ev.get("paper_chunks", []):
                cid = chunk.get("chunk_id", "")
                if cid and cid not in existing:
                    related.add(cid)
            for slide in ev.get("slides", []):
                sid = slide.get("slide_id", "")
                if sid and sid not in existing:
                    related.add(sid)
            for utt in ev.get("utterances", []):
                uid = utt.get("utterance_id", "")
                if uid and uid not in existing:
                    related.add(uid)
            for region in ev.get("visual_regions", []):
                rid = region.get("region_id", "")
                if rid and rid not in existing:
                    related.add(rid)

            for rid in related:
                for t in self.triples_by_node.get(rid, []):
                    if t not in seen:
                        extra.append(t)
                        seen.add(t)
                    if len(extra) >= max_extra_triples:
                        return extra
        return extra

    def build_subgraph(self, scored_triples: List[Tuple[Triple, float]],
                       decomposition: dict, use_hyperedge: bool = False) -> dict:
        selected = [t for t, _ in scored_triples]
        nodes = {}
        edges = []
        for t in selected:
            if t.head_id not in nodes:
                nodes[t.head_id] = {
                    "id": t.head_id, "type": t.head_type,
                    "content": t.head_content, "modalities": [],
                }
            if t.tail_id not in nodes:
                nodes[t.tail_id] = {
                    "id": t.tail_id, "type": t.tail_type,
                    "content": t.tail_content, "modalities": [],
                }
            edges.append({
                "source": t.head_id, "target": t.tail_id,
                "relation": t.relation, "weight": t.weight, "tier": t.tier,
            })

        total_score = sum(s for _, s in scored_triples[:10])
        result = {
            "nodes": list(nodes.values()),
            "edges": edges,
            "triples": len(selected),
            "score": total_score / max(len(scored_triples), 1),
            "decomposition": decomposition,
            "metadata": {"num_nodes": len(nodes), "num_edges": len(edges)},
        }

        if use_hyperedge:
            hyperedges = []
            seen_centers = set()
            for t in selected:
                center_id = t.head_id
                if center_id not in seen_centers:
                    seen_centers.add(center_id)
                    he = self.get_hyperedge(center_id)
                    if he and he.modality_count >= 2:
                        hyperedges.append({
                            "center_id": he.center_node_id,
                            "center_type": he.center_node_type,
                            "center_content": he.center_content,
                            "modality_count": he.modality_count,
                            "evidence_summary": he.evidence_summary,
                            "relations": list(he.relations),
                            "tail_types": list(he.tail_types),
                        })
            result["hyperedges"] = hyperedges

        return result
