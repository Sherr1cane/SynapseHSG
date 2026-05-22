#!/usr/bin/env python3
"""Precompute MLP edge scores for all triples, save to JSON.

Run once: python pipeline/precompute_mlp_scores.py
Output:   dataset/output/mlp_edge_scores.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
pipeline_dir = str(Path(__file__).resolve().parent)
for p in [pipeline_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Register module aliases for pickle
import pipeline.m2hsg.stage2.features as _feat
sys.modules['m2hsg.features'] = _feat
sys.modules['m2hsg.stage2.features'] = _feat

import numpy as np
import torch
import joblib
from pipeline.m2hsg.stage2.mlp import RelationAwareMLP
from constructor import Triple, HSG


def _triple_to_infer_sample(t: Triple) -> dict:
    """Convert a Triple to the dict format expected by MLP feature builder."""
    prov = t.provenance or []
    audio_ct = sum(1 for p in prov if isinstance(p, dict) and p.get("modality") == "audio")
    slide_ct = sum(1 for p in prov if isinstance(p, dict) and p.get("modality") == "slide")
    paper_ct = sum(1 for p in prov if isinstance(p, dict) and p.get("modality") == "paper")
    return {
        "head_type": t.head_type, "head_content": t.head_content,
        "relation": t.relation,
        "tail_type": t.tail_type, "tail_content": t.tail_content,
        "is_repaired": False,
        "modality_mask": {"audio": min(audio_ct, 1), "visual": min(slide_ct, 1), "paper": min(paper_ct, 1)},
        "audio_repr": {"count": audio_ct, "quality_max": 0.0, "duration_sum": 0.0},
        "visual_repr": {"count": slide_ct, "quality_max": 0.0, "has_chart_hint": 0.0},
        "paper_repr": {"count": paper_ct, "quality_max": 0.0, "repaired_count": 0, "similarity_max": 0.0},
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Precompute MLP edge scores")
    ap.add_argument("--data-root", default="hsg_output")
    ap.add_argument("--model-dir", required=True,
                   help="Trained MLP model directory (containing mlp_model.pt and mlp_featurizer.joblib)")
    ap.add_argument("--output", default="dataset/output/mlp_edge_scores.json")
    args = ap.parse_args()

    data_root = args.data_root
    model_dir = args.model_dir
    output_path = args.output

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    print("Loading HSG...")
    hsg = HSG(data_root)
    print(f"  {len(hsg.triples)} triples")

    print(f"Loading MLP from {model_dir}...")
    bundle = joblib.load(f"{model_dir}/mlp_featurizer.joblib")
    fb = bundle["feature_builder"]
    svd = bundle["svd"]
    rel_to_idx = bundle["rel_to_idx"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(f"{model_dir}/mlp_model.pt", map_location=device, weights_only=True)
    emb_w = state.get("rel_emb.weight")
    fc1_w = state.get("fc1.weight")
    rel_emb_dim = emb_w.shape[1] if emb_w is not None else 16
    hidden_dim = fc1_w.shape[0] if fc1_w is not None else 128

    # Get input_dim from dummy
    dummy = fb.transform([{
        "head_type": "Claim", "head_content": "test", "relation": "supported_by",
        "tail_type": "Claim", "tail_content": "test", "is_repaired": False,
        "modality_mask": {"audio": 0, "visual": 0, "paper": 0},
        "audio_repr": {"count": 0, "quality_max": 0.0, "duration_sum": 0.0},
        "visual_repr": {"count": 0, "quality_max": 0.0, "has_chart_hint": 0.0},
        "paper_repr": {"count": 0, "quality_max": 0.0, "repaired_count": 0, "similarity_max": 0.0},
    }])
    input_dim = svd.transform(dummy).shape[1]

    model = RelationAwareMLP.build(
        input_dim=input_dim, rel_count=max(1, len(rel_to_idx)),
        rel_emb_dim=rel_emb_dim, hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  input_dim={input_dim}, device={device}")

    # Batch process all triples
    triples = hsg.triples
    batch_size = 2048
    keys = [(t.head_id, t.relation, t.tail_id) for t in triples]

    print(f"Computing features for {len(triples)} triples...")
    t0 = time.time()

    rows = [_triple_to_infer_sample(t) for t in triples]
    print(f"  Feature dicts: {time.time()-t0:.1f}s")

    t1 = time.time()
    x_sparse = fb.transform(rows)
    print(f"  TF-IDF: {time.time()-t1:.1f}s")

    t2 = time.time()
    x_dense = svd.transform(x_sparse).astype(np.float32)
    print(f"  SVD: {time.time()-t2:.1f}s")

    rel_idx = np.array([rel_to_idx.get(t.relation, 0) for t in triples], dtype=np.int64)

    print(f"Running MLP inference on GPU...")
    t3 = time.time()
    all_scores = []
    for i in range(0, len(x_dense), batch_size):
        xb = torch.from_numpy(x_dense[i:i+batch_size]).float().to(device)
        rb = torch.from_numpy(rel_idx[i:i+batch_size]).long().to(device)
        with torch.no_grad():
            edge_logit, _ = model(xb, rb)
            edge_prob = torch.sigmoid(edge_logit).cpu().numpy()
        all_scores.extend(float(p) for p in edge_prob)
    print(f"  GPU inference: {time.time()-t3:.1f}s")
    print(f"  Total: {time.time()-t0:.1f}s ({len(triples)/(time.time()-t0):.0f} triples/s)")

    # Save
    cache = {}
    for key, score in zip(keys, all_scores):
        cache[f"{key[0]}|{key[1]}|{key[2]}"] = round(score, 6)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved {len(cache)} scores to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
