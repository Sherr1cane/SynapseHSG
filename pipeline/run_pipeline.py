#!/usr/bin/env python3
"""Unified SynapseHSG Pipeline entry point.

Stages:
  1. HSG Construction — raw data (PDF/slides/audio) → kg_triples.jsonl
  2. Load HSG — kg_triples → in-memory graph + FAISS index
  3. Retrieve — session-scoped multi-channel retrieval
  4. Rerank — semantic reranking
  5. Generate — LLM answer generation
  6. Evaluate — LLM-based quality evaluation

Usage:
    # End-to-end: raw data → answers (requires .env with API endpoints)
    python -m pipeline \\
      --raw-data /path/to/raw_data \\
      --golden-data dataset/gold_test.jsonl \\
      --use-hyperedge-anchor --use-linearize

    # Skip HSG construction, evaluate on existing HSG output
    python -m pipeline --max-samples 50

    # Only evaluation, no generation
    python -m pipeline --eval-only --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Disable proxy for local APIs
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# Ensure pipeline/ is on sys.path for sibling module imports (constructor, retriever, etc.)
_pipeline_dir = str(Path(__file__).resolve().parent)
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)


def main():
    p = argparse.ArgumentParser(description="SynapseHSG Pipeline")
    p.add_argument("--raw-data", default=None,
                   help="Raw data root for HSG construction (conf/session/ dirs). "
                        "If set, runs HSG construction first, then inference.")
    p.add_argument("--data-root", default="hsg_output",
                   help="HSG output directory (used if --raw-data is not set, "
                        "or as output path after HSG construction)")
    p.add_argument("--embedding-dir", default="dataset/output/embeddings",
                   help="FAISS embedding index directory")
    p.add_argument("--no-decomposer", action="store_true",
                   help="Skip decomposer, use raw question for retrieval")
    p.add_argument("--decomposer-path", required=True,
                   help="Decomposer LoRA model path")
    p.add_argument("--golden-data", default="dataset/gold_test.jsonl",
                   help="Test data for evaluation")
    p.add_argument("--output", default="pipeline/eval_results.json",
                   help="Output path")
    p.add_argument("--max-samples", type=int, default=50,
                   help="Max samples to evaluate")

    # API endpoints
    p.add_argument("--gen-api", default="http://localhost:8000/v1")
    p.add_argument("--gen-key",
                   default="your_api_key")
    p.add_argument("--gen-model", default="Qwen3.6-27B",
                   help="Generation model name")
    p.add_argument("--eval-api", default="http://localhost:8000/v1")
    p.add_argument("--eval-key",
                   default="your_api_key")
    p.add_argument("--embed-api", default="http://localhost:8001")
    p.add_argument("--embed-key", default="sk-000")

    # M2HSG control
    p.add_argument("--learned-scores", default=None,
                   help="Custom MLP score cache file (default: dataset/output/mlp_edge_scores.json)")
    p.add_argument("--eval-only", action="store_true",
                   help="Only run evaluation on existing results")
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip LLM evaluation, only generate answers")

    # Reranker params
    p.add_argument("--type-bonus", type=float, default=1.2)
    p.add_argument("--use-hyperedge", action="store_true",
                   help="Enable hyperedge-aware reranking and context (ablation)")
    p.add_argument("--use-learned-reranker", action="store_true",
                   help="Use trained MLP reranker (ablation)")
    p.add_argument("--mlp-mode", default="blend", choices=["blend", "filter", "bonus"],
                   help="How to use MLP scores: blend, filter, or bonus")
    p.add_argument("--use-linearize", action="store_true",
                   help="Use graph-linearized context instead of flat text (ablation, Eq.21)")
    p.add_argument("--use-hyperedge-anchor", action="store_true",
                   help="Use hyperedge as topological expansion anchor (ablation)")
    p.add_argument("--no-hsg", action="store_true",
                   help="Skip HSG, use raw text vector retrieval (ablation)")
    p.add_argument("--rerank-top-k", type=int, default=10,
                   help="Number of top triples after reranking (default: 10)")
    p.add_argument("--use-gnn", action="store_true",
                   help="Use GNN-enhanced node embeddings (ablation, Eq.8-10)")
    p.add_argument("--gnn-embeddings", default="dataset/output/gnn_node_embeddings.json",
                   help="Precomputed GNN embeddings path")

    args = p.parse_args()

    print("=" * 60)
    print("SYNAPSEHSG PIPELINE")
    print("=" * 60)

    # Stage 1: HSG Construction (if --raw-data is provided)
    data_root = args.data_root
    if args.raw_data:
        from m2hsg.config import Config as HSGConfig
        from m2hsg.llm_client import shutdown_all_llm_executors
        from build_hsg import run_pipeline as run_hsg_build

        print(f"\n[Stage 1] Building HSG from raw data: {args.raw_data}")
        hsg_cfg = HSGConfig(
            root=Path(args.raw_data).resolve(),
            output_root=Path(args.data_root).resolve(),
            seed=42, limit=None, random_sample=None,
            split_train=0.7, split_dev=0.15, split_test=0.15,
            topk_tags=2, topk_paper=5,
            max_images_per_call=1, max_image_bytes=400000,
            prune_missing_audio=True, enable_llm=True,
            require_asr=False, resume_run_dir=None,
        )
        try:
            run_dir = run_hsg_build(hsg_cfg)
        finally:
            shutdown_all_llm_executors()
        data_root = str(run_dir)
        print(f"[Stage 1] Done. HSG output: {data_root}")
    else:
        print(f"\n[Stage 1] Skipping HSG construction (--raw-data not set)")
        print(f"  Using existing HSG data: {data_root}")

    # Stage 2: Load HSG
    print("\n[Stage 2] Loading HSG...")
    from constructor import HSG, BGEEmbeddingClient
    from retriever import SessionScopedRetriever, get_retrieval_channels
    from reranker import SemanticReranker, LearnableReranker
    from evaluator import Evaluator, Decomposer, RawTextRetriever

    embed_client = BGEEmbeddingClient(args.embed_api, args.embed_key)

    raw_retriever = None
    if args.no_hsg:
        print("  Skipping HSG (--no-hsg), using raw text retrieval")
        hsg = None
        raw_retriever = RawTextRetriever(embed_client, data_root)
    else:
        gnn_path = args.gnn_embeddings if args.use_gnn else None
        hsg = HSG(data_root, embedding_dir=args.embedding_dir,
                  gnn_embeddings_path=gnn_path)

    # Stage 3: Load Decomposer
    if args.no_decomposer:
        print("\n[Stage 3] Skipping decomposer (--no-decomposer)")
        decomposer = None
    else:
        print(f"\n[Stage 3] Loading decomposer from {args.decomposer_path}...")
        decomposer = Decomposer(args.decomposer_path)

    if embed_client is None:
        return

    if args.no_hsg:
        retriever = None
        reranker = None
    else:
        retriever = SessionScopedRetriever(hsg, embed_client)

        learned = None
        if args.use_learned_reranker:
            scores_path = args.learned_scores or "dataset/output/mlp_edge_scores.json"
            print(f"\n[Stage 3b] Loading precomputed MLP reranker scores from {scores_path}...")
            learned = LearnableReranker(scores_path)

        reranker = SemanticReranker(type_bonus=args.type_bonus, learned_reranker=learned,
                                    mlp_mode=args.mlp_mode)

    evaluator = Evaluator(
        hsg, decomposer, retriever, reranker,
        f"{args.gen_api}/chat/completions", args.gen_key,
        f"{args.eval_api}/chat/completions", args.eval_key,
        use_hyperedge=args.use_hyperedge,
        use_linearize=args.use_linearize,
        use_hyperedge_anchor=args.use_hyperedge_anchor,
        gen_model=args.gen_model,
        no_hsg=args.no_hsg,
        raw_retriever=raw_retriever,
        skip_eval=args.skip_eval,
        rerank_top_k=args.rerank_top_k,
    )
    modes = []
    if args.use_hyperedge:
        modes.append("Hyperedge")
    if args.use_learned_reranker:
        modes.append(f"LearnedReranker({args.mlp_mode})")
    if args.use_linearize:
        modes.append("Linearize")
    if args.use_hyperedge_anchor:
        modes.append("HyperedgeAnchor")
    if args.use_gnn:
        modes.append("GNN")
    if modes:
        print(f"  [Ablation modes: {' + '.join(modes)}]")

    evaluator.evaluate_dataset(
        args.golden_data, max_samples=args.max_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
