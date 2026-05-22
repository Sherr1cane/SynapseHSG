#!/usr/bin/env python3
"""SynapseHSG Pipeline — session-scoped multi-channel retrieval + semantic reranking.

Modules:
    constructor.py  — HSG data model, BGE embedding client, subgraph builder
    retriever.py    — Session-scoped multi-channel FAISS retrieval
    reranker.py     — Semantic reranking (channel weight × FAISS similarity)
    decomposer.py   — Fine-tuned Qwen3.5-9B LoRA question decomposition
    evaluator.py    — End-to-end QA evaluation

Usage:
    python -m pipeline --help
    python -m pipeline --max-samples 50
"""

try:
    from constructor import HSG, Triple, Hyperedge, BGEEmbeddingClient
    from retriever import SessionScopedRetriever, get_retrieval_channels
    from reranker import SemanticReranker

    __all__ = [
        "HSG", "Triple", "Hyperedge", "BGEEmbeddingClient",
        "SessionScopedRetriever", "get_retrieval_channels",
        "SemanticReranker",
    ]
except ImportError:
    pass
