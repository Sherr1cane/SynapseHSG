#!/usr/bin/env python3
"""Train MLP reranker for M2HSG edge quality scoring.

Usage:
    python train/train_mlp.py \
        --supervision-dir hsg_output/_m2hsg_stage1/supervision/ \
        --output-dir models/mlp/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure pipeline/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from m2hsg.stage2.mlp import MLPTrainConfig, train_mlp


def main():
    p = argparse.ArgumentParser(description="Train MLP reranker")
    p.add_argument("--supervision-dir", required=True)
    p.add_argument("--output-dir", default="models/mlp_full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rel-emb-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--max-text-features", type=int, default=500)
    p.add_argument("--svd-dim", type=int, default=64)
    args = p.parse_args()

    cfg = MLPTrainConfig(
        supervision_dir=Path(args.supervision_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rel_emb_dim=args.rel_emb_dim,
        hidden_dim=args.hidden_dim,
        max_text_features=args.max_text_features,
        svd_dim=args.svd_dim,
    )
    report = train_mlp(cfg)
    print(f"MLP training complete: {report.get('status', 'unknown')}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
