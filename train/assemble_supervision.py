#!/usr/bin/env python3
"""Assemble MLP supervision data from HSG output.

Reads kg_triples.jsonl from each session in an HSG run directory,
labels each triple with its heuristic tier, generates hard negatives,
and writes per-split supervision JSONL files for MLP training.

Usage:
    python train/assemble_supervision.py \
        --run-dir hsg_output/run_YYYYMMDD_HHMMSS \
        --output-dir train/supervision
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure pipeline/ is importable
_pipeline_dir = str(Path(__file__).resolve().parent.parent / "pipeline")
if _pipeline_dir not in sys.path:
    sys.path.insert(0, _pipeline_dir)

from m2hsg.stage2.supervision import SupervisionConfig, assemble_supervision


def main():
    p = argparse.ArgumentParser(description="Assemble MLP supervision from HSG output")
    p.add_argument("--run-dir", required=True, help="HSG run directory (contains sessions/ and splits.json)")
    p.add_argument("--output-dir", required=True, help="Output directory for supervision JSONL files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives-per-positive", type=int, default=3)
    p.add_argument("--medium-weight", type=float, default=0.7)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    splits_file = run_dir / "splits.json"
    if not splits_file.exists():
        raise SystemExit(
            f"splits.json not found in {run_dir}.\n"
            "Generate it first: python -m pipeline.build_hsg --root ..."
        )

    cfg = SupervisionConfig(
        run_dir=run_dir,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        negatives_per_positive=args.negatives_per_positive,
        medium_weight=args.medium_weight,
    )

    stats = assemble_supervision(cfg)
    print(f"\nSupervision assembled:")
    for split, count in stats.get("split_rows", {}).items():
        loss_count = stats.get("split_loss_rows", {}).get(split, 0)
        print(f"  {split}: {count} rows ({loss_count} with loss)")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
