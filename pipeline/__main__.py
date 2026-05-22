#!/usr/bin/env python3
"""Pipeline CLI entry point.

Usage:
    # End-to-end: raw data → answers
    python -m pipeline \\
      --raw-data /path/to/raw_data \\
      --golden-data dataset/gold_test.jsonl \\
      --decomposer-path models/decomposer_qwen35_9b/final \\
      --use-hyperedge-anchor --use-linearize

    # On existing HSG output
    python -m pipeline --max-samples 50
"""

from pipeline.run_pipeline import main

if __name__ == "__main__":
    main()
