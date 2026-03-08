"""
Download and tokenize all of English Wikipedia (pre-2022) into binary shards.

Data source:
  - 100% English Wikipedia (20220301 snapshot)
  - 6.5M+ articles, ~4B tokens
  - Full encyclopedic knowledge across all domains

Run AFTER train_tokenizer.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tryplicity.data import DataPipeline


def main():
    parser = argparse.ArgumentParser(description="Prepare Tryplicity training data")
    parser.add_argument("--output-dir", type=str, default="/workspace/data/processed")
    parser.add_argument("--tokenizer", type=str, default="/workspace/tokenizer/tryplicity.model")
    parser.add_argument("--total-tokens", type=int, default=20_000_000_000, help="Total tokens to collect (default 20B)")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    pipeline = DataPipeline(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        seq_len=args.seq_len,
    )

    print(f"\n{'='*60}")
    print(f"  English Wikipedia (pre-2022 snapshot)")
    print(f"  Target: {args.total_tokens:,} tokens")
    print(f"{'='*60}\n")
    pipeline.process_wikipedia(num_tokens=args.total_tokens)

    # Summary
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")

    train_size = sum(os.path.getsize(f) for f in (os.path.join(train_dir, f) for f in os.listdir(train_dir)) if os.path.isfile(f)) if os.path.exists(train_dir) else 0
    val_size = sum(os.path.getsize(f) for f in (os.path.join(val_dir, f) for f in os.listdir(val_dir)) if os.path.isfile(f)) if os.path.exists(val_dir) else 0

    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"  Train: {train_size / 1024**3:.1f} GB")
    print(f"  Val:   {val_size / 1024**3:.1f} GB")
    print(f"  Total: {(train_size + val_size) / 1024**3:.1f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
