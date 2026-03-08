"""
Download and tokenize all training data into binary shards.

Data mix (following SmolLM2 recipe):
  - 90% FineWeb-Edu (high-quality web text)
  - 10% StarCoderData (Python code)

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
    parser.add_argument("--skip-web", action="store_true", help="Skip FineWeb-Edu download")
    parser.add_argument("--skip-code", action="store_true", help="Skip code data download")
    args = parser.parse_args()

    pipeline = DataPipeline(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        seq_len=args.seq_len,
    )

    web_tokens = int(args.total_tokens * 0.9)
    code_tokens = int(args.total_tokens * 0.1)

    if not args.skip_web:
        print(f"\n{'='*60}")
        print(f"Phase 1: FineWeb-Edu ({web_tokens:,} tokens)")
        print(f"{'='*60}\n")
        pipeline.process_fineweb_edu(num_tokens=web_tokens)

    if not args.skip_code:
        print(f"\n{'='*60}")
        print(f"Phase 2: StarCoderData ({code_tokens:,} tokens)")
        print(f"{'='*60}\n")
        pipeline.process_code_data(num_tokens=code_tokens)

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
