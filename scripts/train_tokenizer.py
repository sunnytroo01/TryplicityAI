"""
Train a 32K BPE tokenizer from scratch on Wikipedia samples.

Uses SentencePiece — the same approach as Llama and Mistral.
Run this FIRST before processing training data.
"""

import os
import argparse
import tempfile
from pathlib import Path


def download_sample_text(output_path: str, target_mb: int = 500):
    """Download raw Wikipedia text samples for tokenizer training."""
    from datasets import load_dataset

    print(f"Downloading ~{target_mb}MB of Wikipedia text for tokenizer training...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    written = 0
    target_bytes = target_mb * 1024 * 1024

    with open(output_path, "w", encoding="utf-8") as f:
        for article in ds:
            title = article.get("title", "")
            text = article.get("text", "")
            if not text or len(text) < 100:
                continue
            f.write(f"{title}\n\n{text}\n")
            written += len(text.encode("utf-8"))
            if written >= target_bytes:
                break

    print(f"Wrote {written / 1024 / 1024:.1f}MB of Wikipedia text")


def train_tokenizer(input_path: str, output_dir: str, vocab_size: int = 32000):
    """Train a SentencePiece BPE tokenizer."""
    import sentencepiece as spm

    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "tryplicity")

    print(f"Training {vocab_size}-token BPE tokenizer...")
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity",
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )

    # Verify
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    test = "Hello, this is Tryplicity — a language model trained from scratch!"
    encoded = sp.encode(test, out_type=str)
    print(f"Tokenizer trained: {vocab_size} tokens")
    print(f"Test: '{test}'")
    print(f"Tokens: {encoded}")
    print(f"Token IDs: {sp.encode(test, out_type=int)}")
    print(f"Saved to {model_prefix}.model")


def main():
    parser = argparse.ArgumentParser(description="Train Tryplicity tokenizer")
    parser.add_argument("--output-dir", type=str, default="/workspace/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--sample-mb", type=int, default=500, help="MB of text to sample for training")
    args = parser.parse_args()

    text_path = os.path.join(args.output_dir, "raw_sample.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    download_sample_text(text_path, target_mb=args.sample_mb)
    train_tokenizer(text_path, args.output_dir, vocab_size=args.vocab_size)

    # Clean up raw text to save space
    os.remove(text_path)
    print("Done. Tokenizer is ready.")


if __name__ == "__main__":
    main()
