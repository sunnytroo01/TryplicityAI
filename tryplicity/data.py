"""
Data pipeline for Tryplicity pre-training.

Handles:
  - Streaming download of FineWeb-Edu and StarCoderData
  - Tokenization and packing into binary shards
  - Efficient mmap-based DataLoader for training
"""

import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import multiprocessing as mp


class TextDataset(Dataset):
    """Memory-mapped dataset of pre-tokenized binary shards.

    Each shard is a flat file of uint16 token IDs.
    Sequences are packed end-to-end for maximum efficiency (no padding).
    """

    def __init__(self, data_dir: str, seq_len: int = 2048, split: str = "train"):
        self.seq_len = seq_len
        shard_dir = os.path.join(data_dir, split)
        self.shards = sorted(Path(shard_dir).glob("*.bin"))
        assert len(self.shards) > 0, f"No shards found in {shard_dir}"

        # Memory-map all shards and compute total tokens
        self.data = []
        self.total_tokens = 0
        for shard_path in self.shards:
            size = os.path.getsize(shard_path)
            n_tokens = size // 2  # uint16 = 2 bytes
            mmap = np.memmap(shard_path, dtype=np.uint16, mode="r")
            self.data.append(mmap)
            self.total_tokens += n_tokens

        self.n_sequences = self.total_tokens // (seq_len + 1)  # +1 for target shift
        print(f"[{split}] Loaded {len(self.shards)} shards, {self.total_tokens:,} tokens, {self.n_sequences:,} sequences")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        # Find which shard and offset this index maps to
        start = idx * (self.seq_len + 1)
        tokens = self._read_tokens(start, self.seq_len + 1)
        x = torch.from_numpy(tokens[:-1].astype(np.int64))
        y = torch.from_numpy(tokens[1:].astype(np.int64))
        return x, y

    def _read_tokens(self, start: int, length: int) -> np.ndarray:
        """Read `length` tokens starting from global position `start`."""
        result = np.empty(length, dtype=np.uint16)
        written = 0
        offset = start

        for shard in self.data:
            if offset >= len(shard):
                offset -= len(shard)
                continue
            available = min(len(shard) - offset, length - written)
            result[written : written + available] = shard[offset : offset + available]
            written += available
            offset = 0
            if written >= length:
                break

        return result


class DataPipeline:
    """Downloads, tokenizes, and shards training data."""

    SHARD_SIZE = 100_000_000  # 100M tokens per shard

    def __init__(self, output_dir: str, tokenizer_path: str, seq_len: int = 2048):
        self.output_dir = output_dir
        self.seq_len = seq_len
        self.tokenizer_path = tokenizer_path

    def process_fineweb_edu(self, num_tokens: int = 20_000_000_000):
        """Download and tokenize FineWeb-Edu dataset."""
        from datasets import load_dataset
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=self.tokenizer_path)

        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        print(f"Streaming FineWeb-Edu, target: {num_tokens:,} tokens")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        shard_idx = 0
        token_buf = []
        total_tokens = 0
        val_tokens_written = 0
        val_target = num_tokens // 50  # 2% for validation

        for example in ds:
            text = example.get("text", "")
            if not text or len(text) < 50:
                continue

            tokens = sp.encode(text, out_type=int)
            token_buf.extend(tokens)

            while len(token_buf) >= self.SHARD_SIZE:
                shard_tokens = np.array(token_buf[: self.SHARD_SIZE], dtype=np.uint16)
                token_buf = token_buf[self.SHARD_SIZE :]

                # First shard goes to validation
                if val_tokens_written < val_target:
                    path = os.path.join(val_dir, f"shard_{shard_idx:05d}.bin")
                    val_tokens_written += len(shard_tokens)
                else:
                    path = os.path.join(train_dir, f"shard_{shard_idx:05d}.bin")

                shard_tokens.tofile(path)
                shard_idx += 1
                total_tokens += len(shard_tokens)
                print(f"  Shard {shard_idx}: {total_tokens:,} tokens processed")

            if total_tokens >= num_tokens:
                break

        # Write remaining tokens
        if token_buf:
            shard_tokens = np.array(token_buf, dtype=np.uint16)
            path = os.path.join(train_dir, f"shard_{shard_idx:05d}.bin")
            shard_tokens.tofile(path)
            total_tokens += len(shard_tokens)

        print(f"Done: {total_tokens:,} total tokens across {shard_idx + 1} shards")

    def process_code_data(self, num_tokens: int = 2_000_000_000):
        """Download and tokenize StarCoder data (code)."""
        from datasets import load_dataset
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=self.tokenizer_path)

        train_dir = os.path.join(self.output_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        print(f"Streaming StarCoder data, target: {num_tokens:,} tokens")
        ds = load_dataset(
            "bigcode/starcoderdata",
            data_dir="python",
            split="train",
            streaming=True,
        )

        # Find next shard index
        existing = list(Path(train_dir).glob("*.bin"))
        shard_idx = len(existing)

        token_buf = []
        total_tokens = 0

        for example in ds:
            text = example.get("content", "")
            if not text or len(text) < 20:
                continue

            tokens = sp.encode(text, out_type=int)
            token_buf.extend(tokens)

            while len(token_buf) >= self.SHARD_SIZE:
                shard_tokens = np.array(token_buf[: self.SHARD_SIZE], dtype=np.uint16)
                token_buf = token_buf[self.SHARD_SIZE :]
                path = os.path.join(train_dir, f"shard_{shard_idx:05d}.bin")
                shard_tokens.tofile(path)
                shard_idx += 1
                total_tokens += len(shard_tokens)
                print(f"  Code shard {shard_idx}: {total_tokens:,} tokens processed")

            if total_tokens >= num_tokens:
                break

        if token_buf:
            shard_tokens = np.array(token_buf, dtype=np.uint16)
            path = os.path.join(train_dir, f"shard_{shard_idx:05d}.bin")
            shard_tokens.tofile(path)
            total_tokens += len(shard_tokens)

        print(f"Code data done: {total_tokens:,} tokens")
