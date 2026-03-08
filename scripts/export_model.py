"""
Export a trained Tryplicity checkpoint to HuggingFace format.

This lets you:
  - Load the model with transformers
  - Push to HuggingFace Hub
  - Convert to GGUF for llama.cpp inference
"""

import os
import sys
import json
import shutil
import argparse

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tryplicity.config import TryplicityConfig
from tryplicity.model import Tryplicity


def export_hf(checkpoint_path: str, output_dir: str, tokenizer_path: str):
    """Export to HuggingFace-compatible format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = ckpt["config"]
    config = TryplicityConfig(**config_dict)

    model = Tryplicity(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Save model weights
    print("Saving model weights...")
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Save config in HF format
    hf_config = {
        "architectures": ["TryplicityForCausalLM"],
        "model_type": "tryplicity",
        "vocab_size": config.vocab_size,
        "hidden_size": config.dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "hidden_act": "silu",
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer
    if os.path.exists(tokenizer_path):
        shutil.copy(tokenizer_path, os.path.join(output_dir, "tokenizer.model"))

    # Tokenizer config
    tok_config = {
        "model_type": "tryplicity",
        "tokenizer_class": "SentencePieceTokenizer",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nExported to {output_dir}")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Step: {ckpt.get('step', 'unknown')}")
    print(f"  Loss: {ckpt.get('loss', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Export Tryplicity model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", type=str, default="/workspace/tryplicity-export")
    parser.add_argument("--tokenizer", type=str, default="/workspace/tokenizer/tryplicity.model")
    args = parser.parse_args()

    export_hf(args.checkpoint, args.output_dir, args.tokenizer)


if __name__ == "__main__":
    main()
