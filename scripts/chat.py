"""
Interactive chat with a trained Tryplicity model.
"""

import os
import sys
import argparse

import torch
import sentencepiece as spm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tryplicity.config import TryplicityConfig
from tryplicity.model import Tryplicity


def main():
    parser = argparse.ArgumentParser(description="Chat with Tryplicity")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="/workspace/tokenizer/tryplicity.model")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = TryplicityConfig(**ckpt["config"])
    model = Tryplicity(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    print(f"\nTryplicity ({config.total_params/1e6:.0f}M params) loaded on {device}")
    print("Type your message. Press Ctrl+C to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            input_ids = sp.encode(prompt, out_type=int)
            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            with torch.no_grad():
                output = model.generate(
                    input_tensor,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )

            generated = output[0, len(input_ids):].tolist()
            text = sp.decode(generated)
            print(f"Tryplicity: {text}\n")

        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()
