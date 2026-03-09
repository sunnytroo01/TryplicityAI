"""
Chat with Tryplicity — even while it's still training.

Run in a second terminal on the same pod:
  python scripts/chat.py

Auto-loads the latest checkpoint from /workspace/checkpoints/.
Type "reload" to grab a newer checkpoint as training progresses.
"""

import os
import sys
import glob

import torch
import sentencepiece as spm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tryplicity.config import TryplicityConfig
from tryplicity.model import Tryplicity


def find_latest_checkpoint(checkpoint_dir="/workspace/checkpoints"):
    """Find the most recent checkpoint file."""
    candidates = []
    for pattern in ["best.pt", "step_*.pt", "final.pt"]:
        candidates.extend(glob.glob(os.path.join(checkpoint_dir, pattern)))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_model(checkpoint_path, device="cuda"):
    """Load model from checkpoint."""
    print(f"  Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = ckpt.get("config", {})
    config = TryplicityConfig(**config_dict)

    # Detect actual layer count from checkpoint (may be half if pre-G_stack)
    layer_keys = [k for k in ckpt["model_state_dict"] if k.startswith("layers.") and k.endswith(".attn_norm.weight")]
    n_layers = len(layer_keys)
    if n_layers > 0 and n_layers != config.n_layers:
        config.n_layers = n_layers

    model = Tryplicity(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", "?")
    if isinstance(loss, float):
        loss = f"{loss:.4f}"

    print(f"  Step: {step} | Loss: {loss} | Params: {model.num_parameters():,} | Layers: {n_layers}")
    return model, config, step


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = "/workspace/tokenizer/tryplicity.model"

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoints found yet. Training saves best.pt every 500 steps.")
        sys.exit(1)

    print("\n============================================")
    print("  TRYPLICITY — Live Chat")
    print("============================================\n")

    model, config, step = load_model(ckpt_path, device)

    temperature = 0.8
    max_tokens = 200

    print(f"\n  Commands: reload | quit | temp N | tokens N\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() == "quit":
            break

        if prompt.lower() == "reload":
            new_path = find_latest_checkpoint()
            if new_path:
                del model
                torch.cuda.empty_cache()
                model, config, step = load_model(new_path, device)
            else:
                print("  No checkpoints found.")
            continue

        if prompt.lower().startswith("temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"  Temperature: {temperature}")
            except ValueError:
                print("  Usage: temp 0.8")
            continue

        if prompt.lower().startswith("tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"  Max tokens: {max_tokens}")
            except ValueError:
                print("  Usage: tokens 200")
            continue

        input_ids = sp.encode(prompt, out_type=int)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        print(f"\nTryplicity (step {step})> ", end="", flush=True)
        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
            )
        generated = output[0, len(input_ids):].tolist()
        print(sp.decode(generated))
        print()


if __name__ == "__main__":
    main()
