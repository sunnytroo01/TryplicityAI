"""
Tryplicity pre-training script.

Optimized for maximum GPU utilization:
  - BF16 mixed precision
  - FlashAttention (via PyTorch SDPA)
  - Gradient checkpointing
  - Fused AdamW optimizer
  - WSD learning rate schedule (Warmup-Stable-Decay)
  - Torch.compile for kernel fusion
"""

import os
import sys
import math
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tryplicity.config import TryplicityConfig
from tryplicity.model import Tryplicity
from tryplicity.data import TextDataset


def get_lr(step: int, config: TryplicityConfig) -> float:
    """Warmup-Stable-Decay (WSD) learning rate schedule.

    From MiniCPM paper — more efficient than cosine decay:
      1. Linear warmup
      2. Constant LR (stable phase)
      3. Linear decay to min_lr
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    elif step < config.stable_steps:
        return config.learning_rate
    else:
        decay_steps = config.max_steps - config.stable_steps
        decay_progress = (step - config.stable_steps) / max(decay_steps, 1)
        return config.min_lr + (config.learning_rate - config.min_lr) * (1 - decay_progress)


def save_checkpoint(model, optimizer, step, loss, config, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": json.loads(json.dumps({
            k: v for k, v in config.__dict__.items()
        })),
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None):
    """Load training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    """Run evaluation and return average loss."""
    model.eval()
    total_loss = 0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Tryplicity")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = TryplicityConfig.load(args.config)
    else:
        config = TryplicityConfig()

    # Device
    assert torch.cuda.is_available(), "CUDA required for training"
    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Print setup
    print("=" * 60)
    print("TRYPLICITY PRE-TRAINING")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # Build model
    model = Tryplicity(config).to(device)
    n_params = model.num_parameters()
    print(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    print(f"Architecture: {config.n_layers}L, {config.n_heads}H, {config.n_kv_heads}KV, dim={config.dim}")
    print(f"Sequence length: {config.max_seq_len}")
    print(f"Effective batch: {config.batch_size * config.gradient_accumulation_steps * config.max_seq_len:,} tokens")

    # Enable gradient checkpointing to save VRAM
    for layer in model.layers:
        layer._gradient_checkpointing = True

    original_forward = model.forward

    def checkpointed_forward(input_ids, targets=None):
        B, T = input_ids.shape
        x = model.tok_emb(input_ids)
        for layer in model.layers:
            x = torch.utils.checkpoint.checkpoint(
                layer, x, model.rope_cos, model.rope_sin, None,
                use_reentrant=False
            )
        x = model.norm(x)
        logits = model.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    model.forward = checkpointed_forward

    # Compile for extra speed
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer: fused AdamW for better GPU utilization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
        fused=True,
    )

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at step {start_step}")

    # Data
    print(f"\nLoading data from {config.data_dir}...")
    train_dataset = TextDataset(config.data_dir, seq_len=config.max_seq_len, split="train")
    val_dataset = TextDataset(config.data_dir, seq_len=config.max_seq_len, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Training loop
    print(f"\nStarting training: {config.max_steps} steps")
    print(f"LR schedule: WSD (warmup={config.warmup_steps}, stable={config.stable_steps}, decay={config.max_steps - config.stable_steps})")
    print("=" * 60)

    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    tokens_processed = 0
    t0 = time.time()
    running_loss = 0.0

    for step in range(start_step, config.max_steps):
        # Update learning rate
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            accum_loss += loss.item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        tokens_this_step = config.batch_size * config.gradient_accumulation_steps * config.max_seq_len
        tokens_processed += tokens_this_step
        running_loss += accum_loss

        # Logging
        if (step + 1) % config.log_every == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed
            avg_loss = running_loss / config.log_every
            running_loss = 0.0
            print(
                f"step {step+1:>6d}/{config.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"grad_norm {grad_norm:.2f} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"{tokens_processed/1e9:.2f}B tokens"
            )

        # Evaluation
        if (step + 1) % config.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"  >>> val_loss: {val_loss:.4f} | perplexity: {math.exp(val_loss):.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, step + 1, val_loss, config,
                    os.path.join(config.checkpoint_dir, "best.pt")
                )

        # Regular checkpoint
        if (step + 1) % config.save_every == 0:
            save_checkpoint(
                model, optimizer, step + 1, accum_loss, config,
                os.path.join(config.checkpoint_dir, f"step_{step+1:06d}.pt")
            )

    # Final save
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Total time: {total_time/3600:.2f} hours")
    print(f"  Total tokens: {tokens_processed:,}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")

    save_checkpoint(
        model, optimizer, config.max_steps, best_val_loss, config,
        os.path.join(config.checkpoint_dir, "final.pt")
    )


if __name__ == "__main__":
    main()
