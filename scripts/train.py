"""
Tryplicity pre-training — maximum efficiency edition.

Supports: NVIDIA (CUDA) and AMD (ROCm) GPUs, single or multi-GPU via DDP.

  Single GPU:  python scripts/train.py --config configs/tryplicity_350m.json --compile
  Multi-GPU:   torchrun --nproc_per_node=5 scripts/train.py --config configs/tryplicity_350m.json --compile

Every proven efficiency technique stacked together:

  OPTIMIZER:    AdEMAMix — 95% more token-efficient than AdamW (Apple, 2024)
  SCHEDULE:     WSD (Warmup-Stable-Decay) — from MiniCPM
  PROGRESSIVE:  G_stack — start at half depth, double midway (54% speedup)
  KERNELS:      Liger Kernel — fused ops (20% throughput + 60% memory)
  PRECISION:    BF16 mixed precision — 2x throughput
  ATTENTION:    FlashAttention via SDPA — O(n) memory
  MEMORY:       Gradient checkpointing — 70% less activation memory
  COMPILATION:  torch.compile — automatic kernel fusion
  STABILITY:    QK-Norm, Z-loss, embedding scaling
  LEARNING:     Multi-token prediction, batch size warmup
  DATA:         Sequence packing, pre-tokenized mmap shards
  DISTRIBUTED:  DDP for multi-GPU scaling
"""

import os
import sys
import math
import time
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tryplicity.config import TryplicityConfig
from tryplicity.model import Tryplicity
from tryplicity.data import PackedDataset
from tryplicity.optim import AdEMAMix


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize DDP if launched with torchrun."""
    if "RANK" not in os.environ:
        return 0, 0, 1  # rank, local_rank, world_size (single GPU)

    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


def print_main(rank, *args, **kwargs):
    """Only print on rank 0."""
    if is_main(rank):
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def get_device_type():
    """Detect CUDA (NVIDIA) or ROCm (AMD)."""
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("No GPU found. Need CUDA (NVIDIA) or ROCm (AMD).")


def get_lr(step: int, config: TryplicityConfig) -> float:
    """WSD schedule — Warmup-Stable-Decay (MiniCPM)."""
    if step < config.warmup_steps:
        return config.learning_rate * (step / config.warmup_steps)
    elif step < config.stable_steps:
        return config.learning_rate
    else:
        decay_steps = config.max_steps - config.stable_steps
        progress = (step - config.stable_steps) / max(decay_steps, 1)
        return config.min_lr + (config.learning_rate - config.min_lr) * (1 - progress)


def save_checkpoint(model, optimizer, step, loss, config, path, rank=0):
    if not is_main(rank):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Unwrap DDP if needed
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "step": step,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": {k: v for k, v in config.__dict__.items()},
    }, path)
    print(f"  Saved: {path}")


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Handle DDP-wrapped models
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("step", 0)


def g_stack(model, config):
    """G_stack: duplicate layers to double model depth (NeurIPS 2024)."""
    print("\n" + "=" * 60)
    print("  G_STACK: Doubling model depth")
    print(f"  {config.n_layers // 2} layers -> {config.n_layers} layers")
    print("=" * 60)

    old_layers = list(model.layers)
    new_layers = nn.ModuleList()
    for i, layer in enumerate(old_layers):
        new_layers.append(layer)
        cloned = copy.deepcopy(layer)
        with torch.no_grad():
            for p in cloned.parameters():
                p.mul_(0.5)
            for p in old_layers[i].parameters():
                p.mul_(0.5)
        new_layers.append(cloned)

    model.layers = new_layers
    new_params = sum(p.numel() for p in model.parameters())
    print(f"  New parameter count: {new_params:,} ({new_params/1e6:.1f}M)")
    print(f"  Layers: {len(model.layers)}")
    return model


def setup_liger_kernel():
    """Try to use Liger Kernel for fused operations."""
    try:
        from liger_kernel.transformers import (
            LigerRMSNorm,
            LigerCrossEntropyLoss,
        )
        return {
            "rms_norm": LigerRMSNorm,
            "cross_entropy": LigerCrossEntropyLoss,
            "available": True,
        }
    except ImportError:
        return {"available": False}


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    n = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Handle DDP wrapper
            m = model.module if hasattr(model, "module") else model
            _, loss = m(x, y)
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Tryplicity")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile")
    parser.add_argument("--no-progressive", action="store_true")
    parser.add_argument("--optimizer", type=str, default="ademamix", choices=["ademamix", "adamw"])
    args = parser.parse_args()

    # Distributed setup
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    config = TryplicityConfig.load(args.config) if args.config else TryplicityConfig()
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Enable TF32 for NVIDIA Ampere+ (harmless no-op on AMD)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # GPU info
    gpu_name = torch.cuda.get_device_name(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    is_amd = "mi" in gpu_name.lower() or "instinct" in gpu_name.lower() or "radeon" in gpu_name.lower()

    print_main(rank, "=" * 60)
    print_main(rank, "  TRYPLICITY PRE-TRAINING")
    print_main(rank, "  Maximum Efficiency Edition")
    print_main(rank, "=" * 60)
    print_main(rank, f"\n  GPU: {gpu_name} x{world_size}")
    print_main(rank, f"  VRAM: {vram / 1024**3:.1f} GB per GPU ({vram * world_size / 1024**3:.1f} GB total)")
    print_main(rank, f"  Backend: {'ROCm (AMD)' if is_amd else 'CUDA (NVIDIA)'}")
    if not is_amd:
        print_main(rank, f"  CUDA: {torch.version.cuda}")

    # Efficiency stack
    liger = setup_liger_kernel()
    print_main(rank, "\n  Efficiency stack:")
    print_main(rank, f"  Liger Kernel: {'ENABLED' if liger.get('available') else 'not installed'}")
    print_main(rank, f"  FlashAttention (SDPA): YES")
    print_main(rank, f"  BF16 mixed precision: YES")
    print_main(rank, f"  Gradient checkpointing: YES")
    print_main(rank, f"  torch.compile: {'YES' if args.compile else 'NO'}")
    print_main(rank, f"  DDP multi-GPU: {'YES (' + str(world_size) + ' GPUs)' if world_size > 1 else 'NO (single GPU)'}")

    # Progressive training: start at half depth
    use_progressive = not args.no_progressive
    if use_progressive:
        half_config = TryplicityConfig(**{**config.__dict__, "n_layers": config.n_layers // 2})
        model = Tryplicity(half_config).to(device)
        stack_step = config.max_steps // 2
        print_main(rank, f"\n  Progressive training: ON")
        print_main(rank, f"    Phase 1: {half_config.n_layers} layers (steps 0-{stack_step})")
        print_main(rank, f"    Phase 2: {config.n_layers} layers (steps {stack_step}-{config.max_steps})")
    else:
        model = Tryplicity(config).to(device)
        stack_step = None

    n_params = model.num_parameters()
    # Scale effective batch by world_size for multi-GPU
    eff_batch = config.batch_size * config.gradient_accumulation_steps * config.max_seq_len * world_size
    print_main(rank, f"\n  Model: {n_params:,} params ({n_params/1e6:.1f}M)")
    print_main(rank, f"  Architecture: {len(model.layers)}L, {config.n_heads}H, dim={config.dim}")
    print_main(rank, f"  Effective batch: {eff_batch:,} tokens/step")

    # Keep reference to unwrapped model
    original_model = model

    # Gradient checkpointing forward
    def make_checkpointed_forward(m):
        def forward(input_ids, targets=None):
            B, T = input_ids.shape
            x = m.tok_emb(input_ids) * m.emb_scale
            for layer in m.layers:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, m.rope_cos, m.rope_sin, None,
                    use_reentrant=False
                )
            x = m.norm(x)
            logits = m.lm_head(x)
            loss = None
            if targets is not None:
                if liger.get("available") and "cross_entropy" in liger:
                    loss_fn = liger["cross_entropy"]()
                    loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                z_loss = 1e-4 * logits.float().logsumexp(dim=-1).pow(2).mean()
                loss = loss + z_loss
                if m.aux_heads is not None and T > m.n_future_tokens:
                    for i, head in enumerate(m.aux_heads):
                        offset = i + 2
                        if offset < T:
                            aux_logits = head(x[:, :-offset, :])
                            aux_targets = targets[:, offset:]
                            min_len = min(aux_logits.shape[1], aux_targets.shape[1])
                            aux_loss = F.cross_entropy(
                                aux_logits[:, :min_len].reshape(-1, aux_logits.size(-1)),
                                aux_targets[:, :min_len].reshape(-1),
                                ignore_index=-1,
                            )
                            loss = loss + 0.1 * aux_loss
            return logits, loss
        return forward

    model.forward = make_checkpointed_forward(model)

    if args.compile:
        print_main(rank, "\n  Compiling model...")
        model = torch.compile(model)

    # Wrap in DDP for multi-GPU
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        print_main(rank, f"  DDP initialized on {world_size} GPUs")

    # Optimizer
    # Use original_model.parameters() to avoid DDP wrapper issues
    if args.optimizer == "ademamix":
        print_main(rank, f"\n  Optimizer: AdEMAMix (95% more token-efficient than AdamW)")
        optimizer = AdEMAMix(
            original_model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999, 0.9999),
            alpha=5.0,
            T_alpha_beta3=config.warmup_steps * 2,
            weight_decay=config.weight_decay,
        )
    else:
        print_main(rank, f"\n  Optimizer: AdamW")
        # fused=True only works on NVIDIA
        use_fused = not is_amd
        optimizer = torch.optim.AdamW(
            original_model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            fused=use_fused,
        )

    # Resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, original_model, optimizer)
        print_main(rank, f"  Resumed from step {start_step}")

    # Data
    print_main(rank, f"\n  Loading data from {config.data_dir}...")
    train_dataset = PackedDataset(config.data_dir, seq_len=config.max_seq_len, split="train")
    val_dataset = PackedDataset(config.data_dir, seq_len=config.max_seq_len, split="val")

    # Use DistributedSampler for multi-GPU
    train_sampler = None
    val_sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    # Training
    print_main(rank, f"\n{'='*60}")
    print_main(rank, f"  Training: {config.max_steps} steps")
    print_main(rank, f"  LR: WSD (warmup={config.warmup_steps}, stable={config.stable_steps})")
    print_main(rank, f"{'='*60}\n")

    model.train()
    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    tokens_processed = 0
    t0 = time.time()
    running_loss = 0.0
    stacked = False

    for step in range(start_step, config.max_steps):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(step)

        # G_stack: double depth at midpoint
        if use_progressive and not stacked and step >= stack_step:
            # Unwrap DDP before modifying architecture
            original_model = g_stack(original_model, config)
            original_model = original_model.to(device)
            model = original_model
            model.forward = make_checkpointed_forward(original_model)

            # Re-create optimizer for new parameters
            if args.optimizer == "ademamix":
                optimizer = AdEMAMix(
                    original_model.parameters(), lr=config.learning_rate,
                    betas=(0.9, 0.999, 0.9999), alpha=5.0,
                    weight_decay=config.weight_decay,
                )
            else:
                optimizer = torch.optim.AdamW(
                    original_model.parameters(), lr=config.learning_rate,
                    betas=(0.9, 0.95), weight_decay=config.weight_decay,
                    fused=(not is_amd),
                )

            if args.compile:
                model = torch.compile(model)

            # Re-wrap in DDP
            if world_size > 1:
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, device_ids=[local_rank])

            stacked = True
            n_params = original_model.num_parameters()
            print_main(rank, f"  New param count: {n_params:,} ({n_params/1e6:.1f}M)")

        # LR schedule
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Batch size warmup
        if step < config.warmup_steps:
            warmup_frac = max(0.25, step / config.warmup_steps)
            current_accum = max(1, int(config.gradient_accumulation_steps * warmup_frac))
        else:
            current_accum = config.gradient_accumulation_steps

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(current_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / current_accum

            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(original_model.parameters(), config.grad_clip)
        optimizer.step()

        tokens_this_step = config.batch_size * current_accum * config.max_seq_len * world_size
        tokens_processed += tokens_this_step
        running_loss += accum_loss

        # Log
        if (step + 1) % config.log_every == 0:
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed
            avg_loss = running_loss / config.log_every
            running_loss = 0.0
            layers = len(original_model.layers) if hasattr(original_model, 'layers') else '?'
            print_main(rank,
                f"step {step+1:>6d}/{config.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"gnorm {grad_norm:.2f} | "
                f"{tok_per_sec:,.0f} tok/s | "
                f"{tokens_processed/1e9:.2f}B tok | "
                f"{layers}L"
            )

        # Eval
        if (step + 1) % config.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            ppl = math.exp(min(val_loss, 20))
            print_main(rank, f"  >>> val_loss: {val_loss:.4f} | ppl: {ppl:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(original_model, optimizer, step + 1, val_loss, config,
                                os.path.join(config.checkpoint_dir, "best.pt"), rank)

        # Checkpoint
        if (step + 1) % config.save_every == 0:
            save_checkpoint(original_model, optimizer, step + 1, accum_loss, config,
                            os.path.join(config.checkpoint_dir, f"step_{step+1:06d}.pt"), rank)

    # Final
    total_time = time.time() - t0
    print_main(rank, f"\n{'='*60}")
    print_main(rank, f"  TRAINING COMPLETE")
    print_main(rank, f"  Time: {total_time/3600:.2f} hours")
    print_main(rank, f"  Tokens: {tokens_processed:,}")
    print_main(rank, f"  Throughput: {tokens_processed/total_time:,.0f} tok/s avg")
    print_main(rank, f"  Best val loss: {best_val_loss:.4f}")
    print_main(rank, f"{'='*60}")

    save_checkpoint(original_model, optimizer, config.max_steps, best_val_loss, config,
                    os.path.join(config.checkpoint_dir, "final.pt"), rank)

    cleanup_distributed()


if __name__ == "__main__":
    main()
