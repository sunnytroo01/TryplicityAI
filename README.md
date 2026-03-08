# Tryplicity

A transformer language model built entirely from scratch. No fine-tuning, no borrowed weights -- every parameter trained from zero on raw data.

## Efficiency Stack

Every proven technique stacked for maximum compute efficiency:

| Technique | Gain | Source |
|-----------|------|--------|
| **AdEMAMix optimizer** | 95% more token-efficient than AdamW | Apple, 2024 |
| **Progressive G_stack** | 54% speedup (start 12L, grow to 24L) | NeurIPS 2024 |
| **Liger Kernel** | 20% throughput + 60% less memory | LinkedIn |
| **Sequence packing** | 1.7-3x (zero padding waste) | HuggingFace |
| **WSD schedule** | More flexible than cosine decay | MiniCPM |
| **BF16 mixed precision** | 2x throughput over FP32 | Standard |
| **FlashAttention (SDPA)** | O(n) memory, fused kernels | Tri Dao |
| **Gradient checkpointing** | 70% less activation memory | PyTorch |
| **torch.compile** | Automatic kernel fusion | PyTorch |
| **TF32 matmuls** | ~3x free speedup on Ampere+ | NVIDIA |
| **Fused optimizer** | Single kernel param updates | PyTorch |
| **Data dedup + quality filter** | Remove 50% redundant data | Meta SemDeDup |

**Combined theoretical efficiency: ~5-10x vs naive training.**

## Architecture: Tryplicity-350M

```
Parameters:  ~350M (175M in Phase 1, 350M in Phase 2)
Layers:      24 (progressive: 12 -> 24 at midpoint)
Heads:       16 query, 4 KV (Grouped Query Attention)
Dimension:   1024
FFN:         2816 (SwiGLU)
Norm:        RMSNorm (pre-norm)
Positions:   RoPE (Rotary)
Vocab:       32,000 (BPE, SentencePiece)
Context:     2,048
Tying:       Embedding = LM head (saves ~32M params)
```

## Training Data

**100% English Wikipedia** (pre-2022 snapshot, 20220301 dump):
- **6.5M+ articles** covering every domain of human knowledge
- **~4B tokens** of high-quality encyclopedic text
- **Dedup:** Near-duplicate removal via MinHash shingling
- **Filter:** Quality heuristics (length, alpha ratio, repetition)
- **Packing:** Documents concatenated with EOS separators, zero padding waste

## Custom Optimizer: AdEMAMix

Standard Adam uses one momentum term. AdEMAMix uses **two** -- a fast EMA for recent gradients and a slow EMA that retains gradient memory from thousands of steps back. This lets the model "remember" useful learning signals from much earlier in training.

```
Apple research showed: 1.3B model + AdEMAMix on 101B tokens
= same quality as AdamW on 197B tokens (95% more efficient)
```

Also includes Lion optimizer (Google Brain) as an alternative -- uses only the sign of gradients, halving optimizer memory.

## Quick Start on RunPod

### Phase 1: Setup (cheap GPU, ~$1)
```bash
git clone https://github.com/sunnytroo01/TryplicityAI.git
cd TryplicityAI
bash runpod/setup_volume.sh
```

### Phase 2: Train (B200)
```bash
cd /workspace/TryplicityAI
bash runpod/start_training.sh
```

### Chat
```bash
python scripts/chat.py --checkpoint /workspace/checkpoints/best.pt
```

## Project Structure

```
TryplicityAI/
  tryplicity/
    model.py         -- Transformer (RMSNorm, RoPE, GQA, SwiGLU)
    config.py        -- Model and training configuration
    data.py          -- Data pipeline with dedup + packing
    optim.py         -- AdEMAMix + Lion optimizers
  scripts/
    train_tokenizer.py -- Train 32K BPE tokenizer
    prepare_data.py    -- Download, filter, dedup, tokenize
    train.py           -- Training with full efficiency stack
    chat.py            -- Interactive generation
    export_model.py    -- Export to HuggingFace format
  runpod/
    setup_volume.sh    -- One-command volume setup
    start_training.sh  -- One-command training launch
  configs/
    tryplicity_350m.json
```

## License

MIT
