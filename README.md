# Tryplicity

A transformer language model built entirely from scratch. No fine-tuning, no borrowed weights — every parameter trained from zero.

## Architecture

Tryplicity uses a modern decoder-only transformer with every efficiency trick that matters:

| Component | Choice | Why |
|-----------|--------|-----|
| Normalization | RMSNorm (pre-norm) | Faster than LayerNorm, more stable |
| Positions | Rotary (RoPE) | No learned embeddings, extrapolates |
| Attention | Grouped Query (GQA) | 4 KV heads vs 16 query — 4x less KV cache |
| Feed-forward | SwiGLU | Better than ReLU/GELU at same compute |
| Weight tying | Embedding = LM head | Saves ~32M params |
| LR Schedule | Warmup-Stable-Decay | More efficient than cosine (MiniCPM) |
| Precision | BF16 mixed | Full throughput on modern GPUs |
| Memory | Gradient checkpointing | Trades 30% compute for 70% less VRAM |
| Speed | torch.compile + FlashAttention | Fused kernels, O(n) memory attention |
| Optimizer | Fused AdamW | Kernel-fused parameter updates |

## Model: Tryplicity-350M

```
Parameters:  ~350M
Layers:      24
Heads:       16 query, 4 KV (GQA)
Dimension:   1024
FFN:         2816 (SwiGLU)
Vocab:       32,000 (BPE)
Context:     2,048
```

## Training Data

Trained from scratch on high-quality open data:
- **90%** FineWeb-Edu — curated educational web text
- **10%** StarCoderData — Python source code

Total: ~20B tokens, tokenized with a custom 32K BPE tokenizer.

## Quick Start on RunPod

### Phase 1: Setup (cheap GPU, ~$1)
```bash
# Spin up any cheap GPU with a 50GB network volume at /workspace
git clone https://github.com/sunnytroo01/TryplicityAI.git
cd TryplicityAI
bash runpod/setup_volume.sh
# Stop the pod when done
```

### Phase 2: Train (B200 GPU)
```bash
# Spin up a B200, attach the SAME network volume
cd /workspace/TryplicityAI
bash runpod/start_training.sh
```

### Phase 3: Chat
```bash
python scripts/chat.py --checkpoint /workspace/checkpoints/best.pt
```

### Export
```bash
python scripts/export_model.py \
    --checkpoint /workspace/checkpoints/best.pt \
    --output-dir /workspace/tryplicity-export
```

## Project Structure

```
TryplicityAI/
  tryplicity/
    model.py         — Transformer architecture (from scratch)
    config.py        — Model and training configuration
    data.py          — Data pipeline: download, tokenize, shard
  scripts/
    train_tokenizer.py — Train 32K BPE tokenizer
    prepare_data.py    — Download and process training data
    train.py           — Pre-training with full GPU optimization
    chat.py            — Interactive generation
    export_model.py    — Export to HuggingFace format
  runpod/
    setup_volume.sh    — One-command volume setup
    start_training.sh  — One-command training launch
  configs/
    tryplicity_350m.json — Model configuration
```

## License

MIT
