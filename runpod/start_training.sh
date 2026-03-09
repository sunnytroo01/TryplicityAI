#!/bin/bash
# ============================================================
# Tryplicity — Training Launch (4x NVIDIA B200)
# ============================================================
# Just run: bash runpod/start_training.sh
# ============================================================

set -e

echo "============================================"
echo "  TRYPLICITY — Training Launch"
echo "  4x NVIDIA B200"
echo "============================================"

# Install deps (fast, mostly cached)
pip install -q torch sentencepiece datasets accelerate scipy

cd /workspace/TryplicityAI

# Pull latest code
git pull origin main 2>/dev/null || true

# Show GPU info
echo ""
nvidia-smi
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo ""
echo "  Detected: $GPU_COUNT NVIDIA GPU(s)"
echo "  CUDA: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"

# Verify data exists
if [ ! -d "/workspace/data/processed/train" ]; then
    echo "ERROR: Training data not found at /workspace/data/processed/train"
    echo "Run setup_volume.sh first on a cheap pod."
    exit 1
fi

if [ ! -f "/workspace/tokenizer/tryplicity.model" ]; then
    echo "ERROR: Tokenizer not found at /workspace/tokenizer/tryplicity.model"
    echo "Run setup_volume.sh first on a cheap pod."
    exit 1
fi

echo "  Data found. Starting training..."
echo ""

# Performance environment
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVIDIA_TF32_OVERRIDE=1

# Enable error tracebacks on crash
export TORCH_SHOW_CPP_STACKTRACES=1

# Launch: multi-GPU with torchrun, single GPU with python
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "  Multi-GPU mode: $GPU_COUNT GPUs via DDP"
    echo ""
    torchrun --nproc_per_node=$GPU_COUNT \
        scripts/train.py \
        --config configs/tryplicity_350m.json \
        --compile \
        2>&1 | tee /workspace/training.log
else
    echo "  Single GPU mode"
    echo ""
    python scripts/train.py \
        --config configs/tryplicity_350m.json \
        --compile \
        2>&1 | tee /workspace/training.log
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: /workspace/checkpoints/final.pt"
echo "  Log: /workspace/training.log"
echo "============================================"
