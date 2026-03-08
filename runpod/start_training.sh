#!/bin/bash
# ============================================================
# Tryplicity — Phase 2: Training on B200
# ============================================================
# Run this on a B200 GPU pod with the network volume attached.
# Everything is already downloaded — training starts instantly.
# ============================================================

set -e

echo "============================================"
echo "  TRYPLICITY — Training Launch"
echo "============================================"

# Install deps (fast, mostly cached)
pip install -q torch sentencepiece datasets accelerate scipy

cd /workspace/TryplicityAI

# Pull latest code
git pull origin main 2>/dev/null || true

# Print GPU info
echo ""
nvidia-smi
echo ""

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

echo "Data found. Starting training..."
echo ""

# Set environment for maximum performance
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# Enable TF32 for Ampere+ GPUs (free 3x speedup on matmuls)
export NVIDIA_TF32_OVERRIDE=1

# Launch training with torch.compile for maximum throughput
python scripts/train.py \
    --config configs/tryplicity_350m.json \
    --compile \
    2>&1 | tee /workspace/training.log

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: /workspace/checkpoints/final.pt"
echo "  Log: /workspace/training.log"
echo "============================================"
