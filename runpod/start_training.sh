#!/bin/bash
# ============================================================
# Tryplicity — Training Launch
# ============================================================
# Supports: single GPU, multi-GPU (auto-detected), NVIDIA + AMD
#
# Just run: bash runpod/start_training.sh
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

# Detect GPU type and count
if command -v nvidia-smi &>/dev/null; then
    echo ""
    nvidia-smi
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo ""
    echo "  Detected: $GPU_COUNT NVIDIA GPU(s)"
elif command -v rocm-smi &>/dev/null; then
    echo ""
    rocm-smi
    GPU_COUNT=$(rocm-smi -d | grep "GPU" | wc -l)
    # Fallback: count via PyTorch
    if [ "$GPU_COUNT" -eq 0 ]; then
        GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
    fi
    echo ""
    echo "  Detected: $GPU_COUNT AMD GPU(s)"
else
    GPU_COUNT=1
fi

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

# Performance environment
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# NVIDIA-specific
if command -v nvidia-smi &>/dev/null; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NVIDIA_TF32_OVERRIDE=1
fi

# Determine compile flag: skip on AMD (ROCm) — torch.compile has poor ROCm support
COMPILE_FLAG="--compile"
if command -v rocm-smi &>/dev/null; then
    echo "  AMD GPU detected — skipping torch.compile (not stable on ROCm)"
    COMPILE_FLAG=""
fi

# Launch: multi-GPU with torchrun, single GPU with python
if [ "$GPU_COUNT" -gt 1 ]; then
    echo "  Multi-GPU mode: $GPU_COUNT GPUs via DDP"
    echo ""
    torchrun --nproc_per_node=$GPU_COUNT \
        scripts/train.py \
        --config configs/tryplicity_350m.json \
        $COMPILE_FLAG \
        2>&1 | tee /workspace/training.log
else
    echo "  Single GPU mode"
    echo ""
    python scripts/train.py \
        --config configs/tryplicity_350m.json \
        $COMPILE_FLAG \
        2>&1 | tee /workspace/training.log
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: /workspace/checkpoints/final.pt"
echo "  Log: /workspace/training.log"
echo "============================================"
