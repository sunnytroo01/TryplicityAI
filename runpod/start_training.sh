#!/bin/bash
# ============================================================
# Tryplicity — Training Launch (5x AMD MI300X)
# ============================================================
# Hardcoded for: 5x AMD Instinct MI300X on RunPod
#
# Just run: bash runpod/start_training.sh
# ============================================================

set -e

echo "============================================"
echo "  TRYPLICITY — Training Launch"
echo "  5x AMD MI300X (ROCm)"
echo "============================================"

# Install deps (fast, mostly cached)
pip install -q torch sentencepiece datasets accelerate scipy 2>/dev/null || true

cd /workspace/TryplicityAI

# Pull latest code
git pull origin main 2>/dev/null || true

# Show GPU info
echo ""
if command -v rocm-smi &>/dev/null; then
    rocm-smi --showid --showproductname 2>/dev/null || true
fi

GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 5)
echo ""
echo "  Detected: $GPU_COUNT AMD MI300X GPU(s)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  ROCm: $(python3 -c 'import torch; print(torch.version.hip)' 2>/dev/null || echo 'unknown')"

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

echo ""
echo "  Data found. Starting training..."
echo ""

# ============================================================
# AMD MI300X / ROCm environment variables
# ============================================================

# Performance
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true

# RCCL (AMD's NCCL) — P2P GPU communication
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MIN_NCHANNELS=112
export NCCL_IGNORE_CPU_AFFINITY=1

# Tunable GEMM operations (auto-tune matrix ops for MI300X)
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING=1

# Debug — show RCCL init info and enable full tracebacks on crash
export NCCL_DEBUG=WARN
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_error_rank_${RANK:-0}.json

# ============================================================
# Launch training
# ============================================================

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "  Multi-GPU mode: $GPU_COUNT GPUs via DDP (RCCL)"
    echo ""
    torchrun --nproc_per_node=$GPU_COUNT \
        scripts/train.py \
        --config configs/tryplicity_350m.json \
        2>&1 | tee /workspace/training.log
else
    echo "  Single GPU mode"
    echo ""
    python scripts/train.py \
        --config configs/tryplicity_350m.json \
        2>&1 | tee /workspace/training.log
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoint: /workspace/checkpoints/final.pt"
echo "  Log: /workspace/training.log"
echo "============================================"
