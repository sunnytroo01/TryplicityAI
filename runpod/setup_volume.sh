#!/bin/bash
# ============================================================
# Tryplicity — Phase 1: Setup Network Volume
# ============================================================
# Run this on a CHEAP RunPod GPU (RTX 3090 / 4090).
# It downloads everything to /workspace so it persists on
# the network volume. When done, STOP this pod and spin up
# a B200 for training — zero downloads, instant start.
# ============================================================

set -e

echo "============================================"
echo "  TRYPLICITY — Volume Setup"
echo "============================================"

# 1. Install dependencies
echo "[1/5] Installing dependencies..."
pip install -q torch sentencepiece datasets huggingface_hub accelerate peft bitsandbytes trl scipy

# 2. Clone the repo
echo "[2/5] Cloning TryplicityAI..."
cd /workspace
if [ ! -d "TryplicityAI" ]; then
    git clone https://github.com/sunnytroo01/TryplicityAI.git
fi
cd TryplicityAI

# 3. Train tokenizer
echo "[3/5] Training tokenizer (32K BPE)..."
mkdir -p /workspace/tokenizer
python scripts/train_tokenizer.py \
    --output-dir /workspace/tokenizer \
    --vocab-size 32000 \
    --sample-mb 500

# 4. Download and tokenize training data
echo "[4/5] Downloading and tokenizing training data..."
echo "       This will stream ~20B tokens from FineWeb-Edu + StarCoder."
echo "       Expected time: 2-4 hours. Data is saved incrementally."
python scripts/prepare_data.py \
    --output-dir /workspace/data/processed \
    --tokenizer /workspace/tokenizer/tryplicity.model \
    --total-tokens 20000000000

# 5. Verify
echo "[5/5] Verifying..."
echo ""
echo "Tokenizer:"
ls -lh /workspace/tokenizer/
echo ""
echo "Training data:"
du -sh /workspace/data/processed/train/
du -sh /workspace/data/processed/val/
echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "  Now STOP this pod and spin up a B200."
echo "  Attach the same network volume."
echo "  Run: bash runpod/start_training.sh"
echo "============================================"
