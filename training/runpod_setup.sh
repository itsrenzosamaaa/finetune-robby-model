#!/usr/bin/env bash
# runpod_setup.sh — Run this once on your RunPod instance before training.
# Designed for: RunPod PyTorch 2.4+ template (CUDA 12.1)

set -e

echo "=== Installing Unsloth + training deps ==="

# Unsloth installs fastest with the pre-built wheel for your CUDA version.
# This command auto-detects torch + CUDA and picks the right variant.
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --quiet

pip install \
  "transformers>=4.45.0" \
  "trl>=0.12.0" \
  "peft>=0.13.0" \
  "accelerate>=1.0.0" \
  "bitsandbytes>=0.44.0" \
  "datasets>=3.0.0" \
  "huggingface_hub>=0.25.0" \
  --quiet

echo "=== Done. Checking GPU ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo ""
echo "=== Ready to train. Example command: ==="
echo "python training/finetune.py \\"
echo "  --train data/train_train.jsonl \\"
echo "  --val   data/train_val.jsonl \\"
echo "  --output ./output \\"
echo "  --epochs 3 \\"
echo "  --batch-size 4"
