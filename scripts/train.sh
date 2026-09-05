#!/usr/bin/env zsh
# Wamda-3B-Reasoning — LoRA SFT on Apple Silicon (MLX).
# Usage:
#   ./scripts/train.sh            # full run (600 iters)
#   ITERS=50 ./scripts/train.sh   # smoke test
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ITERS="${ITERS:-600}"
BATCH="${BATCH:-4}"
LR="${LR:-1e-4}"

./.venv/bin/python -m mlx_lm.lora \
  --model "$BASE_MODEL" \
  --train \
  --data ./data \
  --adapter-path ./adapters \
  --iters "$ITERS" \
  --batch-size "$BATCH" \
  --val-batches 4 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --save-every 100 \
  --learning-rate "$LR" \
  --num-layers 16 \
  --mask-prompt \
  --fine-tune-type lora
