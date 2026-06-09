#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PTBXL_DIR="${PTBXL_DIR:-../public_dataset/physionet.org/files/ptb-xl/1.0.3}"

.venv/bin/python infer.py \
  --data_dir "$PTBXL_DIR" \
  --seg-checkpoint runs/step3-refined-segmenter/best_seg_model.pt \
  --modes record_mil_ctx_cal \
  --output_dir runs/step4-ptbxl-classifier \
  --epochs 20 \
  --batch-size 32 \
  --feature-batch-size 64 \
  --cls-lr 0.0005 \
  --ptbxl-resolution 100 \
  --source-fs 100 \
  --target-fs 500 \
  --lead ii \
  --seeds 43 \
  --save-best-cls

cp runs/step3-refined-segmenter/best_seg_model.pt runs/step4-ptbxl-classifier/best_seg_model.pt
