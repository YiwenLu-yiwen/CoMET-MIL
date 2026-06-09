#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PTBXL_DIR="${PTBXL_DIR:-../public_dataset/physionet.org/files/ptb-xl/1.0.3}"

.venv/bin/python infer.py \
  --data_dir "$PTBXL_DIR" \
  --seg-checkpoint runs/step4-ptbxl-classifier/best_seg_model.pt \
  --cls-checkpoint runs/step4-ptbxl-classifier/best_cls_model.pt \
  --modes record_mil_ctx_cal \
  --output_dir runs/step6-ptbxl-infer \
  --ptbxl-resolution 100 \
  --source-fs 100 \
  --target-fs 500 \
  --lead ii \
  --plot-limit 12
