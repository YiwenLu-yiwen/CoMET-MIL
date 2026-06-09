#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PTBXL_DIR="${PTBXL_DIR:-../public_dataset/physionet.org/files/ptb-xl/1.0.3}"
SEG_CKPT="${SEG_CKPT:-runs/step3-refined-segmenter/best_seg_model.pt}"
SEEDS="${SEEDS:-43}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-64}"
CLS_LR="${CLS_LR:-0.0005}"
LEAD="${LEAD:-ii}"
RESTRICT_RESOLUTION="${RESTRICT_RESOLUTION:-500}"

.venv/bin/python infer.py \
  --data_dir "$PTBXL_DIR" \
  --seg-checkpoint "$SEG_CKPT" \
  --modes record_mil_ctx_cal \
  --output_dir runs/exp-ptbxl-robustness-100-native \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --cls-lr "$CLS_LR" \
  --ptbxl-resolution 100 \
  --restrict-to-available-resolution "$RESTRICT_RESOLUTION" \
  --source-fs 100 \
  --target-fs 100 \
  --lead "$LEAD" \
  --seeds "$SEEDS" \
  --save-best-cls

.venv/bin/python infer.py \
  --data_dir "$PTBXL_DIR" \
  --seg-checkpoint "$SEG_CKPT" \
  --modes record_mil_ctx_cal \
  --output_dir runs/exp-ptbxl-robustness-100-to-500 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --cls-lr "$CLS_LR" \
  --ptbxl-resolution 100 \
  --restrict-to-available-resolution "$RESTRICT_RESOLUTION" \
  --source-fs 100 \
  --target-fs 500 \
  --lead "$LEAD" \
  --seeds "$SEEDS" \
  --save-best-cls

.venv/bin/python infer.py \
  --data_dir "$PTBXL_DIR" \
  --seg-checkpoint "$SEG_CKPT" \
  --modes record_mil_ctx_cal \
  --output_dir runs/exp-ptbxl-robustness-500-native \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --feature-batch-size "$FEATURE_BATCH_SIZE" \
  --cls-lr "$CLS_LR" \
  --ptbxl-resolution 500 \
  --restrict-to-available-resolution "$RESTRICT_RESOLUTION" \
  --source-fs 500 \
  --target-fs 500 \
  --lead "$LEAD" \
  --seeds "$SEEDS" \
  --save-best-cls
