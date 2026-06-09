#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LUDB_DIR="${LUDB_DIR:-../public_dataset/physionet.org/files/ludb/1.0.1/data}"

.venv/bin/python train.py refine \
  --data_dir "$LUDB_DIR" \
  --seg-checkpoint runs/step2-base-segmenter/best_seg_model.pt \
  --output_dir runs/step3-refined-segmenter \
  --seed 42 \
  --epochs 15 \
  --batch-size 128 \
  --lr 1e-4 \
  --lambda-p-pre 1.0 \
  --lambda-p-absent 1.0 \
  --source-fs 500 \
  --target-fs 500 \
  --n_ludb_train 100 \
  --split-strategy stratified \
  --window-pre-ms 300 \
  --window-post-ms 80 \
  --p-min-overlap-ms 20 \
  --p-label-post-ms 40
