#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LUDB_DIR="${LUDB_DIR:-../public_dataset/physionet.org/files/ludb/1.0.1/data}"

.venv/bin/python evaluate.py \
  --data_dir "$LUDB_DIR" \
  --seg-checkpoint runs/step4-ptbxl-classifier/best_seg_model.pt \
  --output_dir runs/step5-ludb-eval \
  --batch-size 32 \
  --n_ludb_train 100 \
  --eval-split test \
  --boundary-tolerance-points 75 \
  --source-fs 500 \
  --target-fs 500 \
  --plot-limit 12
