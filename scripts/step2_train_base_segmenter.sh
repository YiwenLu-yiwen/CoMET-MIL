#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LUDB_DIR="${LUDB_DIR:-../public_dataset/physionet.org/files/ludb/1.0.1/data}"

.venv/bin/python train.py base \
  --data_dir "$LUDB_DIR" \
  --output_dir runs/step2-base-segmenter \
  --seed 42 \
  --epochs 1000 \
  --seg-lr 0.001 \
  --batch-size 32 \
  --n_ludb_train 100 \
  --split-strategy stratified \
  --source-fs 500 \
  --target-fs 500
