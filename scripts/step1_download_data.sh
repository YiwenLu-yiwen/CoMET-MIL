#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LUDB_DIR="${LUDB_DIR:-../public_dataset/physionet.org/files/ludb/1.0.1/data}"
PTBXL_DIR="${PTBXL_DIR:-../public_dataset/physionet.org/files/ptb-xl/1.0.3}"

.venv/bin/python download_ludb.py --output-dir "$LUDB_DIR"
.venv/bin/python download_ptbxl.py --output-dir "$PTBXL_DIR" --codes AFIB AFLT SR
