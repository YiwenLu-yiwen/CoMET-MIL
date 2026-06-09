#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/step2_train_base_segmenter.sh
bash scripts/step3_refine_segmenter.sh
bash scripts/step4_train_ptbxl_classifier.sh
bash scripts/step5_evaluate_ludb.sh
bash scripts/step6_infer_ptbxl.sh
