#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/kaggle/working/results}"

mkdir -p "${RESULTS_DIR}"

echo "Results dir: ${RESULTS_DIR}"
echo "Starting fragmented execution: baselines + RealMLP/group_model"
echo "AutoGluon, AutoGluon Extreme and HPO are intentionally excluded."

echo "Running block 01: 32 26 6 219 218"
python -m src.pipeline.run_all \
  --task-ids 32 26 6 219 218 \
  --models lightgbm xgboost catboost group_model \
  --output-dir "${RESULTS_DIR}"

echo "Running block 02: 223 220 221 229 230"
python -m src.pipeline.run_all \
  --task-ids 223 220 221 229 230 \
  --models lightgbm xgboost catboost group_model \
  --output-dir "${RESULTS_DIR}"

echo "Running block 03: 31 23 14 16 18"
python -m src.pipeline.run_all \
  --task-ids 31 23 14 16 18 \
  --models lightgbm xgboost catboost group_model \
  --output-dir "${RESULTS_DIR}"

echo "Running block 04: 22 45 3 43 28"
python -m src.pipeline.run_all \
  --task-ids 22 45 3 43 28 \
  --models lightgbm xgboost catboost group_model \
  --output-dir "${RESULTS_DIR}"

echo "Running block 05: 51 4 54 39 52"
python -m src.pipeline.run_all \
  --task-ids 51 4 54 39 52 \
  --models lightgbm xgboost catboost group_model \
  --output-dir "${RESULTS_DIR}"

echo "Running block 06: 11 29 15 37 49"
python -m src.pipeline.run_all \
  --task-ids 11 29 15 37 49 \
  --models lightgbm xgboost catboost group_model \
  --output-dir "${RESULTS_DIR}"

echo "Fragmented execution completed."
echo "Expected main outputs:"
echo "- ${RESULTS_DIR}/raw_train.csv"
echo "- ${RESULTS_DIR}/raw_test.csv"