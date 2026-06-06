#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/project}"
RESULTS_DIR="${RESULTS_DIR:-/kaggle/working/results}"
SEED="${SEED:-42}"
BLOCK_SIZE="${BLOCK_SIZE:-5}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

mkdir -p "${RESULTS_DIR}"

TRAIN_OUTPUT="${RESULTS_DIR}/raw_train.csv"
TEST_OUTPUT="${RESULTS_DIR}/raw_test.csv"

cd "${PROJECT_DIR}"

echo "============================================================"
echo "Execução fragmentada — Baselines + RealMLP/group_model"
echo "Fonte dos datasets: data.load_tabarena.RECOMMENDED_TASK_IDS"
echo "Benchmark: TabArena-v0.1"
echo "Seed: ${SEED}"
echo "Resultados:"
echo "  treino: ${TRAIN_OUTPUT}"
echo "  teste : ${TEST_OUTPUT}"
echo "Fora do escopo: AutoGluon Default, AutoGluon Extreme, HPO/Optuna"
echo "============================================================"

mapfile -t BLOCK_LINES < <(
python - <<PY
from data.load_tabarena import RECOMMENDED_TASK_IDS

block_size = int("${BLOCK_SIZE}")
ids = list(RECOMMENDED_TASK_IDS)

if len(ids) != 30:
    raise SystemExit(f"ERRO: esperado 30 task_ids, encontrado {len(ids)}: {ids}")

for start in range(0, len(ids), block_size):
    block = ids[start:start + block_size]
    print(" ".join(str(x) for x in block))
PY
)

block_number=0

for block_ids in "${BLOCK_LINES[@]}"; do
  block_number=$((block_number + 1))
  block_label="$(printf "%02d" "${block_number}")"

  echo ""
  echo "============================================================"
  echo "Running block ${block_label}: ${block_ids}"
  echo "============================================================"

  PYTHONUNBUFFERED=1 python -u -m src.pipeline.run_all \
    --seed "${SEED}" \
    --task-ids ${block_ids} \
    --include-group-model \
    --train-output "${TRAIN_OUTPUT}" \
    --test-output "${TEST_OUTPUT}" \
    2>&1 | tee "${RESULTS_DIR}/run_block_${block_label}.log"

  echo ""
  echo "Validando resultados após bloco ${block_label}..."
  python - <<PY
from pathlib import Path
import pandas as pd

results_dir = Path("${RESULTS_DIR}")
expected_models = {"lightgbm", "xgboost", "catboost", "group_model"}

for name in ["raw_train.csv", "raw_test.csv"]:
    path = results_dir / name
    print("=" * 80)
    print(name)
    print("exists:", path.exists())

    if not path.exists():
        raise SystemExit(f"ERRO: {path} não encontrado.")

    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    task_ids = sorted(df["task_id"].dropna().astype(int).unique().tolist())
    models = sorted(df["model"].dropna().astype(str).unique().tolist())

    print("task_ids:", task_ids)
    print("models:", models)

    missing_models = expected_models.difference(models)
    if missing_models:
        raise SystemExit(f"ERRO: modelos ausentes em {name}: {sorted(missing_models)}")
PY

  echo ""
  echo "Criando backup após bloco ${block_label}..."
  (
    cd /kaggle/working
    zip -r "results_after_block_${block_label}.zip" results
  )

  echo "Backup criado: /kaggle/working/results_after_block_${block_label}.zip"
done

echo ""
echo "============================================================"
echo "Execução fragmentada concluída."
echo "Arquivos principais:"
echo "- ${TRAIN_OUTPUT}"
echo "- ${TEST_OUTPUT}"
echo "============================================================"
