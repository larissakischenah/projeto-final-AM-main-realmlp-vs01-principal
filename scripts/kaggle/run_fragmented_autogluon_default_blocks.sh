#!/usr/bin/env bash
set -euo pipefail

# Execução fragmentada do AutoGluon Default no Kaggle.
#
# Uso no Kaggle:
#   bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 1
#   bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 2
#   ...
#   bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 6
#
# Cada bloco executa 5 datasets.
# Saídas principais:
#   /kaggle/working/results/autogluon_default_train.csv
#   /kaggle/working/results/autogluon_default_test.csv
#
# Observação:
#   O runner possui checkpoint/resume por par (task_id, model).
#   Reexecutar um bloco deve pular datasets já presentes nos CSVs.

BLOCK="${1:-}"

if [[ -z "$BLOCK" ]]; then
  echo "Uso: bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh <bloco 1-6>"
  exit 1
fi

RESULTS_DIR="/kaggle/working/results"
mkdir -p "$RESULTS_DIR"

COMMON_ARGS=(
  --presets default
  --time-limit 600
  --train-output "$RESULTS_DIR/autogluon_default_train.csv"
  --test-output "$RESULTS_DIR/autogluon_default_test.csv"
  --models-dir "$RESULTS_DIR/ag_models_default"
)

case "$BLOCK" in
  1)
    TASK_IDS=(32 26 6 219 218)
    ;;
  2)
    TASK_IDS=(223 220 221 229 230)
    ;;
  3)
    TASK_IDS=(31 23 14 16 18)
    ;;
  4)
    TASK_IDS=(22 45 3 43 28)
    ;;
  5)
    TASK_IDS=(51 4 54 39 52)
    ;;
  6)
    TASK_IDS=(11 29 15 37 49)
    ;;
  *)
    echo "Bloco inválido: $BLOCK. Use um valor de 1 a 6."
    exit 1
    ;;
esac

echo "============================================================"
echo "AutoGluon Default - bloco $BLOCK"
echo "Task IDs: ${TASK_IDS[*]}"
echo "Resultados em: $RESULTS_DIR"
echo "============================================================"

PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_autogluon \
  "${COMMON_ARGS[@]}" \
  --task-ids "${TASK_IDS[@]}" \
  2>&1 | tee "$RESULTS_DIR/autogluon_default_block_${BLOCK}.log"

echo "============================================================"
echo "Bloco $BLOCK concluído."
echo "Arquivos atuais:"
ls -lh "$RESULTS_DIR"/autogluon_default_*.csv "$RESULTS_DIR"/autogluon_default_block_"$BLOCK".log 2>/dev/null || true
echo "============================================================"
