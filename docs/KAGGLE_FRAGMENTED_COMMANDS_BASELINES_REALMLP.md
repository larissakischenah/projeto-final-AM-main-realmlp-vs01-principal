# Comandos Kaggle — execução fragmentada Baselines + RealMLP

## Configuração comum

Antes de executar qualquer bloco no Kaggle, criar o diretório de saída:

mkdir -p /kaggle/working/results

## Modelos desta fase

Modelos executados nesta etapa:

- lightgbm
- xgboost
- catboost
- group_model

Nesta documentação, group_model representa o RealMLP/modelo do grupo.

Fora do escopo desta etapa:

- AutoGluon Default
- AutoGluon Extreme
- HPO/Optuna

## Bloco 01

Task IDs:

32 26 6 219 218

Comando:

python -m src.pipeline.run_all \
  --task-ids 32 26 6 219 218 \
  --models lightgbm xgboost catboost group_model \
  --output-dir /kaggle/working/results

## Bloco 02

Task IDs:

223 220 221 229 230

Comando:

python -m src.pipeline.run_all \
  --task-ids 223 220 221 229 230 \
  --models lightgbm xgboost catboost group_model \
  --output-dir /kaggle/working/results

## Bloco 03

Task IDs:

31 23 14 16 18

Comando:

python -m src.pipeline.run_all \
  --task-ids 31 23 14 16 18 \
  --models lightgbm xgboost catboost group_model \
  --output-dir /kaggle/working/results

## Bloco 04

Task IDs:

22 45 3 43 28

Comando:

python -m src.pipeline.run_all \
  --task-ids 22 45 3 43 28 \
  --models lightgbm xgboost catboost group_model \
  --output-dir /kaggle/working/results

## Bloco 05

Task IDs:

51 4 54 39 52

Comando:

python -m src.pipeline.run_all \
  --task-ids 51 4 54 39 52 \
  --models lightgbm xgboost catboost group_model \
  --output-dir /kaggle/working/results

## Bloco 06

Task IDs:

11 29 15 37 49

Comando:

python -m src.pipeline.run_all \
  --task-ids 11 29 15 37 49 \
  --models lightgbm xgboost catboost group_model \
  --output-dir /kaggle/working/results

## Verificação rápida após cada bloco

Comando para listar os arquivos gerados:

ls -lh /kaggle/working/results

Comando para inspecionar os resultados:

python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("/kaggle/working/results")

for name in ["raw_train.csv", "raw_test.csv"]:
    path = base / name
    print("=" * 80)
    print(name)
    print("exists:", path.exists())

    if path.exists():
        df = pd.read_csv(path)
        print("shape:", df.shape)
        print("columns:", list(df.columns))

        if "task_id" in df.columns:
            print("task_ids:", sorted(df["task_id"].dropna().unique().tolist()))

        if "model" in df.columns:
            print("models:", sorted(df["model"].dropna().unique().tolist()))
PY

## Backup após cada bloco

Após cada bloco concluído, compactar os resultados:

cd /kaggle/working
zip -r results_after_block_XX.zip results

Substituir XX por:

01, 02, 03, 04, 05 ou 06.