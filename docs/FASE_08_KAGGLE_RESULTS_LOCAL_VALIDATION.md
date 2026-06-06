# Fase 08 — Coleta local dos resultados Kaggle Baselines + RealMLP

## Objetivo

Coletar localmente os resultados finais executados no Kaggle para os modelos baseline e RealMLP, preservando a separação entre treino e teste.

## Origem dos artefatos

Arquivo baixado do Kaggle:

- `results_baselines_realmlp_final.zip`

Cópia local não versionada:

- `artifacts/kaggle_downloads/results_baselines_realmlp_final.zip`

O diretório `artifacts/` está ignorado pelo Git e serve apenas como backup local.

## Arquivos finais coletados

Arquivos versionáveis copiados para `results/`:

- `results/raw_train.csv`
- `results/raw_test.csv`

Logs copiados localmente, mas não versionados:

- `results/run_block_01.log`
- `results/run_block_02.log`
- `results/run_block_03.log`
- `results/run_block_04.log`
- `results/run_block_05.log`
- `results/run_block_06.log`

## Validação local

Os dois CSVs passaram na validação estrutural local.

### `results/raw_train.csv`

- 120 linhas.
- 10 colunas.
- 30 `task_id` únicos.
- 30 datasets únicos.
- 4 modelos por dataset.
- Modelos presentes:
  - `catboost`
  - `group_model`
  - `lightgbm`
  - `xgboost`
- Sem AutoGluon.
- Sem Optuna.
- Sem HPO.
- Sem valores vazios nas colunas esperadas.

### `results/raw_test.csv`

- 120 linhas.
- 10 colunas.
- 30 `task_id` únicos.
- 30 datasets únicos.
- 4 modelos por dataset.
- Modelos presentes:
  - `catboost`
  - `group_model`
  - `lightgbm`
  - `xgboost`
- Sem AutoGluon.
- Sem Optuna.
- Sem HPO.
- Sem valores vazios nas colunas esperadas.

## Colunas esperadas

```text
task_id
dataset
model
auc_ovo
accuracy
g_mean
cross_entropy
fit_time_s
predict_time_s
total_time_s
```

## Regra de nomenclatura

Nos CSVs brutos, o modelo RealMLP permanece registrado tecnicamente como:

```text
group_model
```

Nas tabelas finais, gráficos, relatório e slides, `group_model` deve ser exibido como:

```text
RealMLP
```

Mapeamento recomendado para etapas posteriores:

```python
MODEL_DISPLAY_NAMES = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "group_model": "RealMLP",
}
```

## Uso posterior

- `results/raw_train.csv` será a base principal para os testes estatísticos posteriores.
- `results/raw_test.csv` será reservado para análise complementar de generalização.
- AutoGluon Default, AutoGluon Extreme, HPO/Optuna e análise estatística final permanecem fora do escopo desta fase.
