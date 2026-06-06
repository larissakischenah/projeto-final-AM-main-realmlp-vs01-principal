# Checklist — Execução fragmentada Baselines + RealMLP

## Escopo

Esta checklist cobre somente:

- LightGBM;
- XGBoost;
- CatBoost;
- `group_model`/RealMLP.

Fora do escopo:

- AutoGluon Default;
- AutoGluon Extreme;
- HPO/Optuna.

## Regras obrigatórias

- [ ] usar `data.load_tabarena.RECOMMENDED_TASK_IDS` como fonte de verdade;
- [ ] não usar IDs antigos do OpenML clássico;
- [ ] preservar os 30 datasets TabArena-v0.1 atuais;
- [ ] preservar 10 small, 10 medium e 10 large;
- [ ] preservar o limiar small `< 2.000`;
- [ ] preservar o limiar medium `< 10.000`;
- [ ] preservar problemas binários e multiclasse conforme seleção atual;
- [ ] usar seed 42 explicitamente;
- [ ] usar split 70/30 estratificado via runner atual;
- [ ] manter treino e teste separados;
- [ ] usar `raw_train.csv` como arquivo principal para estatística posterior;
- [ ] usar `raw_test.csv` apenas para análise complementar de generalização;
- [ ] não misturar AutoGluon nesta etapa;
- [ ] não misturar HPO/Optuna nesta etapa.

## Antes de executar no Kaggle

- [ ] pacote Kaggle atualizado a partir da `develop`;
- [ ] pacote contém `scripts/kaggle/run_fragmented_baselines_realmlp_blocks.sh`;
- [ ] notebook Kaggle monta o projeto corretamente;
- [ ] diretório `/kaggle/working/results` criado;
- [ ] `PYTHONPATH=/kaggle/working/project` configurado;
- [ ] dependências mínimas instaladas;
- [ ] `openml` importando;
- [ ] `pytabkit` importando;
- [ ] `lightgbm` importando;
- [ ] `xgboost` importando;
- [ ] `catboost` importando;
- [ ] `group_model` importando;
- [ ] `run_all.py --help` conferido;
- [ ] `RECOMMENDED_TASK_IDS` contém 30 IDs `363xxx`.

## Durante a execução

Para cada bloco:

- [ ] bloco executado;
- [ ] `raw_train.csv` atualizado;
- [ ] `raw_test.csv` atualizado;
- [ ] log `run_block_XX.log` preservado;
- [ ] backup `results_after_block_XX.zip` criado;
- [ ] modelos presentes: `lightgbm`, `xgboost`, `catboost`, `group_model`;
- [ ] ausência de `autogluon`;
- [ ] ausência de `realmlp_hpo`.

## Validação final no Kaggle

Ao final dos seis blocos:

- [ ] `raw_train.csv` existe;
- [ ] `raw_test.csv` existe;
- [ ] `raw_train.csv` contém os 30 datasets;
- [ ] `raw_test.csv` contém os 30 datasets;
- [ ] cada dataset possui LightGBM, XGBoost, CatBoost e `group_model`;
- [ ] total esperado por arquivo: 120 linhas, salvo se houver falha documentada;
- [ ] `group_model` será reportado como RealMLP/modelo do grupo;
- [ ] nenhum resultado AutoGluon foi misturado nesta etapa;
- [ ] nenhum resultado HPO foi misturado nesta etapa;
- [ ] arquivos foram baixados para consolidação local posterior.

## Consolidação local posterior

Após baixar os resultados do Kaggle, a próxima etapa será:

1. copiar os arquivos para `results/`;
2. validar integridade de `raw_train.csv`;
3. validar integridade de `raw_test.csv`;
4. consolidar resultados locais;
5. preparar etapa posterior para AutoGluon Default/Extreme.
