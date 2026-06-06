]633;E;echo "# Execução fragmentada no Kaggle — AutoGluon Default";6a73b01e-a66d-4c50-a9c7-d4d8c7729e6b]633;C# Execução fragmentada no Kaggle — AutoGluon Default

## Objetivo

Executar o AutoGluon Default nos 30 datasets TabArena-v0.1 selecionados para o projeto RealMLP + TabArena.

Esta etapa complementa os resultados já coletados para LightGBM, XGBoost, CatBoost e group_model.

Nos CSVs brutos, group_model deve continuar como group_model.
Em tabelas finais, gráficos, relatório e slides, group_model deve ser exibido como RealMLP.

## Escopo desta fase

Incluído:

- execução de autogluon_default;
- execução fragmentada em 6 blocos de 5 datasets;
- geração de métricas em treino e teste;
- checkpoint/resume por par task_id/model;
- salvamento em /kaggle/working/results/.

Não incluído:

- AutoGluon Extreme;
- Optuna/HPO;
- consolidação estatística final;
- relatório LaTeX;
- slides.

## Arquivos principais

Script de execução:

scripts/kaggle/run_fragmented_autogluon_default_blocks.sh

Runner utilizado:

src/pipeline/run_autogluon.py

Saídas esperadas no Kaggle:

/kaggle/working/results/autogluon_default_train.csv
/kaggle/working/results/autogluon_default_test.csv

Logs esperados no Kaggle:

/kaggle/working/results/autogluon_default_block_1.log
/kaggle/working/results/autogluon_default_block_2.log
/kaggle/working/results/autogluon_default_block_3.log
/kaggle/working/results/autogluon_default_block_4.log
/kaggle/working/results/autogluon_default_block_5.log
/kaggle/working/results/autogluon_default_block_6.log

Diretório de modelos no Kaggle:

/kaggle/working/results/ag_models_default

## Comandos por bloco no Kaggle

Executar a partir da raiz do projeto no Kaggle:

bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 1
bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 2
bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 3
bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 4
bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 5
bash scripts/kaggle/run_fragmented_autogluon_default_blocks.sh 6

## Task IDs por bloco

- Bloco 1: 32 26 6 219 218
- Bloco 2: 223 220 221 229 230
- Bloco 3: 31 23 14 16 18
- Bloco 4: 22 45 3 43 28
- Bloco 5: 51 4 54 39 52
- Bloco 6: 11 29 15 37 49

## Parâmetros usados

O script chama src.pipeline.run_autogluon com:

--presets default
--time-limit 600
--train-output /kaggle/working/results/autogluon_default_train.csv
--test-output /kaggle/working/results/autogluon_default_test.csv
--models-dir /kaggle/working/results/ag_models_default
--task-ids ...

## Regra metodológica

autogluon_default_train.csv será usado como base principal das análises estatísticas futuras.
autogluon_default_test.csv será usado para análise complementar de generalização.

## Critérios de aceite

- 30 linhas em autogluon_default_train.csv;
- 30 linhas em autogluon_default_test.csv;
- 30 task_id únicos em cada arquivo;
- model = autogluon_default;
- ausência de nulos nas métricas esperadas;
- mesmos task_id de results/raw_train.csv e results/raw_test.csv;
- logs dos 6 blocos disponíveis em /kaggle/working/results/.

## Observação sobre resultado legado

O arquivo local results/autogluon.csv existente antes desta fase é parcial/piloto e não representa a execução oficial dos 30 datasets.
