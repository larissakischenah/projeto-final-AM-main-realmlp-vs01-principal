# Execução fragmentada no Kaggle — Baselines + RealMLP/group_model

## Objetivo

Preparar a execução fragmentada dos 30 datasets TabArena no Kaggle para os seguintes modelos:

- LightGBM
- XGBoost
- CatBoost
- group_model, tratado na documentação como RealMLP/modelo do grupo

Esta etapa não executa AutoGluon Default, AutoGluon Extreme nem HPO.

## Premissas já validadas

As conversas anteriores já validaram:

- pacote Kaggle funcional;
- montagem correta do projeto no Kaggle;
- notebooks piloto funcionais;
- execução por --task-ids;
- baselines executando no task_id 363621;
- group_model executando no task_id 363621;
- saída em /kaggle/working/results;
- separação entre treino e teste;
- split 70/30 estratificado com seed 42;
- estatística principal baseada em results/raw_train.csv.

## Estratégia de fragmentação

Os 30 datasets serão executados em 6 blocos de 5 datasets.

A execução fragmentada reduz risco de perda de processamento, facilita retomada por checkpoint/resume e melhora a rastreabilidade por bloco.

## Blocos definidos

Bloco 01:
32 26 6 219 218

Bloco 02:
223 220 221 229 230

Bloco 03:
31 23 14 16 18

Bloco 04:
22 45 3 43 28

Bloco 05:
51 4 54 39 52

Bloco 06:
11 29 15 37 49

## Diretório de saída

Todos os resultados devem ser salvos em:

/kaggle/working/results

## Resultados esperados

Os arquivos principais esperados são:

/kaggle/working/results/raw_train.csv
/kaggle/working/results/raw_test.csv

Também devem ser preservados logs, checkpoints e manifestos quando gerados pelos runners.

## Critério de sucesso

Ao final da execução dos seis blocos:

- raw_train.csv deve conter os 30 datasets;
- raw_test.csv deve conter os 30 datasets;
- cada dataset deve conter LightGBM, XGBoost, CatBoost e group_model;
- group_model deve ser reportado como RealMLP/modelo do grupo;
- AutoGluon e HPO não devem aparecer nos resultados desta etapa;
- treino e teste devem permanecer separados.