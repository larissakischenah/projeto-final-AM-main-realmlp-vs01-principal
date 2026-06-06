# Checklist — Conversa 6 — Execução fragmentada Baselines + RealMLP

## Escopo

Esta checklist cobre somente:

- LightGBM
- XGBoost
- CatBoost
- group_model/RealMLP

Fora do escopo:

- AutoGluon Default
- AutoGluon Extreme
- HPO/Optuna

## Antes de executar no Kaggle

- [ ] pacote Kaggle atualizado a partir da develop;
- [ ] notebook Kaggle monta o projeto corretamente;
- [ ] diretório /kaggle/working/results criado;
- [ ] execução por --task-ids disponível;
- [ ] group_model tratado como RealMLP/modelo do grupo;
- [ ] raw_train.csv e raw_test.csv continuam separados;
- [ ] estatística principal direcionada ao raw_train.csv.

## Bloco 01

Task IDs:

32 26 6 219 218

- [ ] executado;
- [ ] raw_train.csv atualizado;
- [ ] raw_test.csv atualizado;
- [ ] logs preservados;
- [ ] backup parcial realizado.

## Bloco 02

Task IDs:

223 220 221 229 230

- [ ] executado;
- [ ] raw_train.csv atualizado;
- [ ] raw_test.csv atualizado;
- [ ] logs preservados;
- [ ] backup parcial realizado.

## Bloco 03

Task IDs:

31 23 14 16 18

- [ ] executado;
- [ ] raw_train.csv atualizado;
- [ ] raw_test.csv atualizado;
- [ ] logs preservados;
- [ ] backup parcial realizado.

## Bloco 04

Task IDs:

22 45 3 43 28

- [ ] executado;
- [ ] raw_train.csv atualizado;
- [ ] raw_test.csv atualizado;
- [ ] logs preservados;
- [ ] backup parcial realizado.

## Bloco 05

Task IDs:

51 4 54 39 52

- [ ] executado;
- [ ] raw_train.csv atualizado;
- [ ] raw_test.csv atualizado;
- [ ] logs preservados;
- [ ] backup parcial realizado.

## Bloco 06

Task IDs:

11 29 15 37 49

- [ ] executado;
- [ ] raw_train.csv atualizado;
- [ ] raw_test.csv atualizado;
- [ ] logs preservados;
- [ ] backup parcial realizado.

## Validação final no Kaggle

Ao final dos seis blocos:

- [ ] raw_train.csv existe;
- [ ] raw_test.csv existe;
- [ ] raw_train.csv contém os 30 datasets;
- [ ] raw_test.csv contém os 30 datasets;
- [ ] cada dataset possui LightGBM, XGBoost, CatBoost e group_model;
- [ ] group_model será reportado como RealMLP/modelo do grupo;
- [ ] nenhum resultado AutoGluon foi misturado nesta etapa;
- [ ] nenhum resultado HPO foi misturado nesta etapa;
- [ ] arquivos foram baixados para consolidação local posterior.

## Consolidação local posterior

Após baixar os resultados do Kaggle, a próxima etapa será:

1. copiar os arquivos para results/;
2. validar integridade de raw_train.csv;
3. validar integridade de raw_test.csv;
4. consolidar resultados locais;
5. preparar etapa posterior para AutoGluon Default/Extreme.