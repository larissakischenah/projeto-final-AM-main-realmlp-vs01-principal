# Plano de Execução no Kaggle — RealMLP/TabArena

## 1. Objetivo

Este documento define o plano operacional para executar os experimentos do projeto RealMLP/TabArena no Kaggle.

A execução pesada deve ocorrer no Kaggle, enquanto a consolidação, validação, documentação e versionamento final devem ocorrer localmente no repositório Git.

O projeto compara os seguintes modelos:

- RealMLP;
- LightGBM;
- CatBoost;
- XGBoost;
- AutoGluon Default;
- AutoGluon Extreme.

A avaliação estatística principal deve usar os resultados de treinamento, salvos em:

```text
results/raw_train.csv
