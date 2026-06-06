# Execução fragmentada no Kaggle — Baselines + RealMLP/group_model

## Objetivo

Preparar a execução fragmentada dos 30 datasets TabArena-v0.1 no Kaggle para os seguintes modelos:

- LightGBM;
- XGBoost;
- CatBoost;
- `group_model`, tratado na documentação como RealMLP/modelo do grupo.

Esta etapa não executa AutoGluon Default, AutoGluon Extreme nem HPO.

## Fonte de verdade

A fonte de verdade dos datasets é:

```python
data.load_tabarena.RECOMMENDED_TASK_IDS
