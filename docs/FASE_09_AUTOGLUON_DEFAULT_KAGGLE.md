]633;E;echo "# Fase 09 — AutoGluon Default no Kaggle";6a73b01e-a66d-4c50-a9c7-d4d8c7729e6b]633;C# Fase 09 — AutoGluon Default no Kaggle

## Objetivo

Preparar a execução fragmentada do AutoGluon Default no Kaggle para os 30 datasets TabArena-v0.1 do projeto RealMLP + TabArena.

## Estado inicial

A Fase 8 validou localmente os resultados Kaggle de Baselines + RealMLP/group_model:

- results/raw_train.csv: 120 linhas;
- results/raw_test.csv: 120 linhas;
- 30 datasets;
- 4 modelos por dataset: lightgbm, xgboost, catboost e group_model.

O AutoGluon ainda não foi incorporado aos resultados completos.

O arquivo results/autogluon.csv existente foi identificado como parcial/piloto, contendo apenas 2 datasets. Portanto, ele não deve ser usado como resultado oficial da Fase 9.

## Decisão técnica

O runner src/pipeline/run_autogluon.py já atende aos requisitos principais:

- aceita --task-ids;
- aceita --presets default;
- salva treino e teste separadamente;
- usa checkpoint/resume;
- usa /kaggle/working/results como destino preferencial no Kaggle;
- salva modelo como autogluon_default;
- permite continuar após falha isolada quando --fail-fast não é usado.

Portanto, a Fase 9 não altera o runner principal.

## Alteração realizada

Foi criado o script:

scripts/kaggle/run_fragmented_autogluon_default_blocks.sh

Ele divide os 30 datasets em 6 blocos de 5 datasets e executa AutoGluon Default com time_limit=600.

## Saídas esperadas no Kaggle

/kaggle/working/results/autogluon_default_train.csv
/kaggle/working/results/autogluon_default_test.csv

Logs esperados:

/kaggle/working/results/autogluon_default_block_1.log
/kaggle/working/results/autogluon_default_block_2.log
/kaggle/working/results/autogluon_default_block_3.log
/kaggle/working/results/autogluon_default_block_4.log
/kaggle/working/results/autogluon_default_block_5.log
/kaggle/working/results/autogluon_default_block_6.log

## Validação local realizada

A venv local foi recriada no caminho correto do projeto com Python 3.11.15.

A instalação foi feita com uv sync.

A validação confirmou Python 3.11.15 e pandas 2.3.3.

O script fragmentado foi validado com:

bash -n scripts/kaggle/run_fragmented_autogluon_default_blocks.sh

Resultado: sintaxe OK.

## Regra de nomenclatura

Nos CSVs brutos, group_model deve permanecer como group_model.

Em tabelas finais, gráficos, relatório e slides, group_model deve aparecer como RealMLP.

Mapeamento obrigatório nas fases finais:

lightgbm -> LightGBM
xgboost -> XGBoost
catboost -> CatBoost
group_model -> RealMLP
autogluon_default -> AutoGluon Default

## Próximo passo

Regenerar o pacote Kaggle, enviar ou atualizar no Kaggle e executar os 6 blocos do AutoGluon Default.

Depois da execução, baixar os resultados e validar localmente:

- 30 linhas em autogluon_default_train.csv;
- 30 linhas em autogluon_default_test.csv;
- ausência de nulos nas métricas;
- mesmos task_ids dos arquivos results/raw_train.csv e results/raw_test.csv.
