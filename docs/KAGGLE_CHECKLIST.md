# Checklist Kaggle — RealMLP/TabArena

## 1. Checklist antes de ir para o Kaggle

- [ ] Estar na branch `develop` atualizada.
- [ ] Conferir `git status` limpo antes de criar branch de trabalho.
- [ ] Confirmar que a branch atual é `docs/kaggle-execution-plan`.
- [ ] Confirmar que a separação treino/teste está documentada.
- [ ] Confirmar que `run_all.py` gera `raw_train.csv` e `raw_test.csv`.
- [ ] Confirmar que `run_autogluon.py` gera `autogluon_train.csv` e `autogluon_test.csv`.
- [ ] Confirmar que `run_autogluon.py` aceita execução por `--task-ids`.
- [ ] Confirmar que AutoGluon Default usa `time_limit=600`.
- [ ] Confirmar a decisão metodológica/técnica para AutoGluon Extreme.
- [ ] Confirmar que `cache/`, modelos treinados e logs pesados não serão versionados.
- [ ] Não executar os 30 datasets localmente.

---

## 2. Checklist do pacote Kaggle

- [ ] Código fonte incluído.
- [ ] Diretório `src/` incluído.
- [ ] Diretório `data/` incluído quando necessário.
- [ ] Diretório `notebooks/` incluído quando necessário.
- [ ] Diretório `docs/` incluído quando útil para consulta.
- [ ] `.env` removido.
- [ ] `.git/` removido.
- [ ] `cache/` removido.
- [ ] `__pycache__/` removido.
- [ ] Arquivos `*.pyc` removidos.
- [ ] Modelos AutoGluon removidos.
- [ ] Logs pesados removidos.

---

## 3. Checklist do ambiente Kaggle

- [ ] Internet habilitada quando necessária.
- [ ] Dataset/snapshot do projeto adicionado ao notebook.
- [ ] Diretório `/kaggle/working/results/` criado.
- [ ] Dependências instaladas.
- [ ] Imports principais funcionando.
- [ ] `openml` funcionando.
- [ ] Dependência do RealMLP funcionando.
- [ ] Runner localizado corretamente no ambiente Kaggle.

---

## 4. Checklist para execução fragmentada

- [ ] Selecionar subconjunto de `task_id`.
- [ ] Executar primeiro 1 dataset piloto.
- [ ] Conferir se os CSVs foram criados.
- [ ] Conferir se treino e teste foram separados.
- [ ] Conferir se checkpoint/resume funcionou.
- [ ] Só depois ampliar para mais datasets.
- [ ] Registrar datasets concluídos.
- [ ] Registrar datasets com falha.
- [ ] Registrar datasets com timeout.

---

## 5. Comandos-base

### Runner geral

```bash
python src/pipeline/run_all.py \
  --task-ids 11 15 29 \
  --output-dir /kaggle/working/results
