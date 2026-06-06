# Histórico de adaptações do projeto

Registro das principais decisões técnicas e modificações feitas ao template base,
com a motivação de cada uma. Ordenado cronologicamente por fase de execução.

---

## Fase 0 — Ambiente

### Acesso à GPU no container Docker

**Arquivo:** `docker-compose.yml` / runtime do host  
**Problema:** ao rodar experimentos com RealMLP (PyTorch Lightning), o container
não reconhecia a GPU — `nvidia-smi` falhava com "Failed to initialize NVML".  
**Causa:** o runtime NVIDIA não é reinicializado automaticamente entre sessões.  
**Solução:** reiniciar o container (`docker compose down && docker compose up -d`)
inicializa o NVML corretamente. Após o reinício: RTX 4060 Ti 16 GB visível,
`torch.cuda.is_available() = True`, modelos treinando na GPU.

---

## Fase 1 — Seleção de datasets

### Ajuste do limiar do regime *small*

**Arquivo:** `data/load_tabarena.py` — constante `REGIME_THRESHOLDS["small"]`  
**Problema:** o limiar original era `n < 1.000`, mas o TabArena-v0.1 disponibiliza
apenas **3** datasets de classificação com menos de 1.000 amostras — insuficiente
para preencher os 10 datasets exigidos para o regime *small*.  
**Solução:** elevar **somente** o limiar superior do *small* de 1.000 para **2.000**.
Os demais limiares (*medium*: 10.000, *large*: ≥ 10.000) foram mantidos para
não distorcer as comparações entre regimes.

### IDs oficiais TabArena-v0.1

**Arquivo:** `data/load_tabarena.py` — constante `RECOMMENDED_TASK_IDS`  
**Problema:** o template continha IDs de exemplo do OpenML clássico (e.g., 32, 26, 6)
que não pertencem ao benchmark TabArena-v0.1.  
**Solução:** substituir pelos 51 IDs oficiais do OpenML benchmark suite `study_id=457`
(intervalo 363612–363712), depois selecionar os 30 conforme os critérios abaixo.

### Critérios de seleção dos 30 datasets

**Arquivo:** `data/load_tabarena.py` — `RECOMMENDED_TASK_IDS`  
**Critérios aplicados:**
- 10 datasets por regime (small / medium / large) — totalizando 30.
- Apenas tarefas de classificação (excluídas as 13 de regressão do TabArena-v0.1).
- Pelo menos um dataset binário e um multiclasse por regime.
- Dentro de cada regime, preferência pelos menores `n` (carregamento e treino mais rápido).

---

## Fase 2 — Pipeline de experimentos

### Progresso em tempo real e checkpoint

**Arquivo:** `src/pipeline/run_all.py`  
**Motivação:** runs com 30 datasets e múltiplos modelos demoram horas; sem indicação
de progresso não é possível saber se a execução travou.  
**Modificações:**
- Cabeçalho `[i/N  NN%] nome  n= cls= reg=` impresso antes de cada dataset.
- `flush=True` em todos os `print` para garantir saída em tempo real via `tee`.
- Checkpoint: salva `raw.csv` após concluir **todos os modelos de um dataset**;
  se a execução for interrompida, o progresso até o último dataset completo é preservado.
- Resume automático: ao reiniciar, pares `(task_id, model)` já presentes no CSV
  são pulados sem re-treinar.
- Execução recomendada: `PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_all ... 2>&1 | tee results/run_all.log`

### Correção: NaN em colunas contínuas

**Arquivo:** `src/pipeline/run_all.py` — função `_prepare_for_pytabkit`  
**Problema:** o módulo `pytabkit` (`sklearn_base.py`) lança
`ValueError: NaN values in continuous columns are currently not allowed!` para
**todos** os modelos com sufixo `_TD` (LGBM, XGB, CatBoost, RealMLP) ao receber
DataFrames com valores ausentes. O erro afetava 6 dos 30 datasets selecionados
(aqueles com `has_missing=True`):
`Fitness_Club`, `MIC`, `Marketing_Campaign`, `polish_companies_bankruptcy`,
`jm1`, `HR_Analytics_Job_Change`.  
**Solução:** imputação antes de qualquer modelo:
- Colunas numéricas → mediana do conjunto de treino.
- Colunas não-numéricas (object / category) → moda do conjunto de treino.
- Os mesmos valores são aplicados ao conjunto de teste (sem vazamento de dados).

### Correção: CategoricalDtype com strings causa falha de cast

**Arquivo:** `src/pipeline/run_all.py` — função `_prepare_for_pytabkit`  
**Problema:** o dataset `polish_companies_bankruptcy` possui uma coluna com dtype
`category` cujas categorias são strings (ex.: `'Aug'`). A validação interna do
sklearn (`check_array`) tenta converter o DataFrame inteiro para `float64` e falha
com `ValueError: Cannot cast object dtype to float64`.  
**Causa raiz:** quando um DataFrame misto (numérico + categórico) é validado pelo
sklearn com `dtype=None`, ele infere `float64` como tipo comum e tenta o cast.
Colunas com `CategoricalDtype` de strings não sobrevivem ao cast.  
**Solução:** converter todas as colunas com `CategoricalDtype` para `object` antes
de qualquer chamada a modelos pytabkit. O pytabkit reconhece tanto `category` quanto
`object` como indicador de coluna categórica, portanto a conversão não altera o
comportamento dos modelos.  
**Escopo:** `_prepare_for_pytabkit` é chamada para **todos** os datasets
(não só os com NaN), pois a questão de dtype é independente da presença de valores ausentes.

### Implementação do modelo do grupo

**Arquivo:** `src/models/group_model.py`  
**Modificação:** substituído o placeholder `raise NotImplementedError(...)` pela
implementação concreta:
```python
from pytabkit import RealMLP_TD_S_Classifier
return RealMLP_TD_S_Classifier(random_state=seed)
```
`RealMLP_TD_S_Classifier` usa hiperparâmetros defaults obtidos por meta-aprendizado
em centenas de datasets tabulares (Holzmuller et al., NeurIPS 2024).

---

## Fase 2.6 — Otimização de hiperparâmetros (RealMLP_HPO)

### Escolha da variante HPO

**Arquivo:** `src/models/group_model.py` — `build_group_model_hpo`  
**Contexto:** a rubrica reserva 10% para "busca de hiperparâmetros do modelo do grupo".
O README menciona Optuna como opcional para modelos pytabkit com defaults `_TD`.
Em vez de configurar um loop Optuna externo, usamos a variante `RealMLP_HPO_Classifier`
do pytabkit, que executa busca interna de hiperparâmetros controlada por `time_limit_s`.  
**Parâmetro:** `time_limit_s=180` (3 minutos por dataset) como compromisso entre
cobertura da busca e custo computacional total (~90 min para os 30 datasets).

### Flags no runner

**Arquivo:** `src/pipeline/run_all.py`  
**Adicionado:** `--include-hpo` e `--hpo-time-limit` (padrão 180 s).  
O modelo HPO é inserido no mesmo dicionário de factories e beneficia do mecanismo
de resume — baselines e `group_model` já computados são pulados automaticamente.

---

## Fase 2.7 — AutoGluon (preset default)

### Runner run_autogluon.py

**Arquivo:** `src/pipeline/run_autogluon.py` (novo)  
**Contexto:** o `usage.txt` referencia `uv run python -m src.pipeline.run_autogluon`
mas o arquivo não existia no template.  
**Implementação:**
- Suporte a `--presets default extreme`, `--time-limit`, `--task-ids`, `--output`.
- Mesma função `_prepare_for_pytabkit` para consistência de pré-processamento.
- Progresso `[run/total  NN%] nome  preset= n= cls= reg=` com `flush=True`.
- Checkpoint imediato após cada resultado (dataset × preset).
- Resume: se `--output` já existir, pares `(task_id, model)` presentes são pulados.
- Modelos salvos em `results/ag_models/<task_id>_<preset>/` para inspeção posterior.

### Parâmetro `path` no build_autogluon

**Arquivo:** `src/models/automl.py`  
**Problema:** por padrão, AutoGluon salva modelos em `AutogluonModels/` na raiz,
sem distinção entre datasets. Com 30 datasets × 2 presets, os diretórios colidiriam.  
**Solução:** adicionado parâmetro opcional `path` à função `build_autogluon`,
repassado ao construtor de `TabularPredictor`.

### Correção: OOM na GPU por OHE irrestrito no RealMLP_TD_S

**Arquivo:** `src/models/group_model.py` — `build_group_model`  
**Modelos afetados:** `group_model` e `realmlp_hpo` (ambos via pytabkit). Baselines
(lightgbm, xgboost, catboost) e AutoGluon têm mecanismos próprios e não são afetados
por este parâmetro.  
**Problema:** o preset `RealMLP_TD_S` omite `max_one_hot_cat_size` no campo `tfms`,
fazendo com que o encoder use seu valor padrão `-1` (sem limite — OHE irrestrito para
todas as colunas categóricas). Em datasets com colunas de alta cardinalidade, isso
causa explosão da entrada: no dataset `kddcup09_appetency` (não selecionado nos 30,
mas presente no TabArena-v0.1), uma coluna com ~15.415 categorias expandiu a entrada
de 212 para ~71.678 colunas; a matriz de treino de 32k linhas sozinha ocupou ~9,2 GB
na GPU → OOM. O diagnóstico inicial ("embeddings numéricos PLR/PBLD") estava errado
— o RealMLP-TD-S sequer habilita embeddings numéricos por padrão; o verdadeiro eixo
era o OHE categórico.  
**Solução:** passar `max_one_hot_cat_size=9, embedding_size=8` ao construtor:
- Colunas com ≤ 9 categorias continuam recebendo OHE (comportamento padrão do _TD completo).
- Colunas com > 9 categorias passam a usar embeddings aprendidos (dense, dimensão 8),
  reduzindo drasticamente o tamanho da entrada.
- `kddcup09_appetency`: entrada 71.678 → ~388 dims; uso de GPU 14+ GB → 0,21 GB.
- `batch_size` padrão restaurado para 256 (o recurso anterior de 32 era workaround
  do sintoma, não da causa). Proteção OOM mantida com retry em `batch_size=64`.  
**Verificação:** execução completa dos 30 datasets sem erros, zero OOM, zero NaN.

### Correção: OOM na GPU por explosão de features via OHE interno do AutoGluon

**Arquivo:** `src/pipeline/run_autogluon.py` — função `_label_encode_high_cardinality`  
**Modelos afetados:** exclusivamente o AutoGluon (`run_autogluon.py`). Os modelos
pytabkit (lightgbm, xgboost, catboost, group_model, realmlp_hpo) processam colunas
categóricas com encoding próprio e não sofrem deste problema.  
**Problema:** o AutoGluon aplica OHE (one-hot encoding) internamente em colunas
categóricas para alimentar modelos que exigem entrada numérica (NN_TORCH, FASTAI,
entre outros do preset `best_quality`). Um dataset com poucas colunas mas de alta
cardinalidade pode resultar em dezenas de milhares de features após OHE: o
Amazon_employee_access tem apenas 10 colunas, mas RESOURCE (7.518 únicos) + MGR_ID
(4.243) + ROLE_FAMILY_DESC (2.358) + demais produziriam ~14.000 colunas binárias —
multiplicadas por 108 modelos com 8-fold bagging, o consumo de memória torna-se
inviável e causa OOM na GPU.  
**Diagnóstico:** inspecionados os 30 datasets selecionados; identificados 3 com
colunas de alta cardinalidade acima do limiar de 50 únicos:

| Dataset | Coluna | Únicos | Ação |
|---------|--------|--------|------|
| Amazon_employee_access | RESOURCE | 7.518 | label encoding |
| Amazon_employee_access | MGR_ID | 4.243 | label encoding |
| Amazon_employee_access | ROLE_FAMILY_DESC | 2.358 | label encoding |
| Amazon_employee_access | ROLE_DEPTNAME | 449 | label encoding |
| Amazon_employee_access | ROLE_TITLE | 343 | label encoding |
| Marketing_Campaign | Dt_Customer | 663 | label encoding |
| HR_Analytics_Job_Change | city | 123 | label encoding |

Os demais 27 datasets têm cardinalidade máxima ≤ 50 e não são afetados.  
**Solução:** função `_label_encode_high_cardinality(X_train, X_test, max_card=50)`
chamada após `_prepare_for_pytabkit` no loop principal do `run_autogluon.py`.
Colunas com `nunique() > 50` recebem label encoding (mapeamento categoria→inteiro),
fit exclusivamente no treino — mesmos índices aplicados ao teste para evitar
vazamento. O AutoGluon recebe a coluna como inteiro e a trata como numérica,
sem tentar OHE.  
**Nota:** os resultados dos baselines e RealMLP em `raw.csv` não são afetados —
esses modelos foram rodados via `run_all.py`, que não inclui esta etapa.

### Correção: eval_metric inválida para classificação binária

**Arquivo:** `src/models/automl.py` — `build_autogluon`  
**Problema:** `eval_metric="roc_auc_ovo_macro"` é rejeitada pelo AutoGluon para
datasets binários (`problem_type='binary'`); a métrica OVO só é válida para multiclasse.  
**Solução:** adicionado parâmetro `n_classes` à função; a métrica interna é definida
como `"roc_auc"` se `n_classes == 2`, ou `"roc_auc_ovo_macro"` caso contrário.
O `run_autogluon.py` passa `n_classes=ds.n_classes` obtido do `load_task`.
A métrica interna do AutoGluon afeta apenas a seleção de modelos internamente;
a avaliação final continua sendo calculada via `roc_auc_score` do sklearn.

### Escolha do preset e time_limit

**Preset:** `default` (`best_quality`) — mais rápido que `extreme` (que usa 4h/dataset
e inclui foundation models como TabPFNv2); deixado como segunda fase se houver tempo.  
**time_limit=600s por dataset:** sem limite explícito, `best_quality` roda até esgotar
todos os modelos base, podendo levar horas por dataset. Com 600s: estimativa ~5h para
os 30 datasets; AutoGluon treina ao menos LightGBM, XGBoost e CatBoost antes de
qualquer stacking.  
**Comportamento ao atingir o limite:** o AutoGluon **não falha** — encerra o treino
de forma controlada e retorna o melhor ensemble disponível até aquele momento. Resultado
sempre válido. O protocolo de benchmark time-constrained é metodologicamente padrão
no TabArena e defensável na apresentação.

---

## Infraestrutura geral

### .gitignore — exclusões adicionais

**Arquivo:** `.gitignore`  
**Adicionados em duas etapas:**

Etapa 1 — configurações e credenciais:
- Diretórios de configuração local de ferramentas de desenvolvimento.
- `.env`, `.env.*` — arquivos de variáveis de ambiente (potenciais credenciais).
- `.bash_history` — histórico de terminal.
- `uv.lock` — lock file gerado automaticamente; não versionado neste projeto.

Etapa 2 — logs e temporários:
- `*.log` — logs de execução gerados pelos runners (`run_all.log`, etc.).
  Movido para seção própria (estava misturado com auxiliares LaTeX).
- `*.tmp`, `*.temp` — arquivos temporários genéricos.
- `*.bak`, `*.orig` — backups gerados por ferramentas de merge e patch.
- `*.swp`, `*.swo`, `*~` — arquivos temporários de editores (vim, emacs).
- `nohup.out` — saída de processos rodados em background com `nohup`.
- `desktop.ini` — metadado de pasta do Windows.

Os itens `.venv/`, `results/`, `cache/` e `AutogluonModels/` já estavam cobertos
pelo template original.

## Correção científica — separação treino/teste nas métricas

- Separada a avaliação dos modelos em dois artefatos explícitos:
  - `results/raw_train.csv`: métricas calculadas no conjunto de treinamento, usadas como base da estatística principal.
  - `results/raw_test.csv`: métricas calculadas no conjunto de teste, usadas apenas como análise complementar de generalização.
- Mantido o split estratificado 70/30 implementado em `src/pipeline/split.py`.
- Ajustado `src/pipeline/evaluate.py` para permitir avaliar um estimador já treinado em diferentes partições, sem treinar o modelo duas vezes.
- Ajustado `src/pipeline/run_all.py` para treinar uma vez, avaliar em treino e teste, e salvar checkpoints separados.
- Mantida compatibilidade legada com `--output`, que quando informado salva também as métricas de teste no caminho indicado.
- Ajustado `notebooks/04_demo_stats_regime.ipynb` para usar `results/raw_train.csv` como base da análise estatística principal.
- Preservado `results/raw_test.csv` como evidência complementar de generalização.
## Conversa 2 — Preparação técnica dos runners para Kaggle

Branch: `fix/kaggle-runners-readiness`

Alterações planejadas/aplicadas:
- Preparação do runner AutoGluon para execução no Kaggle sem execução pesada local.
- Separação das saídas AutoGluon em métricas de treino e teste:
  - `autogluon_train.csv`
  - `autogluon_test.csv`
- Manutenção de compatibilidade legada via `--output`, gravando métricas de teste quando solicitado.
- Suporte explícito a execução por `--task-ids`, permitindo rodar datasets individualmente no Kaggle.
- Checkpoint/resume por par `(task_id, model)`.
- Diretório padrão de resultados em `/kaggle/working/results/` quando o ambiente Kaggle é detectado; caso contrário, `results/`.
- AutoGluon Default com `time_limit=600` por padrão, salvo argumento explícito.
- AutoGluon Extreme preparado tecnicamente com `time_limit=14400` por padrão, sem execução local.
- Checkpoint mais frequente no runner geral `run_all.py`, salvando após cada modelo.
- Validação prevista apenas com `--help`, `py_compile` e inspeções estruturais.

## Conversa 3 — Documentação do Plano Kaggle

Branch: `docs/kaggle-execution-plan`

Objetivo:
- documentar o plano operacional de execução no Kaggle;
- criar checklist de execução, coleta, backup e consolidação;
- deixar explícita a separação entre resultados de treino e teste;
- evitar execução pesada local antes do Kaggle.

Arquivos adicionados:
- `docs/KAGGLE_EXECUTION_PLAN.md`
- `docs/KAGGLE_CHECKLIST.md`

Arquivos atualizados:
- `docs/HISTORY.md`

Resultado:
- plano operacional Kaggle documentado;
- checklist de execução criado;
- comandos por modelo/dataset definidos;
- coleta dos resultados documentada;
- consolidação local documentada;
- backup dos resultados documentado;
- nenhuma execução pesada local realizada.

Próximo passo recomendado:
- abrir Conversa 4 para validação do pacote Kaggle ou execução piloto no Kaggle.
## Conversa 4 — Exportação para Kaggle e execução fragmentada

- Criado script `scripts/build_kaggle_project_package.py` para gerar pacote ZIP limpo do projeto.
- Definido pacote de saída em `dist/realmlp-tabarena-kaggle-package.zip`.
- Documentado o processo em `docs/KAGGLE_EXPORT_PACKAGE.md`.
- Garantidas exclusões de `.git/`, `.env`, `cache/`, `logs/`, `dist/`, checkpoints, bytecode Python e artefatos pesados de modelos.
- Nenhum modelo ou notebook pesado foi executado localmente nesta etapa.

## Kaggle pilot execution — task 363621

- Criada branch `docs/kaggle-pilot-evidence` para registrar evidências dos pilotos executados no Kaggle.
- Validado upload do pacote limpo como Kaggle Dataset.
- Validado notebook piloto fragmentado com `src/pipeline/run_all.py`.
- Executado piloto em um único dataset small: task_id `363621` (`blood-transfusion-service-center`).
- Gerados CSVs separados de treino e teste em `/kaggle/working/results`.
- Piloto 1 validou os baselines: LightGBM, XGBoost e CatBoost.
- Piloto 2 validou o modelo do grupo via `--include-group-model`.
- O identificador bruto `group_model` corresponde ao RealMLP/modelo oficial do grupo para fins de documentação, manifesto e relatório.
- AutoGluon Default não foi executado nesta etapa.
- AutoGluon Extreme não foi executado nesta etapa.
- Execução completa dos 30 datasets não foi iniciada nesta etapa.
- Evidências locais adicionadas em `results/kaggle_pilots/`.
- Notebooks de execução piloto adicionados em `notebooks/kaggle_pilots/`.

### Validação metodológica relacionada

- `src/pipeline/split.py` mantém split estratificado 70/30 com `DEFAULT_TEST_SIZE = 0.30`, `DEFAULT_SEED = 42` e `stratify=y`.
- A análise estatística principal permanece baseada em `results/raw_train.csv`, conforme registrado em `notebooks/04_demo_stats_regime.ipynb`.
- `results/raw_test.csv` permanece reservado para análise complementar de generalização.
