"""Runner do AutoGluon (presets default e extreme) nos 30 datasets do TabArena.

Uso:
    # preset default (best_quality), todos os datasets
    PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_autogluon --seed 42 2>&1 | tee results/autogluon.log

    # ambos os presets com time-limit explícito
    PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_autogluon \\
        --presets default extreme --time-limit 3600 2>&1 | tee results/autogluon.log

    # subset de datasets
    PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_autogluon \\
        --task-ids 363621 363685 2>&1 | tee results/autogluon_subset.log

Resume automático: se --output já existir, pares (task_id, model) já presentes são pulados.
Modelos gravados em results/ag_models/<task_id>_<preset>/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task
from src.models.automl import build_autogluon
from src.pipeline.evaluate import EvaluationResult, g_mean_score
from src.pipeline.split import stratified_split

_LABEL = "__target__"


def _prepare_for_pytabkit(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara DataFrames para modelos pytabkit:
    1. Converte CategoricalDtype → object (sklearn validation tenta cast para float64).
    2. Imputa NaN: mediana para numérico, moda para categórico (object).
    Fit nos dados de treino; mesmos valores aplicados no teste."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.select_dtypes(include="category").columns:
        X_train[col] = X_train[col].astype(object)
        X_test[col] = X_test[col].astype(object)

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    if num_cols and X_train[num_cols].isna().any().any():
        fill = X_train[num_cols].median()
        X_train[num_cols] = X_train[num_cols].fillna(fill)
        X_test[num_cols] = X_test[num_cols].fillna(fill)
    if cat_cols and X_train[cat_cols].isna().any().any():
        fill = X_train[cat_cols].mode().iloc[0]
        X_train[cat_cols] = X_train[cat_cols].fillna(fill)
        X_test[cat_cols] = X_test[cat_cols].fillna(fill)
    return X_train, X_test


def _label_encode_high_cardinality(
    X_train: pd.DataFrame, X_test: pd.DataFrame, max_card: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encoda colunas categóricas com cardinalidade > max_card.

    Aplicado exclusivamente no runner do AutoGluon (não afeta os modelos
    pytabkit/baselines). Previne que o AutoGluon aplique OHE internamente
    em colunas de alta cardinalidade, o que causaria explosão de features
    e erro de OOM na GPU.

    Datasets afetados nos 30 selecionados (threshold=50):
      - Amazon_employee_access: RESOURCE (7518), MGR_ID (4243),
        ROLE_FAMILY_DESC (2358), ROLE_DEPTNAME (449), ROLE_TITLE (343)
      - Marketing_Campaign: Dt_Customer (663)
      - HR_Analytics_Job_Change: city (123)

    Fit exclusivamente no conjunto de treino; mesmos mapeamentos aplicados
    ao teste para evitar vazamento de dados."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        if X_train[col].nunique() > max_card:
            cats = {v: i for i, v in enumerate(sorted(X_train[col].dropna().unique()))}
            X_train[col] = X_train[col].map(cats).fillna(-1).astype(int)
            X_test[col] = X_test[col].map(cats).fillna(-1).astype(int)
    return X_train, X_test


def _evaluate_autogluon(
    predictor,
    preset_str: str,
    time_limit: int | None,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> EvaluationResult:
    train_df = X_train.copy()
    train_df[_LABEL] = y_train

    t0 = time.perf_counter()
    predictor.fit(train_df, presets=preset_str, time_limit=time_limit)
    fit_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predictor.predict(X_test).to_numpy()
    y_proba_df = predictor.predict_proba(X_test)
    predict_time_s = time.perf_counter() - t0

    classes = np.unique(y_train)
    # Reordena colunas para coincidir com a ordem de np.unique (esperada por roc_auc_score)
    try:
        y_proba = y_proba_df[list(classes)].to_numpy()
    except KeyError:
        y_proba = y_proba_df.to_numpy()

    if classes.size == 2:
        auc = float(roc_auc_score(y_test, y_proba[:, 1]))
    else:
        auc = float(roc_auc_score(y_test, y_proba, multi_class="ovo", labels=classes))

    return EvaluationResult(
        auc_ovo=auc,
        accuracy=float(accuracy_score(y_test, y_pred)),
        g_mean=g_mean_score(y_test, y_pred),
        cross_entropy=float(log_loss(y_test, y_proba, labels=classes)),
        fit_time_s=fit_time_s,
        predict_time_s=predict_time_s,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Executa AutoGluon nos datasets do TabArena.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--presets",
        nargs="+",
        default=["default"],
        choices=["default", "extreme"],
        help="presets AutoGluon a executar (default=best_quality, extreme=extreme_quality)",
    )
    p.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="time_limit_seconds por dataset×preset (sobrescreve o default do preset)",
    )
    p.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="lista opcional de task IDs; se omitido usa RECOMMENDED_TASK_IDS",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/autogluon.csv"),
        help="CSV de saída (checkpoint após cada resultado)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids or RECOMMENDED_TASK_IDS
    n = len(task_ids)
    total_runs = n * len(args.presets)

    # Resume: carrega resultados já gravados para pular pares concluídos
    done: set[tuple[int, str]] = set()
    rows: list[dict] = []
    if args.output.exists():
        existing = pd.read_csv(args.output)
        rows = existing.to_dict("records")
        done = {(int(r["task_id"]), str(r["model"])) for r in rows}
        if done:
            print(f"Resume: {len(done)} resultado(s) já presentes em {args.output}", flush=True)

    run_no = 0
    for i, task_id in enumerate(task_ids, 1):
        for preset in args.presets:
            run_no += 1
            model_key = f"autogluon_{preset}"
            pct = run_no * 100 // total_runs

            if (task_id, model_key) in done:
                print(
                    f"[{run_no}/{total_runs}  {pct:3d}%] SKIP {task_id} / {preset} (já concluído)",
                    flush=True,
                )
                continue

            ds = load_task(task_id)
            print(
                f"\n[{run_no}/{total_runs}  {pct:3d}%] {ds.name}"
                f"  preset={preset}  n={ds.n_samples}  cls={ds.n_classes}  reg={ds.regime}",
                flush=True,
            )

            X_train, X_test, y_train, y_test = stratified_split(ds.X, ds.y, seed=args.seed)
            X_train, X_test = _prepare_for_pytabkit(X_train, X_test)
            X_train, X_test = _label_encode_high_cardinality(X_train, X_test)

            model_path = str(Path("results/ag_models") / f"{task_id}_{preset}")
            predictor, preset_str, time_limit = build_autogluon(
                label=_LABEL,
                seed=args.seed,
                preset=preset,
                time_limit_seconds=args.time_limit,
                path=model_path,
                n_classes=ds.n_classes,
            )

            metrics = _evaluate_autogluon(
                predictor, preset_str, time_limit,
                X_train, y_train, X_test, y_test,
            )

            row = {"task_id": task_id, "dataset": ds.name, "model": model_key}
            row.update(metrics.to_dict())
            rows.append(row)

            print(
                f"  -> AUC={metrics.auc_ovo:.4f}"
                f"  ACC={metrics.accuracy:.4f}"
                f"  G-Mean={metrics.g_mean:.4f}"
                f"  CE={metrics.cross_entropy:.4f}"
                f"  time={metrics.fit_time_s + metrics.predict_time_s:.1f}s",
                flush=True,
            )

            # checkpoint imediato para não perder progresso
            pd.DataFrame(rows).to_csv(args.output, index=False)

    print(f"\nResultados gravados em {args.output}  ({len(rows)} linhas)", flush=True)


if __name__ == "__main__":
    main()
