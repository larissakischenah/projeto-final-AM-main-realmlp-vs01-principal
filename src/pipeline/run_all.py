"""Executa o pipeline completo: para cada dataset, treina e avalia todos os modelos.

Uso:
    python -m src.pipeline.run_all --seed 42
    python -m src.pipeline.run_all --seed 42 --train-output results/raw_train.csv --test-output results/raw_test.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task
from functools import partial

from src.models.baselines import BASELINE_FACTORIES
from src.models.group_model import build_group_model, build_group_model_hpo
from src.pipeline.evaluate import evaluate_fitted_estimator
from src.pipeline.split import stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("results/raw_train.csv"),
        help="caminho do CSV de saída com métricas no conjunto de treinamento",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("results/raw_test.csv"),
        help="caminho do CSV de saída com métricas no conjunto de teste",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="compatibilidade legada: se informado, também salva as métricas de teste neste caminho",
    )
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="opcional: lista de task IDs do OpenML; se omitido, usa RECOMMENDED_TASK_IDS",
    )
    parser.add_argument(
        "--include-group-model",
        action="store_true",
        help="se passado, inclui o modelo do grupo (build_group_model)",
    )
    parser.add_argument(
        "--include-hpo",
        action="store_true",
        help="se passado, inclui RealMLP_HPO (busca interna de hiperparâmetros)",
    )
    parser.add_argument(
        "--hpo-time-limit",
        type=int,
        default=180,
        help="time_limit_s por dataset para o RealMLP_HPO (padrão: 180s)",
    )
    return parser.parse_args()


def _prepare_for_pytabkit(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara DataFrames para modelos pytabkit:
    1. Converte CategoricalDtype → object (sklearn validation tenta cast para float64).
    2. Imputa NaN: mediana para numérico, moda para categórico (object).
    Fit nos dados de treino; mesmos valores aplicados no teste."""
    X_train = X_train.copy()
    X_test = X_test.copy()

    # CategoricalDtype com strings quebra o cast interno do sklearn; object funciona
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


def main() -> None:
    args = parse_args()
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids if args.task_ids else RECOMMENDED_TASK_IDS
    n = len(task_ids)

    # Resume: pula pares (task_id, model) já gravados nos checkpoints de treino/teste
    done: set[tuple[int, str]] = set()
    train_rows: list[dict] = []
    test_rows: list[dict] = []

    if args.train_output.exists():
        existing_train = pd.read_csv(args.train_output)
        train_rows = existing_train.to_dict("records")

    if args.test_output.exists():
        existing_test = pd.read_csv(args.test_output)
        test_rows = existing_test.to_dict("records")

    if train_rows and test_rows:
        done_train = {(int(r["task_id"]), str(r["model"])) for r in train_rows}
        done_test = {(int(r["task_id"]), str(r["model"])) for r in test_rows}
        done = done_train & done_test
        if done:
            n_ds = len({task_id for task_id, _ in done})
            print(
                f"Resume: {n_ds} dataset(s) já presentes em "
                f"{args.train_output} e {args.test_output}",
                flush=True,
            )

    for i, task_id in enumerate(task_ids, 1):
        ds = load_task(task_id)
        pct = i * 100 // n
        print(
            f"\n[{i}/{n}  {pct:3d}%] {ds.name}"
            f"  n={ds.n_samples}  cls={ds.n_classes}  reg={ds.regime}",
            flush=True,
        )

        X_train, X_test, y_train, y_test = stratified_split(ds.X, ds.y, seed=args.seed)
        X_train, X_test = _prepare_for_pytabkit(X_train, X_test)

        factories: dict[str, callable] = dict(BASELINE_FACTORIES)
        if args.include_group_model:
            factories["group_model"] = build_group_model
        if args.include_hpo:
            factories["realmlp_hpo"] = partial(build_group_model_hpo, time_limit_s=args.hpo_time_limit)

        for model_name, factory in factories.items():
            if (task_id, model_name) in done:
                print(f"  -> {model_name}: SKIP (já no checkpoint)", flush=True)
                continue
            estimator = factory(args.seed)

            t0 = time.perf_counter()
            estimator.fit(X_train, y_train)
            fit_time_s = time.perf_counter() - t0

            classes = np.unique(np.concatenate([y_train, y_test]))

            train_metrics = evaluate_fitted_estimator(
                estimator=estimator,
                X=X_train,
                y=y_train,
                classes=classes,
                fit_time_s=fit_time_s,
            )
            test_metrics = evaluate_fitted_estimator(
                estimator=estimator,
                X=X_test,
                y=y_test,
                classes=classes,
                fit_time_s=fit_time_s,
            )

            base_row = {"task_id": task_id, "dataset": ds.name, "model": model_name}

            train_row = dict(base_row)
            train_row.update(train_metrics.to_dict())
            train_rows.append(train_row)

            test_row = dict(base_row)
            test_row.update(test_metrics.to_dict())
            test_rows.append(test_row)

            print(
                f"  -> {model_name}: "
                f"TRAIN AUC={train_metrics.auc_ovo:.4f}"
                f" ACC={train_metrics.accuracy:.4f}"
                f" G-Mean={train_metrics.g_mean:.4f}"
                f" | TEST AUC={test_metrics.auc_ovo:.4f}"
                f" ACC={test_metrics.accuracy:.4f}"
                f" G-Mean={test_metrics.g_mean:.4f}"
                f" time={fit_time_s + test_metrics.predict_time_s:.1f}s",
                flush=True,
            )

            # checkpoint após cada modelo para não perder progresso parcial
            pd.DataFrame(train_rows).to_csv(args.train_output, index=False)
            pd.DataFrame(test_rows).to_csv(args.test_output, index=False)
            if args.output is not None:
                pd.DataFrame(test_rows).to_csv(args.output, index=False)

        # checkpoint redundante ao fim do dataset para reforçar persistência
        pd.DataFrame(train_rows).to_csv(args.train_output, index=False)
        pd.DataFrame(test_rows).to_csv(args.test_output, index=False)
        if args.output is not None:
            pd.DataFrame(test_rows).to_csv(args.output, index=False)

    print(
        f"\nResultados de treino gravados em {args.train_output}  ({len(train_rows)} linhas)",
        flush=True,
    )
    print(
        f"Resultados de teste gravados em {args.test_output}  ({len(test_rows)} linhas)",
        flush=True,
    )
    if args.output is not None:
        print(
            f"Compatibilidade legada: métricas de teste também gravadas em {args.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
