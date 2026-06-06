"""Runner do AutoGluon para execução no Kaggle.

Este runner foi preparado para:
- executar datasets individualmente via --task-ids;
- separar métricas de treino e teste;
- fazer checkpoint/resume por par (task_id, model);
- salvar resultados preferencialmente em /kaggle/working/results quando estiver no Kaggle;
- usar time_limit=600 para AutoGluon Default, salvo argumento explícito;
- evitar que uma falha isolada interrompa toda a execução quando --fail-fast não for usado.

Uso local leve, sem treinar:
    uv run python -m src.pipeline.run_autogluon --help

Uso Kaggle, AutoGluon Default em um dataset:
    PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_autogluon \
        --presets default --task-ids 32 \
        2>&1 | tee /kaggle/working/results/autogluon_default_task_32.log

Uso Kaggle, AutoGluon Extreme em um dataset:
    PYTHONUNBUFFERED=1 uv run python -u -m src.pipeline.run_autogluon \
        --presets extreme --task-ids 32 \
        2>&1 | tee /kaggle/working/results/autogluon_extreme_task_32.log
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from data.load_tabarena import RECOMMENDED_TASK_IDS, load_task
from src.models.automl import build_autogluon
from src.pipeline.evaluate import EvaluationResult, g_mean_score
from src.pipeline.split import stratified_split

_LABEL = "__target__"


def _default_results_dir() -> Path:
    """Usa /kaggle/working/results quando disponível; caso contrário, results/."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "results"
    return Path("results")


def _prepare_for_autogluon(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepara DataFrames sem vazamento de dados.

    1. Converte CategoricalDtype para object.
    2. Imputa NaN numérico com mediana do treino.
    3. Imputa NaN categórico com moda do treino.
    """
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
        fill = X_train[cat_cols].mode(dropna=True)
        if not fill.empty:
            fill_values = fill.iloc[0]
            X_train[cat_cols] = X_train[cat_cols].fillna(fill_values)
            X_test[cat_cols] = X_test[cat_cols].fillna(fill_values)

    return X_train, X_test


def _label_encode_high_cardinality(
    X_train: pd.DataFrame, X_test: pd.DataFrame, max_card: int = 50
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encoda colunas categóricas de alta cardinalidade.

    Fit exclusivamente no treino; mesmos mapeamentos aplicados ao teste.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        if X_train[col].nunique(dropna=True) > max_card:
            cats = {v: i for i, v in enumerate(sorted(X_train[col].dropna().unique()))}
            X_train[col] = X_train[col].map(cats).fillna(-1).astype(int)
            X_test[col] = X_test[col].map(cats).fillna(-1).astype(int)

    return X_train, X_test


def _predict_proba_aligned(
    predictor: Any,
    X: pd.DataFrame,
    classes: np.ndarray,
) -> np.ndarray:
    """Obtém predict_proba com colunas alinhadas à ordem de classes."""
    y_proba_df = predictor.predict_proba(X)

    try:
        return y_proba_df[list(classes)].to_numpy()
    except KeyError:
        return y_proba_df.to_numpy()


def _evaluate_fitted_autogluon(
    predictor: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    classes: np.ndarray,
    fit_time_s: float,
) -> EvaluationResult:
    """Avalia AutoGluon já treinado em uma partição específica."""
    t0 = time.perf_counter()
    y_pred = predictor.predict(X).to_numpy()
    y_proba = _predict_proba_aligned(predictor, X, classes)
    predict_time_s = time.perf_counter() - t0

    if classes.size == 2:
        auc = float(roc_auc_score(y, y_proba[:, 1]))
    else:
        auc = float(roc_auc_score(y, y_proba, multi_class="ovo", labels=classes))

    return EvaluationResult(
        auc_ovo=auc,
        accuracy=float(accuracy_score(y, y_pred)),
        g_mean=g_mean_score(y, y_pred),
        cross_entropy=float(log_loss(y, y_proba, labels=classes)),
        fit_time_s=fit_time_s,
        predict_time_s=predict_time_s,
    )


def _checkpoint(
    train_rows: list[dict],
    test_rows: list[dict],
    train_output: Path,
    test_output: Path,
    legacy_output: Path | None,
) -> None:
    """Grava checkpoints de treino/teste e saída legada opcional."""
    pd.DataFrame(train_rows).to_csv(train_output, index=False)
    pd.DataFrame(test_rows).to_csv(test_output, index=False)

    if legacy_output is not None:
        pd.DataFrame(test_rows).to_csv(legacy_output, index=False)


def parse_args() -> argparse.Namespace:
    results_dir = _default_results_dir()

    p = argparse.ArgumentParser(description="Executa AutoGluon nos datasets do TabArena.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--presets",
        nargs="+",
        default=["default"],
        choices=["default", "extreme"],
        help="presets AutoGluon a executar: default=best_quality, extreme=extreme_quality",
    )
    p.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help=(
            "time_limit em segundos por dataset×preset; se omitido, "
            "default usa 600s e extreme usa 14400s"
        ),
    )
    p.add_argument(
        "--task-ids",
        type=int,
        nargs="*",
        default=None,
        help="lista opcional de task IDs; se omitido usa RECOMMENDED_TASK_IDS",
    )
    p.add_argument(
        "--train-output",
        type=Path,
        default=results_dir / "autogluon_train.csv",
        help="CSV de saída com métricas no conjunto de treinamento",
    )
    p.add_argument(
        "--test-output",
        type=Path,
        default=results_dir / "autogluon_test.csv",
        help="CSV de saída com métricas no conjunto de teste",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="compatibilidade legada: se informado, também salva métricas de teste neste caminho",
    )
    p.add_argument(
        "--models-dir",
        type=Path,
        default=results_dir / "ag_models",
        help="diretório para modelos AutoGluon",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="interrompe a execução na primeira falha; por padrão, registra e segue",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    task_ids = args.task_ids or RECOMMENDED_TASK_IDS
    n = len(task_ids)
    total_runs = n * len(args.presets)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    done: set[tuple[int, str]] = set()

    if args.train_output.exists():
        train_rows = pd.read_csv(args.train_output).to_dict("records")

    if args.test_output.exists():
        test_rows = pd.read_csv(args.test_output).to_dict("records")

    if train_rows and test_rows:
        done_train = {(int(r["task_id"]), str(r["model"])) for r in train_rows}
        done_test = {(int(r["task_id"]), str(r["model"])) for r in test_rows}
        done = done_train & done_test
        if done:
            print(
                f"Resume: {len(done)} par(es) task_id/model já presentes em "
                f"{args.train_output} e {args.test_output}",
                flush=True,
            )

    run_no = 0

    for i, task_id in enumerate(task_ids, 1):
        for preset in args.presets:
            run_no += 1
            model_key = f"autogluon_{preset}"
            pct = run_no * 100 // total_runs

            if (task_id, model_key) in done:
                print(
                    f"[{run_no}/{total_runs} {pct:3d}%] SKIP task_id={task_id} "
                    f"model={model_key} (checkpoint encontrado)",
                    flush=True,
                )
                continue

            try:
                ds = load_task(task_id)
                print(
                    f"\n[{run_no}/{total_runs} {pct:3d}%] "
                    f"task_id={task_id} dataset={ds.name} preset={preset} "
                    f"n={ds.n_samples} cls={ds.n_classes} reg={ds.regime}",
                    flush=True,
                )

                X_train, X_test, y_train, y_test = stratified_split(ds.X, ds.y, seed=args.seed)
                X_train, X_test = _prepare_for_autogluon(X_train, X_test)
                X_train, X_test = _label_encode_high_cardinality(X_train, X_test)

                model_path = args.models_dir / f"{task_id}_{preset}"
                predictor, preset_str, time_limit = build_autogluon(
                    label=_LABEL,
                    seed=args.seed,
                    preset=preset,
                    time_limit_seconds=args.time_limit,
                    path=str(model_path),
                    n_classes=ds.n_classes,
                )

                train_df = X_train.copy()
                train_df[_LABEL] = y_train

                print(
                    f"  -> fit AutoGluon preset={preset_str} "
                    f"time_limit={time_limit}s path={model_path}",
                    flush=True,
                )

                t0 = time.perf_counter()
                try:
                    predictor.fit(train_df, presets=preset_str, time_limit=time_limit)
                except Exception as fit_exc:
                    # AutoGluon pode deixar artefato parcial válido quando o limite de tempo
                    # ou algum modelo interno falha. Tentamos avaliar antes de descartar.
                    print(
                        f"  !! fit lançou exceção: {type(fit_exc).__name__}: {fit_exc}",
                        flush=True,
                    )
                    print("  !! tentando usar resultado parcial, se houver modelo válido...", flush=True)
                    try:
                        leaderboard = predictor.leaderboard(silent=True)
                        if leaderboard is None or leaderboard.empty:
                            raise RuntimeError("leaderboard vazio; sem modelo parcial válido") from fit_exc
                    except Exception as partial_exc:
                        print(
                            f"  !! sem resultado parcial válido: "
                            f"{type(partial_exc).__name__}: {partial_exc}",
                            flush=True,
                        )
                        raise

                fit_time_s = time.perf_counter() - t0
                classes = np.unique(np.concatenate([y_train, y_test]))

                train_metrics = _evaluate_fitted_autogluon(
                    predictor=predictor,
                    X=X_train,
                    y=y_train,
                    classes=classes,
                    fit_time_s=fit_time_s,
                )
                test_metrics = _evaluate_fitted_autogluon(
                    predictor=predictor,
                    X=X_test,
                    y=y_test,
                    classes=classes,
                    fit_time_s=fit_time_s,
                )

                base_row = {
                    "task_id": task_id,
                    "dataset": ds.name,
                    "model": model_key,
                    "preset": preset,
                    "time_limit_s": time_limit,
                }

                train_row = dict(base_row)
                train_row.update(train_metrics.to_dict())
                train_rows.append(train_row)

                test_row = dict(base_row)
                test_row.update(test_metrics.to_dict())
                test_rows.append(test_row)

                print(
                    f"  -> TRAIN AUC={train_metrics.auc_ovo:.4f} "
                    f"ACC={train_metrics.accuracy:.4f} "
                    f"G-Mean={train_metrics.g_mean:.4f} "
                    f"CE={train_metrics.cross_entropy:.4f}",
                    flush=True,
                )
                print(
                    f"  -> TEST  AUC={test_metrics.auc_ovo:.4f} "
                    f"ACC={test_metrics.accuracy:.4f} "
                    f"G-Mean={test_metrics.g_mean:.4f} "
                    f"CE={test_metrics.cross_entropy:.4f} "
                    f"time={test_metrics.total_time_s:.1f}s",
                    flush=True,
                )

                _checkpoint(
                    train_rows=train_rows,
                    test_rows=test_rows,
                    train_output=args.train_output,
                    test_output=args.test_output,
                    legacy_output=args.output,
                )

            except Exception:
                print(
                    f"  !! FALHA task_id={task_id} model={model_key}. "
                    "Execução continuará porque --fail-fast não foi usado.",
                    flush=True,
                )
                traceback.print_exc()

                _checkpoint(
                    train_rows=train_rows,
                    test_rows=test_rows,
                    train_output=args.train_output,
                    test_output=args.test_output,
                    legacy_output=args.output,
                )

                if args.fail_fast:
                    raise

    print(
        f"\nResultados de treino gravados em {args.train_output} ({len(train_rows)} linhas)",
        flush=True,
    )
    print(
        f"Resultados de teste gravados em {args.test_output} ({len(test_rows)} linhas)",
        flush=True,
    )
    if args.output is not None:
        print(
            f"Compatibilidade legada: métricas de teste também gravadas em {args.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
