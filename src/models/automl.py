"""AutoGluon 1.4 nos presets `default` e `extreme`.

O preset `default` usa `best_quality` com orçamento padrão de 600 segundos
por dataset, salvo argumento explícito.

O preset `extreme` usa `extreme_quality` e fica preparado para execução pesada
no Kaggle. Quando nenhum limite explícito é informado, usa 14400 segundos
(4 horas), conforme planejamento original do projeto.
"""

from __future__ import annotations

from autogluon.tabular import TabularPredictor


DEFAULT_TIME_LIMIT_SECONDS = {
    "default": 600,
    "extreme": 14_400,
}


def build_autogluon(
    label: str,
    seed: int = 42,
    preset: str = "default",
    time_limit_seconds: int | None = None,
    path: str | None = None,
    n_classes: int = 2,
) -> tuple[TabularPredictor, str, int]:
    """Constroi um TabularPredictor do AutoGluon.

    Args:
        label: nome da coluna alvo no DataFrame de treino.
        seed: semente fixa mantida na assinatura para padronizar os runners.
        preset: "default" ou "extreme".
        time_limit_seconds: substitui o orçamento padrão do preset, se informado.
        path: diretório onde AutoGluon grava os modelos.
        n_classes: número de classes; determina a métrica interna do AutoGluon.

    Returns:
        Tupla com:
            - predictor AutoGluon não treinado;
            - string de preset aceita pelo AutoGluon;
            - time_limit resolvido em segundos.
    """
    presets_map = {
        "default": "best_quality",
        "extreme": "extreme_quality",
    }
    if preset not in presets_map:
        raise ValueError(f"Preset desconhecido: {preset}. Use 'default' ou 'extreme'.")

    resolved_time_limit = (
        time_limit_seconds
        if time_limit_seconds is not None
        else DEFAULT_TIME_LIMIT_SECONDS[preset]
    )

    # roc_auc_ovo_macro só é válido para multiclasse no AutoGluon.
    eval_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"

    predictor = TabularPredictor(
        label=label,
        eval_metric=eval_metric,
        verbosity=2,
        path=path,
    )
    return predictor, presets_map[preset], resolved_time_limit
