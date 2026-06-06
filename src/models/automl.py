"""AutoGluon 1.4 nos presets `default` (curto) e `extreme` (4 horas).

O preset `extreme` inclui internamente foundation models como TabPFN v2 e
representa o teto de comparacao do projeto.
"""

from __future__ import annotations

from autogluon.tabular import TabularPredictor


def build_autogluon(
    label: str,
    seed: int = 42,
    preset: str = "default",
    time_limit_seconds: int | None = None,
    path: str | None = None,
    n_classes: int = 2,
) -> TabularPredictor:
    """Constroi um TabularPredictor do AutoGluon.

    Args:
        label: nome da coluna alvo no DataFrame de treino.
        seed: semente fixa para reprodutibilidade.
        preset: 'default' (rapido) ou 'extreme' (4h, com foundation models).
        time_limit_seconds: substitui o default do preset, se informado.
        path: diretório onde AutoGluon grava os modelos (padrão: AutogluonModels/).
        n_classes: número de classes; determina a métrica interna do AutoGluon.
    """
    presets_map = {
        "default": "best_quality",
        "extreme": "extreme_quality",
    }
    if preset not in presets_map:
        raise ValueError(f"Preset desconhecido: {preset}. Use 'default' ou 'extreme'.")

    # roc_auc_ovo_macro só é válido para multiclasse no AutoGluon
    eval_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"

    predictor = TabularPredictor(
        label=label,
        eval_metric=eval_metric,
        verbosity=2,
        path=path,
    )
    return predictor, presets_map[preset], time_limit_seconds
