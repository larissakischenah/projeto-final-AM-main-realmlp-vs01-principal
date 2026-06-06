"""Carregamento padronizado dos 30 datasets do TabArena-v0.1.

Os 30 datasets sao selecionados a partir dos 51 datasets curados do TabArena-v0.1
(NeurIPS 2025), estratificados por regime de tamanho. Cada dataset e carregado
via OpenML, com cache local para evitar download repetido.

Para a lista oficial de task IDs do TabArena, consulte:
    https://tabarena.ai
    https://github.com/autogluon/tabarena

A constante RECOMMENDED_TASK_IDS abaixo contem 30 IDs estratificados por tamanho.
Caso a lista oficial seja atualizada, basta substituir os IDs aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import openml
import pandas as pd

CACHE_DIR = Path(os.environ.get("TABARENA_CACHE", "./cache/tabarena"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
openml.config.set_root_cache_directory(str(CACHE_DIR))

# Limiar superior do regime *small* elevado de 1.000 -> 2.000: o TabArena-v0.1 so oferece
# 3 datasets de classificacao com n<1.000 (insuficiente para os 10 exigidos). Os demais
# limiares foram preservados.
REGIME_THRESHOLDS = {"small": 2_000, "medium": 10_000}


RECOMMENDED_TASK_IDS: list[int] = [
    # --- SMALL (n < 2.000): 10 datasets ---
    363621,  # blood-transfusion-service-center   n=748    bin
    363629,  # diabetes                           n=768    bin
    363614,  # anneal                             n=898    multi(5)
    363626,  # credit-g                           n=1000   bin
    363685,  # maternal_health_risk               n=1014   multi(3)
    363696,  # qsar-biodeg                        n=1054   bin
    363707,  # website_phishing                   n=1353   multi(3)
    363671,  # Fitness_Club                       n=1500   bin   (missing)
    363711,  # MIC                                n=1699   multi(8) (missing)
    363682,  # Is-this-a-good-customer            n=1723   bin
    # --- MEDIUM (2.000 <= n < 10.000): 10 datasets ---
    363684,  # Marketing_Campaign                 n=2240   bin   (missing)
    363674,  # hazelnut-spread-contaminant        n=2400   bin
    363700,  # seismic-bumps                      n=2584   bin
    363702,  # splice                             n=3190   multi(3)
    363620,  # Bioresponse                        n=3751   bin
    363677,  # hiva_agnostic                      n=3845   multi(3)
    363704,  # students_dropout_and_academic      n=4424   multi(3)
    363623,  # churn                              n=5000   bin
    363694,  # polish_companies_bankruptcy        n=5910   bin   (missing)
    363706,  # taiwanese_bankruptcy_prediction    n=6819   bin
    # --- LARGE (n >= 10.000): 10 datasets ---
    363619,  # Bank_Customer_Churn                n=10000  bin
    363676,  # heloc                              n=10459  bin
    363712,  # jm1                                n=10885  bin   (missing)
    363632,  # E-CommereShippingData              n=10999  bin
    363691,  # online_shoppers_intention          n=12330  bin
    363681,  # in_vehicle_coupon_recommendation   n=12684  bin
    363679,  # HR_Analytics_Job_Change            n=19158  bin   (missing)
    363627,  # credit_card_clients_default        n=30000  bin
    363613,  # Amazon_employee_access             n=32769  bin
    363699,  # SDSS17                             n=78053  multi(3)
]
"""30 task IDs do TabArena-v0.1 (OpenML benchmark suite, study id=457), estratificados por
regime de tamanho: 10 small (n<2.000), 10 medium (2.000<=n<10.000), 10 large (n>=10.000).

Observacao: o limiar superior do regime *small* foi elevado de 1.000 para 2.000 porque o
TabArena-v0.1 oferece apenas 3 datasets de classificacao com n<1.000 (insuficiente para os
10 exigidos); os demais limiares foram preservados. Cada regime contem pelo menos um
problema de classificacao binaria e um de multiclasse.
"""


@dataclass(frozen=True)
class TabularDataset:
    task_id: int
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categorical_indicator: list[bool]
    attribute_names: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    @property
    def n_categorical(self) -> int:
        return int(sum(self.categorical_indicator))

    @property
    def has_missing(self) -> bool:
        return bool(self.X.isna().any().any())

    @property
    def regime(self) -> str:
        return classify_regime(self.n_samples)


def classify_regime(n_samples: int) -> str:
    """Retorna 'small', 'medium' ou 'large' conforme o numero de amostras."""
    if n_samples < REGIME_THRESHOLDS["small"]:
        return "small"
    if n_samples < REGIME_THRESHOLDS["medium"]:
        return "medium"
    return "large"


def load_task(task_id: int) -> TabularDataset:
    """Carrega um dataset do OpenML pelo task_id, com cache."""
    task = openml.tasks.get_task(task_id, download_data=True)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    return TabularDataset(
        task_id=task_id,
        name=dataset.name,
        X=X,
        y=np.asarray(y),
        categorical_indicator=list(categorical_indicator),
        attribute_names=list(attribute_names),
    )


def iter_datasets(task_ids: list[int] | None = None) -> Iterator[TabularDataset]:
    """Itera sobre todos os datasets configurados."""
    ids = task_ids if task_ids is not None else RECOMMENDED_TASK_IDS
    for task_id in ids:
        yield load_task(task_id)


def summarize(task_ids: list[int] | None = None) -> pd.DataFrame:
    """Tabela-resumo (n_samples, n_features, n_classes, regime, missing)."""
    rows = []
    for ds in iter_datasets(task_ids):
        rows.append(
            {
                "task_id": ds.task_id,
                "name": ds.name,
                "n_samples": ds.n_samples,
                "n_features": ds.n_features,
                "n_classes": ds.n_classes,
                "n_categorical": ds.n_categorical,
                "has_missing": ds.has_missing,
                "regime": ds.regime,
            }
        )
    return pd.DataFrame(rows)
