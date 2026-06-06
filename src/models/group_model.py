"""Modelo principal do grupo: RealMLP (NeurIPS 2024, via pytabkit).

RealMLP_TD_S_Classifier usa defaults meta-tunados (sufixo _TD) obtidos por
meta-aprendizado em centenas de datasets tabulares (Holzmuller et al., 2024).
"""

from __future__ import annotations


def build_group_model(seed: int = 42):
    from pytabkit import RealMLP_TD_S_Classifier
    # max_one_hot_cat_size=9: limita OHE a colunas com ≤9 categorias; colunas de
    # alta cardinalidade recebem embeddings aprendidos em vez de OHE irrestrito.
    # Sem esse limite, o preset _TD_S usa max_one_hot_cat_size=-1 (sem limite),
    # o que pode inflar a entrada de centenas para dezenas de milhares de colunas
    # em datasets com categorias de alta cardinalidade (ex.: kddcup09_appetency:
    # 212 → 71.678 dims), causando OOM na GPU.
    return RealMLP_TD_S_Classifier(
        random_state=seed,
        max_one_hot_cat_size=9,
        embedding_size=8,
    )


def build_group_model_hpo(seed: int = 42, time_limit_s: int = 180):
    """RealMLP com busca interna de hiperparâmetros (HPO via pytabkit).

    Usado como evidência de otimização de hiperparâmetros na rubrica (10%).
    time_limit_s controla o orçamento de busca por dataset.
    """
    from pytabkit import RealMLP_HPO_Classifier
    return RealMLP_HPO_Classifier(random_state=seed, time_limit_s=time_limit_s)
