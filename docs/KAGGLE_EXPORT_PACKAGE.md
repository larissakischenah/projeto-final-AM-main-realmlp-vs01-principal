# Kaggle Export Package

Este documento descreve o pacote limpo do projeto RealMLP/TabArena preparado para upload no Kaggle.

## Objetivo

Gerar um arquivo ZIP contendo apenas os arquivos necessários para execução no Kaggle, evitando o envio de arquivos pesados, caches, logs, ambientes locais, credenciais e artefatos de modelos.

## Script

O pacote é gerado por:

```bash
python scripts/build_kaggle_project_package.py
