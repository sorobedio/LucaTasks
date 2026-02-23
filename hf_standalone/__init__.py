#!/usr/bin/env python
# encoding: utf-8
"""
Lightweight HuggingFace-backed embedding predictors.
"""

from .predict_embedding import (
    predict_embedding_luca,
    predict_embedding_esm,
    predict_embedding_dnabert2,
    predict_embedding_dnaberts,
)

__all__ = [
    "predict_embedding_luca",
    "predict_embedding_esm",
    "predict_embedding_dnabert2",
    "predict_embedding_dnaberts",
]

