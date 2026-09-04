"""Model interface. Every predictor implements fit()/predict() so the backtest
scoreboard compares them apples-to-apples.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import PARAM_NAMES


class Model(ABC):
    """Abstract predictor.

    fit(series) learns from a :class:`~orbitalmind.dataio.SeriesData` (7-day
    history); predict(t_seconds) returns a dict {param: ndarray} of predicted
    errors at the given absolute epoch-second timestamps.
    """

    name: str = "model"

    @abstractmethod
    def fit(self, series) -> "Model":
        ...

    @abstractmethod
    def predict(self, t_seconds: np.ndarray) -> dict:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class PredictMixin:
    """Helper to assemble a per-parameter prediction dict with a common length."""

    @staticmethod
    def empty_like(t_seconds: np.ndarray) -> dict:
        n = np.asarray(t_seconds).size
        return {p: np.zeros(n, dtype=float) for p in PARAM_NAMES}
