"""Floor baselines.

These set the bar every real model must beat on the SW scoreboard. Because the
data are *errors* already centered near zero, the constant predictors are
surprisingly strong references for the normality objective.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from .base import Model, PredictMixin


class ZeroModel(Model, PredictMixin):
    """Predict 0 for every parameter (the data are near-zero errors)."""

    name = "zero"

    def fit(self, series):
        return self

    def predict(self, t_seconds):
        return self.empty_like(t_seconds)


class MeanModel(Model, PredictMixin):
    """Predict a constant per parameter: robust (median) or mean of training."""

    def __init__(self, robust: bool = True):
        self.robust = robust
        self.name = "median" if robust else "mean"
        self._const = {}

    def fit(self, series):
        for p in PARAM_NAMES:
            v = series.values[p]
            self._const[p] = float(np.median(v) if self.robust else np.mean(v))
        return self

    def predict(self, t_seconds):
        n = np.asarray(t_seconds).size
        return {p: np.full(n, self._const[p]) for p in PARAM_NAMES}


class PersistenceModel(Model, PredictMixin):
    """Predict the last observed value per parameter (constant extrapolation)."""

    name = "persistence"

    def __init__(self):
        self._last = {}

    def fit(self, series):
        for p in PARAM_NAMES:
            self._last[p] = float(series.values[p][-1]) if len(series) else 0.0
        return self

    def predict(self, t_seconds):
        n = np.asarray(t_seconds).size
        return {p: np.full(n, self._last[p]) for p in PARAM_NAMES}
