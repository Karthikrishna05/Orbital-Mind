"""Nonlinear ML challengers on time / orbital-harmonic features.

These are kept only if they beat the robust-harmonic bar on the leaderboard --
on 46-143 points they are prone to overfitting, so they compete, they don't win
by default.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import PARAM_NAMES
from ..config import DEFAULT_PERIODS_HOURS
from ..features import fourier_design
from .base import Model


class _FeatureModel(Model):
    """Shared feature construction for tabular ML regressors."""

    def __init__(self, kind="GEO", n_poly=2, harmonics=2):
        self.kind = kind
        self.periods_hours = DEFAULT_PERIODS_HOURS.get(kind, DEFAULT_PERIODS_HOURS["GEO"])
        self.n_poly = n_poly
        self.harmonics = harmonics
        self._t0 = 0.0
        self._est = {}

    def _design(self, t_seconds):
        return fourier_design(t_seconds, self._t0, self.periods_hours,
                              n_poly=self.n_poly, harmonics=self.harmonics)

    def _make(self):
        raise NotImplementedError

    def fit(self, series):
        self._t0 = series.t0()
        X = self._design(series.t)
        for p in PARAM_NAMES:
            est = self._make()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X, series.values[p])
            self._est[p] = est
        return self

    def predict(self, t_seconds):
        X = self._design(np.asarray(t_seconds, dtype=float))
        return {p: self._est[p].predict(X) for p in PARAM_NAMES}


class GradientBoostingModel(_FeatureModel):
    def __init__(self, kind="GEO", **kw):
        super().__init__(kind=kind, **kw)
        self.name = "gbr"

    def _make(self):
        # 'huber' loss -> robust to the outlier bursts.
        return GradientBoostingRegressor(
            loss="huber", n_estimators=200, max_depth=2,
            learning_rate=0.05, subsample=0.9, random_state=0)


class MLPModel(_FeatureModel):
    def __init__(self, kind="GEO", **kw):
        super().__init__(kind=kind, **kw)
        self.name = "mlp"

    def _make(self):
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh",
                         alpha=1e-2, max_iter=2000, random_state=0))
