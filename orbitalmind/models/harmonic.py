"""Harmonic + polynomial-trend regression -- the bar-to-beat.

Fits, per parameter, a linear model on [trend, orbital sin/cos] features. Robust
(Huber) fitting keeps the outlier bursts from biasing the mean function; the goal
is to remove systematic drift + periodicity so the residual is iid-Gaussian
(maximizing the SW/SF score), not to minimize RMSE.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression

from .. import PARAM_NAMES
from ..config import DEFAULT_PERIODS_HOURS
from ..features import fourier_design
from .base import Model


class HarmonicModel(Model):
    def __init__(self, kind: str = "GEO", periods_hours=None, n_poly: int = 2,
                 harmonics: int = 1, robust: bool = True, huber_epsilon: float = 1.35):
        self.kind = kind
        self.periods_hours = (periods_hours if periods_hours is not None
                              else DEFAULT_PERIODS_HOURS.get(kind, DEFAULT_PERIODS_HOURS["GEO"]))
        self.n_poly = n_poly
        self.harmonics = harmonics
        self.robust = robust
        self.huber_epsilon = huber_epsilon
        self.name = ("harmonic_robust" if robust else "harmonic_ols") + \
            f"_p{n_poly}h{harmonics}"
        self._t0 = 0.0
        self._models = {}

    def _design(self, t_seconds):
        return fourier_design(t_seconds, self._t0, self.periods_hours,
                              n_poly=self.n_poly, harmonics=self.harmonics)

    def fit(self, series):
        self._t0 = series.t0()
        X = self._design(series.t)
        for p in PARAM_NAMES:
            y = series.values[p]
            if self.robust and X.shape[0] > X.shape[1] + 2:
                try:
                    est = HuberRegressor(epsilon=self.huber_epsilon, max_iter=3000,
                                         fit_intercept=False, alpha=1e-6, tol=1e-5)
                    est.fit(X, y)
                except Exception:
                    est = LinearRegression(fit_intercept=False).fit(X, y)
            else:
                est = LinearRegression(fit_intercept=False).fit(X, y)
            self._models[p] = est
        return self

    def predict(self, t_seconds):
        X = self._design(np.asarray(t_seconds, dtype=float))
        return {p: self._models[p].predict(X) for p in PARAM_NAMES}
