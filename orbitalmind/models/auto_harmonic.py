"""Harmonic model whose periods are detected per channel from the training data
(via Lomb-Scargle), instead of hardcoded orbital periods.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression

from .. import PARAM_NAMES
from ..config import DEFAULT_PERIODS_HOURS
from ..features import fourier_design, dominant_periods
from .base import Model


class AutoHarmonicModel(Model):
    def __init__(self, kind: str = "GEO", n_periods: int = 3, n_poly: int = 2,
                 harmonics: int = 1, robust: bool = True):
        self.kind = kind
        self.n_periods = n_periods
        self.n_poly = n_poly
        self.harmonics = harmonics
        self.robust = robust
        self.name = f"auto_harmonic_np{n_periods}p{n_poly}h{harmonics}"
        self._t0 = 0.0
        self._periods = {}   # channel -> [hours]
        self._models = {}

    def fit(self, series):
        self._t0 = series.t0()
        fallback = DEFAULT_PERIODS_HOURS.get(self.kind, DEFAULT_PERIODS_HOURS["GEO"])
        for p in PARAM_NAMES:
            y = series.values[p]
            periods = dominant_periods(series.t, y, n_periods=self.n_periods)
            if not periods:
                periods = list(fallback[:self.n_periods])
            self._periods[p] = periods
            X = fourier_design(series.t, self._t0, periods,
                               n_poly=self.n_poly, harmonics=self.harmonics)
            if self.robust and X.shape[0] > X.shape[1] + 2:
                try:
                    est = HuberRegressor(epsilon=1.35, max_iter=3000,
                                         fit_intercept=False, alpha=1e-6, tol=1e-5)
                    est.fit(X, y)
                except Exception:
                    est = LinearRegression(fit_intercept=False).fit(X, y)
            else:
                est = LinearRegression(fit_intercept=False).fit(X, y)
            self._models[p] = est
        return self

    def predict(self, t_seconds):
        t = np.asarray(t_seconds, dtype=float)
        out = {}
        for p in PARAM_NAMES:
            X = fourier_design(t, self._t0, self._periods[p],
                               n_poly=self.n_poly, harmonics=self.harmonics)
            out[p] = self._models[p].predict(X)
        return out

    def periods_summary(self) -> str:
        return "; ".join(f"{p}:[{','.join(f'{h:.1f}h' for h in self._periods.get(p, []))}]"
                         for p in PARAM_NAMES)
