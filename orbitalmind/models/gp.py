"""Gaussian Process regression -- a principled challenger.

Why GP fits this competition especially well:
  * predictive residuals are Gaussian *by construction* (the SW/SF objective),
  * it handles arbitrary/irregular query timestamps natively,
  * it yields a predictive std -> the confidence interval the Note asks for.

Per parameter: kernel = C*Matern(trend) + C*ExpSineSquared(orbital period) + White.
Time is measured in days; y is normalized internally.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C, Matern, ExpSineSquared, WhiteKernel,
)

from .. import PARAM_NAMES
from ..config import DEFAULT_PERIODS_HOURS
from ..features import days_since
from .base import Model


class GPModel(Model):
    def __init__(self, kind: str = "GEO", period_hours: float | None = None,
                 n_restarts: int = 4, alpha: float = 1e-6, n_periods: int = 2,
                 max_noise_frac: float = 0.5):
        self.kind = kind
        periods = DEFAULT_PERIODS_HOURS.get(kind, DEFAULT_PERIODS_HOURS["GEO"])
        self.periods_days = ([period_hours / 24.0] if period_hours
                             else [P / 24.0 for P in periods[:n_periods]])
        self.n_restarts = n_restarts
        self.alpha = alpha
        # Cap the white-noise term so it cannot absorb all the structure (the
        # failure mode that made the untuned GP collapse to the mean).
        self.max_noise_frac = max_noise_frac
        self.name = "gp_matern_periodic"
        self._t0 = 0.0
        self._gp = {}
        self._pred_std = {}

    def _kernel(self):
        # y is normalized (unit variance) via normalize_y=True, so scales ~1.
        kern = C(1.0, (1e-4, 1e4)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)
        for pd in self.periods_days:
            kern = kern + C(1.0, (1e-4, 1e4)) * ExpSineSquared(
                length_scale=1.0, periodicity=pd,
                length_scale_bounds=(1e-2, 1e3), periodicity_bounds="fixed")
        # Cap the white-noise term so it cannot absorb all the structure.
        kern = kern + WhiteKernel(noise_level=0.1 * self.max_noise_frac,
                                  noise_level_bounds=(1e-8, self.max_noise_frac))
        return kern

    def fit(self, series):
        self._t0 = series.t0()
        X = days_since(series.t, self._t0).reshape(-1, 1)
        for p in PARAM_NAMES:
            y = series.values[p]
            gp = GaussianProcessRegressor(
                kernel=self._kernel(), alpha=self.alpha,
                n_restarts_optimizer=self.n_restarts, normalize_y=True,
                random_state=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
            self._gp[p] = gp
        return self

    def predict(self, t_seconds):
        X = days_since(np.asarray(t_seconds, dtype=float), self._t0).reshape(-1, 1)
        out = {}
        for p in PARAM_NAMES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mean, std = self._gp[p].predict(X, return_std=True)
            out[p] = mean
            self._pred_std[p] = std
        return out
