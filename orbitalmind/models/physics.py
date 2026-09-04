"""Physics-informed harmonic model: Fourier/orbital basis + solar-geometry features.

Appends the deterministic solar-geometry proxies (declination, equation of time,
Greenwich hour angle, eclipse-season terms) to the harmonic design matrix. Aimed
at GEO, whose thermal/eclipse-driven structure the plain harmonic may miss.
Robustly fit per channel; a small L2 keeps the extra columns from overfitting the
short series.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge

from .. import PARAM_NAMES
from ..config import DEFAULT_PERIODS_HOURS
from ..features import fourier_design
from ..features.astronomical import astro_features
from .base import Model


class PhysicsHarmonicModel(Model):
    def __init__(self, kind="GEO", n_poly=2, harmonics=2, robust=True):
        self.kind = kind
        self.periods_hours = DEFAULT_PERIODS_HOURS.get(kind, DEFAULT_PERIODS_HOURS["GEO"])
        self.n_poly = n_poly
        self.harmonics = harmonics
        self.robust = robust
        self.name = "physics_harmonic"
        self._t0 = 0.0
        self._models = {}

    def _design(self, t_seconds):
        base = fourier_design(t_seconds, self._t0, self.periods_hours,
                              n_poly=self.n_poly, harmonics=self.harmonics)
        astro, _ = astro_features(t_seconds)
        return np.column_stack([base, astro])

    def fit(self, series):
        self._t0 = series.t0()
        X = self._design(series.t)
        for p in PARAM_NAMES:
            y = series.values[p]
            if self.robust and X.shape[0] > X.shape[1] + 2:
                try:
                    est = HuberRegressor(epsilon=1.35, max_iter=3000,
                                         fit_intercept=False, alpha=1e-3, tol=1e-5)
                    est.fit(X, y)
                except Exception:
                    est = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)
            else:
                est = Ridge(alpha=1.0, fit_intercept=False).fit(X, y)
            self._models[p] = est
        return self

    def predict(self, t_seconds):
        X = self._design(np.asarray(t_seconds, dtype=float))
        return {p: self._models[p].predict(X) for p in PARAM_NAMES}
