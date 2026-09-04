"""Residual-whitening wrapper (two-stage stacking).

Fit a base model, then fit a second 'corrector' model to the base's *training
residuals* to absorb any leftover systematic / autocorrelated structure. The final
prediction is base + corrector, so the scored residual is closer to iid-Gaussian
(higher SW/SF W). Works for arbitrary timestamps because the corrector is itself a
pure function of time.
"""
from __future__ import annotations

import copy

import numpy as np

from .. import PARAM_NAMES
from .base import Model


class StackedResidualModel(Model):
    def __init__(self, base: Model, corrector: Model, name: str | None = None):
        self.base = base
        self.corrector = corrector
        self.name = name or f"stack({base.name}+{corrector.name})"
        self._series_ref = None

    def fit(self, series):
        self.base.fit(series)
        base_pred = self.base.predict(series.t)
        resid_series = copy.copy(series)
        resid_series.values = {p: np.asarray(series.values[p], dtype=float)
                               - np.asarray(base_pred[p], dtype=float)
                               for p in PARAM_NAMES}
        self.corrector.fit(resid_series)
        return self

    def predict(self, t_seconds):
        base_pred = self.base.predict(t_seconds)
        corr_pred = self.corrector.predict(t_seconds)
        return {p: np.asarray(base_pred[p]) + np.asarray(corr_pred[p])
                for p in PARAM_NAMES}
