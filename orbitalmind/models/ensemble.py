"""Ensemble: combine several base models' predictions.

Averaging decent-but-different predictors tends to cancel model-specific quirks
and leave a more Gaussian residual. Median combination is robust to a single
member misbehaving on a channel.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from .base import Model


class EnsembleModel(Model):
    def __init__(self, members, kind: str = "GEO", combine: str = "median",
                 name: str | None = None):
        # members: list of (label, factory(kind)->Model)
        self.members = members
        self.kind = kind
        self.combine = combine
        self.name = name or f"ensemble_{combine}"
        self._fitted = []

    def fit(self, series):
        self._fitted = []
        for label, factory in self.members:
            try:
                self._fitted.append(factory(self.kind).fit(series))
            except Exception:
                continue
        return self

    def predict(self, t_seconds):
        preds = [m.predict(t_seconds) for m in self._fitted]
        out = {}
        for p in PARAM_NAMES:
            stack = np.vstack([np.asarray(pr[p], dtype=float) for pr in preds])
            out[p] = np.median(stack, axis=0) if self.combine == "median" \
                else np.mean(stack, axis=0)
        return out
