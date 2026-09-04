"""Regime-matched training.

Idea (independently used by another SIH team): the error series is sampled in two
operational regimes -- a coarse ~2h cadence and dense ~15min bursts -- and these
behave differently (the bursts carry the outliers/upload events). Training one
model on the mixed record blends two regimes. Instead, fit a base model per regime
and, at prediction time, route each query timestamp to the model trained on the
regime whose cadence matches the query's local cadence.

Everything here is training-only; no test truth is used.
"""
from __future__ import annotations

import copy

import numpy as np

from .. import PARAM_NAMES
from .base import Model


def _local_gap_minutes(t):
    """Per-point local sampling gap = min(gap to previous, gap to next), in minutes."""
    t = np.asarray(t, dtype=float)
    n = t.size
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([np.inf])
    d = np.diff(t)
    prev = np.concatenate([[np.inf], d])
    nxt = np.concatenate([d, [np.inf]])
    return np.minimum(prev, nxt) / 60.0


class RegimeMatchedModel(Model):
    def __init__(self, base_factory, kind="GEO", dense_gap_min=45.0,
                 min_points=20, name=None):
        self.base_factory = base_factory
        self.kind = kind
        self.dense_gap_min = dense_gap_min
        self.min_points = min_points
        self.name = name or "regime_matched"
        self._dense = None
        self._coarse = None
        self._fallback = None

    def _subset(self, series, mask):
        s = copy.copy(series)
        s.t = np.asarray(series.t)[mask]
        s.datetimes = np.asarray(series.datetimes, dtype=object)[mask]
        s.values = {p: np.asarray(series.values[p])[mask] for p in PARAM_NAMES}
        return s

    def fit(self, series):
        gaps = _local_gap_minutes(series.t)
        dense_mask = gaps <= self.dense_gap_min
        coarse_mask = ~dense_mask

        self._fallback = self.base_factory(self.kind).fit(series)
        self._dense = (self.base_factory(self.kind).fit(self._subset(series, dense_mask))
                       if dense_mask.sum() >= self.min_points else self._fallback)
        self._coarse = (self.base_factory(self.kind).fit(self._subset(series, coarse_mask))
                        if coarse_mask.sum() >= self.min_points else self._fallback)
        return self

    def predict(self, t_seconds):
        t = np.asarray(t_seconds, dtype=float)
        gaps = _local_gap_minutes(t)
        dense_q = gaps <= self.dense_gap_min

        out = {p: np.empty(t.size) for p in PARAM_NAMES}
        # route dense query points to the dense-trained model, coarse to coarse
        for mask, model in ((dense_q, self._dense), (~dense_q, self._coarse)):
            if not mask.any():
                continue
            pred = model.predict(t[mask])
            for p in PARAM_NAMES:
                out[p][mask] = np.asarray(pred[p], dtype=float)
        return out
