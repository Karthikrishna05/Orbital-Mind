"""Composite per-channel model.

Routes each of the four channels to a *named* sub-model, so a principled,
domain-motivated composition can be expressed directly (e.g. take the stacked
model's strong position forecasts for x/y/z, and the change-point clock model's
forecast for the clock channel) without wholesale replacement. Distinct sub-models
are fit once each; channels sharing a sub-model reuse it.

Unlike the per-channel *selector*, this composition is fixed by design (not chosen
from validation), so it carries no selection-leakage — it encodes prior knowledge
about which model suits which channel.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from .base import Model


class CompositeChannelModel(Model):
    def __init__(self, channel_to_label, factories, kind="GEO", name="composite"):
        # channel_to_label: {'x': 'pos', 'y': 'pos', 'z': 'pos', 'clock': 'clk'}
        # factories:        {'pos': fn(kind)->Model, 'clk': fn(kind)->Model}
        self.channel_to_label = channel_to_label
        self.factories = factories
        self.kind = kind
        self.name = name
        self._fitted = {}

    def fit(self, series):
        for label in set(self.channel_to_label.values()):
            self._fitted[label] = self.factories[label](self.kind).fit(series)
        return self

    def predict(self, t_seconds):
        preds = {label: m.predict(t_seconds) for label, m in self._fitted.items()}
        return {ch: np.asarray(preds[self.channel_to_label[ch]][ch], dtype=float)
                for ch in PARAM_NAMES}
