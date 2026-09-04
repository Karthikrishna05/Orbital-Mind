"""Per-channel model selector (a meta-model), selected by multi-fold validation.

The competition score is averaged over the 4 channels (X, Y, Z, clock), so the
average is maximized by choosing the best model *for each channel independently*.

Selection is leak-free and robust:
  * scored on rolling multi-fold validation *inside the training data* (never the
    day-8 test file),
  * averaged over folds (not a single noisy holdout),
  * guarded on both spread (std) and bias (|mean|) so it does not pick a
    "spread-gamer" or a drift-prone extrapolator,
then the chosen model per channel is refit on the full training history.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from ..backtest.splits import rolling_day_folds
from ..evaluation import swtest
from .base import Model
from .persistence import MeanModel


def _perchannel_rolling(factory, series, kind, n_folds=3):
    """Average per-channel (W, std, mean_abs) over rolling folds for one model."""
    acc = {p: {"W": [], "std": [], "mean": []} for p in PARAM_NAMES}
    for fit_s, val_t, val_actual in rolling_day_folds(series, n_folds=n_folds):
        try:
            m = factory(kind).fit(fit_s)
            pred = m.predict(val_t)
        except Exception:
            continue
        for p in PARAM_NAMES:
            r = np.asarray(pred[p]) - np.asarray(val_actual[p])
            r = r[np.isfinite(r)]
            if r.size < 3 or (r.max() - r.min()) <= 1e-19:
                continue
            acc[p]["W"].append(swtest(r).W)
            acc[p]["std"].append(float(np.std(r)))
            acc[p]["mean"].append(float(abs(np.mean(r))))
    out = {}
    for p in PARAM_NAMES:
        if acc[p]["W"]:
            out[p] = (float(np.mean(acc[p]["W"])), float(np.mean(acc[p]["std"])),
                      float(np.mean(acc[p]["mean"])))
        else:
            out[p] = (float("nan"), float("inf"), float("inf"))
    return out


class PerChannelSelector(Model):
    def __init__(self, candidate_factories, kind: str = "GEO",
                 std_tol: float = 0.25, n_folds: int = 3, name: str = "per_channel_best"):
        # candidate_factories: list of (label, factory(kind)->Model)
        self.candidate_factories = candidate_factories
        self.kind = kind
        self.std_tol = std_tol
        self.n_folds = n_folds
        self.name = name
        self._chosen = {}
        self._full_models = {}

    def fit(self, series):
        # per-channel baseline spread (constant predictor) for the guards
        base_metrics = _perchannel_rolling(lambda k: MeanModel(robust=True),
                                           series, self.kind, self.n_folds)
        base_std = {p: (base_metrics[p][1] if np.isfinite(base_metrics[p][1]) else 1e-9)
                    for p in PARAM_NAMES}

        # score every candidate per channel on rolling folds
        cand_metrics = {}
        for label, factory in self.candidate_factories:
            cand_metrics[label] = _perchannel_rolling(factory, series, self.kind, self.n_folds)

        # choose per channel: max W among those within std & bias guards
        for p in PARAM_NAMES:
            scored = []
            for label, _ in self.candidate_factories:
                W, std, mean_abs = cand_metrics[label][p]
                if not np.isfinite(W):
                    continue
                scored.append((label, W, std, mean_abs))
            if not scored:
                self._chosen[p] = self.candidate_factories[0][0]
                continue
            guarded = [c for c in scored
                       if c[2] <= base_std[p] * (1.0 + self.std_tol)
                       and c[3] <= base_std[p]]
            pool = guarded if guarded else scored
            self._chosen[p] = max(pool, key=lambda c: c[1])[0]

        # refit each distinct chosen model on the FULL training history
        label2factory = dict(self.candidate_factories)
        for label in set(self._chosen.values()):
            self._full_models[label] = label2factory[label](self.kind).fit(series)
        return self

    def predict(self, t_seconds):
        preds = {label: m.predict(t_seconds) for label, m in self._full_models.items()}
        return {p: np.asarray(preds[self._chosen[p]][p], dtype=float) for p in PARAM_NAMES}

    def chosen_summary(self) -> str:
        return ", ".join(f"{p}:{self._chosen.get(p, '?')}" for p in PARAM_NAMES)
