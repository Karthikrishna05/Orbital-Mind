"""Time-aware splits.

The real task: fit on 7 days, predict day-8 at arbitrary timestamps. We already
HAVE day 8 (the test CSVs), so ``day8_split`` is the faithful backtest. To avoid
over-reading a single day-8 realization, ``holdout_last_day`` provides an internal
rolling split within the 7-day training set.
"""
from __future__ import annotations

import copy

import numpy as np

from .. import PARAM_NAMES


def day8_split(train, test):
    """Return (train_series, query_t, actual_dict) for the real day-8 backtest."""
    actual = {p: np.asarray(test.values[p], dtype=float) for p in PARAM_NAMES}
    return train, np.asarray(test.t, dtype=float), actual


def holdout_last_day(series, holdout_days: float = 1.0):
    """Split a series into (fit_series, query_t, actual_dict) by holding out the
    final ``holdout_days`` as the query window (internal validation)."""
    t = np.asarray(series.t, dtype=float)
    if t.size == 0:
        raise ValueError("empty series")
    cutoff = t[-1] - holdout_days * 86400.0
    fit_mask = t <= cutoff
    q_mask = ~fit_mask
    if fit_mask.sum() < 5 or q_mask.sum() < 3:
        # fall back to a 80/20 index split for short series
        k = max(int(t.size * 0.8), 5)
        fit_mask = np.zeros(t.size, dtype=bool); fit_mask[:k] = True
        q_mask = ~fit_mask

    fit_series = _subset(series, fit_mask)
    query_t = t[q_mask]
    actual = {p: np.asarray(series.values[p])[q_mask] for p in PARAM_NAMES}
    return fit_series, query_t, actual


def rolling_day_folds(series, n_folds: int = 3, horizon_days: float = 1.0):
    """Yield up to ``n_folds`` expanding-window folds for leak-free validation.

    Each fold holds out one day-long horizon at the end of a progressively shorter
    history: fit on everything before the horizon, validate on points inside it.
    This averages out the noise of a single small holdout (crucial for the tiny
    MEO series). The day-8 test file is never involved.
    """
    t = np.asarray(series.t, dtype=float)
    if t.size == 0:
        return
    t_end = t[-1]
    made = 0
    for k in range(n_folds):
        hi = t_end - k * horizon_days * 86400.0
        lo = hi - horizon_days * 86400.0
        val_mask = (t > lo) & (t <= hi)
        fit_mask = t <= lo
        if fit_mask.sum() < 5 or val_mask.sum() < 3:
            continue
        fit_series = _subset(series, fit_mask)
        val_t = t[val_mask]
        val_actual = {p: np.asarray(series.values[p])[val_mask] for p in PARAM_NAMES}
        yield fit_series, val_t, val_actual
        made += 1
    if made == 0:
        # fall back to a single 80/20 index split for very short series
        try:
            yield holdout_last_day(series)
        except Exception:
            return


def _subset(series, mask):
    s = copy.copy(series)
    s.t = np.asarray(series.t)[mask]
    s.datetimes = np.asarray(series.datetimes, dtype=object)[mask]
    s.values = {p: np.asarray(series.values[p])[mask] for p in PARAM_NAMES}
    return s
