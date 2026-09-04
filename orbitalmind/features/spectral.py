"""Data-driven period detection via the Lomb-Scargle periodogram.

Lomb-Scargle is designed for *irregularly* sampled series, so it estimates the
dominant cycles directly from the training data instead of us hardcoding orbital
periods. Feeding the real periods into the harmonic model lets it remove the true
systematic signal, leaving a whiter (more Gaussian) residual. Training-data only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lombscargle, find_peaks


def dominant_periods(t_seconds, y, n_periods: int = 3,
                     min_hours: float = 2.0, max_hours: float = 30.0,
                     grid: int = 4000):
    """Return up to ``n_periods`` dominant periods (in hours), strongest first.

    Falls back to an empty list when the channel has no usable variation.
    """
    t = np.asarray(t_seconds, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 6:
        return []
    t = t - t[0]
    y = y - np.mean(y)
    if not np.any(np.abs(y) > 1e-12):
        return []

    span_hours = (t[-1] - t[0]) / 3600.0
    # can't resolve periods longer than the observation span; cap sensibly
    hi = min(max_hours, max(span_hours / 1.5, min_hours * 2))
    periods_hours = np.linspace(min_hours, hi, grid)
    ang = 2.0 * np.pi / (periods_hours * 3600.0)
    try:
        power = lombscargle(t, y, ang, normalize=True)
    except Exception:
        return []

    peaks, _ = find_peaks(power)
    if peaks.size == 0:
        peaks = np.argsort(power)[-n_periods:]
    order = peaks[np.argsort(power[peaks])[::-1]]
    chosen = periods_hours[order[:n_periods]]
    return [float(p) for p in chosen]
