"""Time / orbital-harmonic features for arbitrary timestamps.

Everything is a pure function of the query timestamps, so models fit on the 7-day
history can predict at *any* future timestamp (the competition's requirement).
"""
from __future__ import annotations

import numpy as np


def days_since(t_seconds: np.ndarray, t0: float) -> np.ndarray:
    """Elapsed time in days since reference epoch ``t0`` (seconds)."""
    return (np.asarray(t_seconds, dtype=float) - t0) / 86400.0


def fourier_design(t_seconds: np.ndarray, t0: float, periods_hours,
                   n_poly: int = 2, harmonics: int = 1,
                   standardize_tau: float = 1.0) -> np.ndarray:
    """Design matrix [intercept, polynomial trend, sin/cos harmonics].

    Parameters
    ----------
    t_seconds : array
        Query timestamps (epoch seconds).
    t0 : float
        Reference epoch (usually training start) for the trend origin.
    periods_hours : iterable of float
        Periodicities (hours) to encode as sin/cos pairs.
    n_poly : int
        Polynomial trend degree (0 = intercept only, 1 = linear, ...).
    harmonics : int
        Number of harmonics per listed period (k = 1..harmonics).
    standardize_tau : float
        Divisor (in days) to scale the trend variable for conditioning.
    """
    tau = days_since(t_seconds, t0)
    cols = [np.ones_like(tau)]
    ts = tau / standardize_tau
    for d in range(1, n_poly + 1):
        cols.append(ts ** d)
    for P in periods_hours:
        period_days = P / 24.0
        for k in range(1, harmonics + 1):
            w = 2.0 * np.pi * k / period_days
            cols.append(np.sin(w * tau))
            cols.append(np.cos(w * tau))
    return np.column_stack(cols)
