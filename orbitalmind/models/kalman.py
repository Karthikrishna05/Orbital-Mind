"""Self-contained local-linear-trend Kalman filter (no statsmodels dependency).

State per parameter: [level, slope]. The transition uses the actual gap dt between
irregular observations (continuous white-noise-acceleration model), so it handles
non-uniform sampling natively. Good for smooth clock drift; for strongly periodic
channels it will lose to the harmonic model -- which is exactly what the
leaderboard is for.

Prediction at an arbitrary future timestamp projects the final filtered state
forward: level + slope * (t - t_last).
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from ..dataio.clean import robust_stats
from .base import Model


def _filter(t, y, q, r):
    """Run the LLT Kalman filter; return final (state, cov, t_last) and the
    accumulated one-step negative log-likelihood (for hyperparameter selection)."""
    x = np.array([y[0], 0.0])
    P = np.array([[r, 0.0], [0.0, r]])
    nll = 0.0
    H = np.array([1.0, 0.0])
    for i in range(1, len(t)):
        dt = max(t[i] - t[i - 1], 1e-9)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = q * np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                          [dt ** 2 / 2.0, dt]])
        x = F @ x
        P = F @ P @ F.T + Q
        # innovation
        yhat = H @ x
        S = H @ P @ H + r
        v = y[i] - yhat
        K = (P @ H) / S
        x = x + K * v
        P = P - np.outer(K, H @ P)
        nll += 0.5 * (np.log(2 * np.pi * S) + v * v / S)
    return x, P, t[-1], nll


class KalmanLLTModel(Model):
    def __init__(self, kind: str = "GEO"):
        self.kind = kind
        self.name = "kalman_llt"
        self._state = {}
        self._t_last = {}

    def fit(self, series):
        t = np.asarray(series.t, dtype=float)
        for p in PARAM_NAMES:
            y = np.asarray(series.values[p], dtype=float)
            if t.size < 3:
                self._state[p] = np.array([y[-1] if y.size else 0.0, 0.0])
                self._t_last[p] = t[-1] if t.size else 0.0
                continue
            _, sigma = robust_stats(y)
            r = max(sigma ** 2, 1e-8)
            # select process-noise level q by minimizing one-step NLL over a grid
            best = None
            for q in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
                try:
                    state, _, tl, nll = _filter(t, y, q, r)
                except Exception:
                    continue
                if best is None or nll < best[0]:
                    best = (nll, state, tl)
            self._state[p] = best[1]
            self._t_last[p] = best[2]
        return self

    def predict(self, t_seconds):
        t = np.asarray(t_seconds, dtype=float)
        out = {}
        for p in PARAM_NAMES:
            level, slope = self._state[p]
            out[p] = level + slope * (t - self._t_last[p])
        return out
