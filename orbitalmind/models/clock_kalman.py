"""Two-state clock Kalman for the satclockerror channel only.

The satellite clock error is well modelled as a phase/frequency process:
  state = [bias, drift];  bias_{k+1} = bias_k + drift_k*dt;  drift_{k+1} = drift_k
with process noise combining white-frequency (q1) and random-walk-frequency (q2)
terms -- the standard two-state clock model, integrated over the irregular dt.
The (q1, q2) pair is chosen by minimizing one-step predictive negative
log-likelihood (a data-driven stand-in for Allan-variance tuning).

x/y/z are NOT clock-like, so they are handled by a robust harmonic base; this
model only replaces the clock channel.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from ..dataio.clean import robust_stats
from .base import Model
from .harmonic import HarmonicModel


def _clock_filter(t, y, q1, q2, r):
    """Two-state clock Kalman; returns final (state, t_last) and one-step NLL."""
    x = np.array([y[0], 0.0])
    P = np.array([[r, 0.0], [0.0, r]])
    H = np.array([1.0, 0.0])
    nll = 0.0
    for i in range(1, len(t)):
        dt = max(t[i] - t[i - 1], 1e-9)
        F = np.array([[1.0, dt], [0.0, 1.0]])
        # white-frequency (q1) + random-walk-frequency (q2) discrete process noise
        Q = np.array([[q1 * dt + q2 * dt**3 / 3.0, q2 * dt**2 / 2.0],
                      [q2 * dt**2 / 2.0, q2 * dt]])
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H + r
        v = y[i] - H @ x
        K = (P @ H) / S
        x = x + K * v
        P = P - np.outer(K, H @ P)
        nll += 0.5 * (np.log(2 * np.pi * S) + v * v / S)
    return x, t[-1], nll


class ClockKalmanModel(Model):
    def __init__(self, kind="GEO"):
        self.kind = kind
        self.name = "clock_kalman"
        self._base = None            # harmonic for x/y/z
        self._state = None
        self._t_last = 0.0

    def fit(self, series):
        self._base = HarmonicModel(kind=self.kind, n_poly=2, harmonics=2,
                                   robust=True).fit(series)
        t = np.asarray(series.t, dtype=float)
        y = np.asarray(series.values["clock"], dtype=float)
        if t.size < 3:
            self._state = np.array([y[-1] if y.size else 0.0, 0.0])
            self._t_last = t[-1] if t.size else 0.0
            return self
        _, sigma = robust_stats(y)
        r = max(sigma**2, 1e-8)
        best = None
        for q1 in (1e-10, 1e-8, 1e-6, 1e-4):
            for q2 in (1e-14, 1e-12, 1e-10, 1e-8):
                try:
                    state, tl, nll = _clock_filter(t, y, q1, q2, r)
                except Exception:
                    continue
                if best is None or nll < best[0]:
                    best = (nll, state, tl)
        self._state, self._t_last = best[1], best[2]
        return self

    def predict(self, t_seconds):
        t = np.asarray(t_seconds, dtype=float)
        out = self._base.predict(t)                     # x/y/z from harmonic base
        b, d = self._state
        out["clock"] = b + d * (t - self._t_last)       # clock from Kalman state
        return out
