"""Change-point-aware clock model.

Detects upload-reset change-points in the clock series and fits the two-state
clock Kalman on the *final segment only*, so a pre-upload trend does not bias the
current drift. x/y/z use a robust harmonic base (with the same segment restriction
optionally applied). This is the "refine the Kalman clock component" step, done
honestly: it improves the clean part of the clock, and cannot predict the timing of
the next (unobservable) upload.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from ..dataio.clean import robust_stats
from ..features.changepoint import last_segment_mask
from .base import Model
from .harmonic import HarmonicModel
from .clock_kalman import _clock_filter


class SegmentedClockModel(Model):
    def __init__(self, kind="GEO", k: float = 6.0, min_seg: int = 4):
        self.kind = kind
        self.k = k
        self.min_seg = min_seg
        self.name = "segmented_clock"
        self._base = None
        self._state = None
        self._t_last = 0.0
        self._n_cps = 0

    def fit(self, series):
        self._base = HarmonicModel(kind=self.kind, n_poly=2, harmonics=2,
                                   robust=True).fit(series)
        t = np.asarray(series.t, dtype=float)
        y = np.asarray(series.values["clock"], dtype=float)
        if t.size < 3:
            self._state = np.array([y[-1] if y.size else 0.0, 0.0])
            self._t_last = t[-1] if t.size else 0.0
            return self

        mask, cps = last_segment_mask(y, k=self.k, min_seg=self.min_seg)
        self._n_cps = len(cps)
        ts, ys = t[mask], y[mask]

        _, sigma = robust_stats(ys)
        r = max(sigma ** 2, 1e-8)
        best = None
        for q1 in (1e-10, 1e-8, 1e-6, 1e-4):
            for q2 in (1e-14, 1e-12, 1e-10, 1e-8):
                try:
                    state, tl, nll = _clock_filter(ts, ys, q1, q2, r)
                except Exception:
                    continue
                if best is None or nll < best[0]:
                    best = (nll, state, tl)
        self._state, self._t_last = best[1], best[2]
        return self

    def predict(self, t_seconds):
        t = np.asarray(t_seconds, dtype=float)
        out = self._base.predict(t)
        b, d = self._state
        out["clock"] = b + d * (t - self._t_last)
        return out
