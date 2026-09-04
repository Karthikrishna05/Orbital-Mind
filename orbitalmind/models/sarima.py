"""SARIMA (seasonal ARIMA) challenger.

Caveat: ARIMA/SARIMA assume *evenly spaced* samples, which this data is NOT. To
apply it we resample each channel onto a uniform grid (linear interpolation), fit
SARIMAX, forecast ahead, and interpolate the fitted+forecast path back to the
requested (arbitrary) timestamps. The resampling is a real approximation -- the
training-fold leaderboard is the fair judge of whether it helps here.
"""
from __future__ import annotations

import warnings

import numpy as np

from .. import PARAM_NAMES
from ..config import DEFAULT_PERIODS_HOURS
from .base import Model

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False


class SARIMAModel(Model):
    def __init__(self, kind: str = "GEO", grid_hours: float = 1.0,
                 order=(1, 0, 1), seasonal=True):
        self.kind = kind
        self.grid_hours = grid_hours
        self.order = order
        period_h = DEFAULT_PERIODS_HOURS.get(kind, DEFAULT_PERIODS_HOURS["GEO"])[0]
        self.s = max(2, int(round(period_h / grid_hours))) if seasonal else 0
        self.seasonal = seasonal
        self.name = f"sarima{order}s{self.s}"
        self._grid_t = {}
        self._path_t = {}
        self._path_v = {}

    def fit(self, series):
        t = np.asarray(series.t, dtype=float)
        step = self.grid_hours * 3600.0
        grid_t = np.arange(t[0], t[-1] + step / 2, step)
        for p in PARAM_NAMES:
            y = np.asarray(series.values[p], dtype=float)
            yg = np.interp(grid_t, t, y)
            self._grid_t[p] = grid_t
            fitted = None
            if _HAVE_SM and grid_t.size >= max(10, 2 * self.s + 5):
                seas = (1, 0, 0, self.s) if (self.seasonal and self.s >= 2) else (0, 0, 0, 0)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = SARIMAX(yg, order=self.order, seasonal_order=seas,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
                    fitted = np.asarray(res.fittedvalues, dtype=float)
                    self._res = getattr(self, "_res", {}); self._res[p] = res
                except Exception:
                    fitted = None
            if fitted is None:
                # fallback: the grid series itself (no model) -> interpolation only
                self._path_t[p] = grid_t
                self._path_v[p] = yg
                self._res = getattr(self, "_res", {}); self._res[p] = None
            else:
                fitted = np.nan_to_num(fitted, nan=float(np.mean(yg)))
                self._path_t[p] = grid_t
                self._path_v[p] = fitted
        return self

    def predict(self, t_seconds):
        t = np.asarray(t_seconds, dtype=float)
        step = self.grid_hours * 3600.0
        out = {}
        for p in PARAM_NAMES:
            grid_t = self._grid_t[p]
            path_t, path_v = self._path_t[p], self._path_v[p]
            res = getattr(self, "_res", {}).get(p)
            tmax = t.max() if t.size else grid_t[-1]
            if res is not None and tmax > grid_t[-1]:
                n_ahead = int(np.ceil((tmax - grid_t[-1]) / step))
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fc = np.asarray(res.forecast(n_ahead), dtype=float)
                    fut_t = grid_t[-1] + step * np.arange(1, n_ahead + 1)
                    path_t = np.concatenate([path_t, fut_t])
                    path_v = np.concatenate([path_v, np.nan_to_num(fc)])
                except Exception:
                    pass
            out[p] = np.interp(t, path_t, path_v)
        return out
