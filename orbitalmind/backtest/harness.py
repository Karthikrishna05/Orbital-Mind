"""Run a model through a query-based backtest and score its residuals.

The score is the evaluator-faithful ``swtest`` statistic on the residual
(predicted - actual), per parameter, equal-weight averaged -- exactly the
competition's Priority 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import PARAM_NAMES
from ..evaluation import residuals, evaluate_residuals, ResidualReport
from ..models.harmonic import HarmonicModel


@dataclass
class BacktestResult:
    dataset: str
    model: str
    report: ResidualReport
    pred: dict
    actual: dict

    @property
    def W_avg(self):
        return self.report.W_avg


def run_model(model, train_series, query_t, actual: dict, dataset: str = "",
              stat: str = "swtest", with_ci: bool = False) -> BacktestResult:
    """Fit ``model`` on ``train_series``, predict at ``query_t``, score residuals."""
    model.fit(train_series)
    pred = model.predict(query_t)
    resid = residuals(pred, actual)
    report = evaluate_residuals(resid, stat=stat, with_ci=with_ci)
    return BacktestResult(dataset=dataset, model=getattr(model, "name", repr(model)),
                          report=report, pred=pred, actual=actual)


def rolling_validation(model_factory, series, kind: str = "GEO",
                       stat: str = "swtest", n_folds: int = 3) -> dict:
    """Leak-free internal validation averaged over rolling day-folds.

    model_factory: callable(kind) -> fresh Model. Returns averaged metrics
    {W, std, mean_abs, H, n_folds}; the day-8 test file is never used.
    """
    from .splits import rolling_day_folds
    Ws, stds, means, Hs = [], [], [], []
    for fit_s, val_t, val_actual in rolling_day_folds(series, n_folds=n_folds):
        try:
            res = run_model(model_factory(kind), fit_s, val_t, val_actual, stat=stat)
        except Exception:
            continue
        rep = res.report
        if np.isfinite(rep.W_avg):
            Ws.append(rep.W_avg); stds.append(rep.std_avg)
            means.append(rep.mean_abs_avg); Hs.append(rep.H_avg)
    if not Ws:
        return {"W": float("nan"), "std": float("nan"), "mean_abs": float("nan"),
                "H": float("nan"), "n_folds": 0}
    return {"W": float(np.mean(Ws)), "std": float(np.mean(stds)),
            "mean_abs": float(np.mean(means)), "H": float(np.mean(Hs)),
            "n_folds": len(Ws)}


def truth_diagnostics(actual: dict, query_t, kind: str = "GEO",
                      stat: str = "swtest") -> dict:
    """Context diagnostics on the day-8 ground truth itself.

    NOTE: there is no clean "oracle ceiling" for this metric. W rewards residual
    *shape*, not fit quality, so (a) an over-flexible fit drives residuals to zero
    (degenerate W) and (b) adding Gaussian spread can *raise* W by masking genuine
    outliers. These diagnostics instead characterize how non-Gaussian the truth is:

      W_raw       : swtest W of the raw actual day-8 values (equal-weight avg).
      W_detrended : swtest W of the residual after a robust harmonic in-sample fit
                    to the truth -- i.e. the truth's own innovations. When the truth
                    has heavy outliers this is LOW, showing the residual normality
                    any model must fight against.
    """
    from ..dataio.loader import SeriesData
    raw = {p: np.asarray(actual[p], dtype=float) for p in PARAM_NAMES}
    zero = {p: np.zeros_like(raw[p]) for p in PARAM_NAMES}
    W_raw = evaluate_residuals({p: raw[p] - zero[p] for p in PARAM_NAMES},
                               stat=stat).W_avg

    s = SeriesData(name="truth", kind=kind,
                   t=np.asarray(query_t, dtype=float),
                   datetimes=np.arange(len(query_t)),
                   values=raw)
    model = HarmonicModel(kind=kind, n_poly=2, harmonics=1, robust=True)
    try:
        model.fit(s)
        pred = model.predict(s.t)
        W_det = evaluate_residuals(residuals(pred, actual), stat=stat).W_avg
    except Exception:
        W_det = float("nan")
    return {"W_raw": W_raw, "W_detrended": W_det}
