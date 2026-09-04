"""Run every registered model over every dataset -> leaderboard rows."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dataio import load_all
from ..config import DATASETS
from ..backtest import day8_split, run_model, truth_diagnostics, rolling_validation
from .registry import MODEL_REGISTRY, build_model


@dataclass
class LeaderboardRow:
    dataset: str
    model: str
    W_avg: float
    p_avg: float
    H_avg: float
    mean_abs_avg: float
    std_avg: float
    W_pooled: float
    n: int
    W_holdout: float = float("nan")     # internal rolling-validation W (day-8 excluded)
    std_holdout: float = float("nan")   # internal-validation residual std
    H_holdout: float = float("nan")     # internal-validation reject-rate
    mean_abs_holdout: float = float("nan")  # internal-validation |mean| (bias guard)
    n_folds: int = 0
    per_param_W: dict = field(default_factory=dict)


def run_leaderboard(datasets=None, model_names=None, stat: str = "swtest",
                    include_test: bool = True):
    """Return (rows, diagnostics).

    rows: list[LeaderboardRow] sorted by (dataset, -W_avg).
    diagnostics: {dataset: {"W_raw", "W_detrended"}} truth-context (see
        :func:`~orbitalmind.backtest.truth_diagnostics`); not a hard ceiling.

    include_test=False computes ONLY the leak-free rolling-fold validation and
    never touches the day-8 test file (day-8 fields are left as NaN).
    """
    data = load_all()
    datasets = datasets or list(DATASETS.keys())
    model_names = model_names or list(MODEL_REGISTRY.keys())

    rows = []
    diagnostics = {}
    for ds in datasets:
        train, test = data[ds]
        kind = DATASETS[ds]["kind"]

        if include_test:
            _, q_t, actual = day8_split(train, test)
            try:
                diagnostics[ds] = truth_diagnostics(actual, q_t, kind=kind, stat=stat)
            except Exception:
                diagnostics[ds] = {"W_raw": float("nan"), "W_detrended": float("nan")}

        for name in model_names:
            if include_test:
                res = run_model(build_model(name, kind), train, q_t, actual,
                                dataset=ds, stat=stat, reject_outliers=True, mad_threshold=3.0)
                rep = res.report
                test_fields = dict(
                    W_avg=rep.W_avg, p_avg=rep.p_avg, H_avg=rep.H_avg,
                    mean_abs_avg=rep.mean_abs_avg, std_avg=rep.std_avg,
                    W_pooled=rep.W_pooled, n=rep.n_points,
                    per_param_W={k: v.W for k, v in rep.per_param.items()})
            else:
                nan = float("nan")
                test_fields = dict(W_avg=nan, p_avg=nan, H_avg=nan,
                                   mean_abs_avg=nan, std_avg=nan, W_pooled=nan,
                                   n=0, per_param_W={})

            # leak-free internal validation (rolling folds within train only)
            val = rolling_validation(lambda k, _n=name: build_model(_n, k),
                                     train, kind=kind, stat=stat, n_folds=3)

            rows.append(LeaderboardRow(
                dataset=ds, model=name,
                W_holdout=val["W"], std_holdout=val["std"], H_holdout=val["H"],
                mean_abs_holdout=val["mean_abs"], n_folds=val["n_folds"],
                **test_fields,
            ))

    def sort_key(r):
        w = r.W_avg if np.isfinite(r.W_avg) else (
            r.W_holdout if np.isfinite(r.W_holdout) else -1e9)
        return (r.dataset, -w)

    rows.sort(key=sort_key)
    return rows, diagnostics
