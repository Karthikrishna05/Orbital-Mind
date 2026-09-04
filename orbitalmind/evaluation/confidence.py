"""Confidence interval for the Shapiro-Wilk W statistic.

The Note asks for a confidence interval on the SW score but does not specify the
method. We use a nonparametric bootstrap (percentile) CI on W: resample the
residuals with replacement, recompute W, and report the [alpha/2, 1-alpha/2]
percentiles. This is documented as our chosen interpretation.
"""
from __future__ import annotations

import numpy as np

from .shapiro import shapiro_wilk


def bootstrap_ci_W(x, B: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the SW W statistic.

    Returns
    -------
    dict with keys: W (point estimate), lo, hi, B, alpha.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    point = shapiro_wilk(x).W if n >= 3 else float("nan")
    if n < 4:
        return {"W": point, "lo": float("nan"), "hi": float("nan"),
                "B": 0, "alpha": alpha}
    rng = np.random.default_rng(seed)
    ws = np.empty(B)
    k = 0
    for _ in range(B):
        sample = x[rng.integers(0, n, size=n)]
        if sample.max() - sample.min() < 1e-19:
            continue
        try:
            ws[k] = shapiro_wilk(sample).W
            k += 1
        except ValueError:
            continue
    ws = ws[:k]
    if ws.size == 0:
        return {"W": point, "lo": float("nan"), "hi": float("nan"),
                "B": 0, "alpha": alpha}
    lo = float(np.percentile(ws, 100 * alpha / 2))
    hi = float(np.percentile(ws, 100 * (1 - alpha / 2)))
    return {"W": point, "lo": lo, "hi": hi, "B": int(ws.size), "alpha": alpha}
