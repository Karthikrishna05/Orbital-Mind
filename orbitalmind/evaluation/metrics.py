"""Assemble residuals and produce the exact competition report card.

Residual r = predicted - actual, per parameter (x, y, z, clock), equal weight.
  Priority 1: Shapiro-Wilk W (higher better), p-value, H, averaged over params.
  Priority 2: mean & std of residual (tiebreak).
Both a per-parameter average (primary, per the Note's wording) and a pooled
standardized variant (ambiguity hedge) are reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import PARAM_NAMES
from .shapiro import STAT_FUNCS
from .confidence import bootstrap_ci_W


@dataclass
class ParamResult:
    param: str
    n_raw: int
    n: int
    W: float
    p: float
    H: int
    mean: float
    std: float
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")


@dataclass
class ResidualReport:
    per_param: dict = field(default_factory=dict)   # param -> ParamResult
    W_avg: float = float("nan")
    p_avg: float = float("nan")
    H_avg: float = float("nan")
    mean_abs_avg: float = float("nan")
    std_avg: float = float("nan")
    W_pooled: float = float("nan")
    p_pooled: float = float("nan")
    H_pooled: int = 0
    n_points: int = 0

    def summary_row(self):
        """Flat dict for leaderboard tables."""
        return {
            "W_avg": self.W_avg,
            "p_avg": self.p_avg,
            "H_avg": self.H_avg,
            "mean_abs_avg": self.mean_abs_avg,
            "std_avg": self.std_avg,
            "W_pooled": self.W_pooled,
            "n": self.n_points,
        }


def residuals(pred: dict, actual: dict) -> dict:
    """Per-parameter residual arrays (predicted - actual)."""
    out = {}
    for p in PARAM_NAMES:
        out[p] = np.asarray(pred[p], dtype=float) - np.asarray(actual[p], dtype=float)
    return out


def mad_filter(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Robust outlier rejection using Median Absolute Deviation.
    
    Removes points that are more than `threshold` MADs from the median.
    """
    if data.size < 3:
        return data
    med = np.median(data)
    d = np.abs(data - med)
    mdev = np.median(d)
    if mdev == 0.0:
        return data
    s = d / mdev
    return data[s < threshold]


def evaluate_residuals(resid: dict, alpha: float = 0.05, stat: str = "swtest",
                       with_ci: bool = False, ci_B: int = 1000,
                       reject_outliers: bool = False, mad_threshold: float = 3.0) -> ResidualReport:
    """Compute the Priority-1/2 report from per-parameter residual arrays.

    Parameters
    ----------
    stat : str
        Which normality statistic to use: "swtest" (default, evaluator-faithful),
        "shapiro_wilk" (Note's literal wording), or "shapiro_francia".
    reject_outliers : bool
        If True, applies a MAD filter to remove severe hardware outliers before scoring.
    """
    stat_fn = STAT_FUNCS[stat]
    per_param = {}
    Ws, ps, Hs, means_abs, stds = [], [], [], [], []
    for p in PARAM_NAMES:
        r_raw = np.asarray(resid[p], dtype=float)
        r_raw = r_raw[np.isfinite(r_raw)]
        n_raw = r_raw.size
        
        r = mad_filter(r_raw, threshold=mad_threshold) if reject_outliers else r_raw
        n = r.size
        if n >= 3 and (r.max() - r.min()) > 1e-19:
            sw = stat_fn(r, alpha=alpha)
            W, pv, H = sw.W, sw.p, sw.H
        else:
            W, pv, H = float("nan"), float("nan"), 0
        ci_lo = ci_hi = float("nan")
        if with_ci and n >= 4:
            ci = bootstrap_ci_W(r, B=ci_B, alpha=alpha)
            ci_lo, ci_hi = ci["lo"], ci["hi"]
        pr = ParamResult(param=p, n_raw=n_raw, n=n, W=W, p=pv, H=H,
                         mean=float(np.mean(r)) if n else float("nan"),
                         std=float(np.std(r, ddof=1)) if n > 1 else float("nan"),
                         ci_lo=ci_lo, ci_hi=ci_hi)
        per_param[p] = pr
        if np.isfinite(W):
            Ws.append(W); ps.append(pv); Hs.append(H)
        if n:
            means_abs.append(abs(pr.mean))
            if n > 1:
                stds.append(pr.std)

    rep = ResidualReport(per_param=per_param)
    rep.W_avg = float(np.mean(Ws)) if Ws else float("nan")
    rep.p_avg = float(np.mean(ps)) if ps else float("nan")
    rep.H_avg = float(np.mean(Hs)) if Hs else float("nan")
    rep.mean_abs_avg = float(np.mean(means_abs)) if means_abs else float("nan")
    rep.std_avg = float(np.mean(stds)) if stds else float("nan")

    # Pooled standardized residuals (each param z-scored then concatenated).
    pooled = []
    for p in PARAM_NAMES:
        r_raw = np.asarray(resid[p], dtype=float)
        r_raw = r_raw[np.isfinite(r_raw)]
        r = mad_filter(r_raw, threshold=mad_threshold) if reject_outliers else r_raw
        if r.size > 1:
            s = np.std(r, ddof=1)
            if s > 0:
                pooled.append((r - np.mean(r)) / s)
    if pooled:
        allr = np.concatenate(pooled)
        rep.n_points = int(allr.size)
        if allr.size >= 3 and (allr.max() - allr.min()) > 1e-19:
            sw = stat_fn(allr, alpha=alpha)
            rep.W_pooled, rep.p_pooled, rep.H_pooled = sw.W, sw.p, sw.H
    return rep
