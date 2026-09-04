"""Robust outlier utilities.

Important: at final evaluation we CANNOT drop evaluator-supplied query points, so
outlier handling lives inside the *models* (robust fitting), not in the scored
residual. These helpers flag outliers for diagnostics and for models that want to
down-weight training outliers.
"""
from __future__ import annotations

import numpy as np


def robust_stats(x: np.ndarray):
    """Return (median, MAD-based sigma) for a 1-D array.

    sigma = 1.4826 * MAD is a consistent estimator of std for Gaussian data.
    """
    x = np.asarray(x, dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad
    return med, sigma


def mad_outlier_mask(x: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """Boolean mask (True = outlier) using the robust modified z-score.

    Points with |0.6745 * (x - median) / MAD| > thresh are flagged. Falls back to
    the standard z-score when MAD == 0 (constant-ish data).
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        std = np.std(x)
        if std <= 0:
            return np.zeros_like(x, dtype=bool)
        z = np.abs(x - med) / std
        return z > thresh
    modified_z = 0.6745 * np.abs(x - med) / mad
    return modified_z > thresh
