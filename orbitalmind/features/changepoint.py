"""Change-point / upload-reset detection.

The satellite clock (and, to a lesser degree, the ephemeris) error jumps
discontinuously whenever the ground segment uploads a fresh correction. Within a
segment the process is smooth; across an upload it resets. Detecting those
boundaries lets a model fit only the *current* segment (so pre-reset data does not
bias the drift) and lets us flag that a reset could occur.

Lightweight and dependency-free: a robust jump detector on first differences.
"""
from __future__ import annotations

import numpy as np


def detect_changepoints(y, k: float = 6.0, min_seg: int = 4):
    """Return indices i (each the first point of a new segment) where |y[i]-y[i-1]|
    is a robust outlier among all step sizes.

    k        : threshold in robust-sigma units (median + k*MAD of |diff|).
    min_seg  : minimum spacing between accepted change-points.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 2 * min_seg + 1:
        return []
    d = np.abs(np.diff(y))
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    sigma = 1.4826 * mad if mad > 0 else np.std(d)
    if sigma <= 0:
        return []
    thresh = med + k * sigma
    raw = [i + 1 for i in range(d.size) if d[i] > thresh]
    # enforce minimum segment spacing (keep the strongest jump in a cluster)
    cps = []
    for i in raw:
        if cps and i - cps[-1] < min_seg:
            if d[i - 1] > d[cps[-1] - 1]:
                cps[-1] = i
        else:
            cps.append(i)
    return cps


def last_segment_mask(y, k: float = 6.0, min_seg: int = 4):
    """Boolean mask selecting only the final segment (after the last change-point)."""
    y = np.asarray(y, dtype=float)
    cps = detect_changepoints(y, k=k, min_seg=min_seg)
    mask = np.zeros(y.size, dtype=bool)
    start = cps[-1] if cps else 0
    # guard: if the final segment is too short to fit, fall back to the whole series
    if y.size - start < min_seg:
        start = 0
    mask[start:] = True
    return mask, cps
