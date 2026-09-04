"""Per-timestamp spike-risk flag.

Upload/maneuver spikes are exogenous ground-segment events, unpredictable in
timing from history — so rather than forecast them, we report how *likely* a query
timestamp is to be spike-contaminated. Risk rises when (a) the query sits in a
dense/burst sampling cluster (where spikes historically concentrate), (b) it falls
in the equinox eclipse season, and (c) the training series showed a high burst
outlier rate. Purely a diagnostic; it never alters the point forecast.
"""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from ..dataio.clean import mad_outlier_mask
from ..features.astronomical import astro_features
from ..models.regime import _local_gap_minutes


def training_burst_outlier_rate(series, dense_gap_min: float = 45.0):
    """Fraction of dense-regime training points that are outliers, averaged over
    the four channels — a scalar measure of how spike-prone this series is."""
    gaps = _local_gap_minutes(series.t)
    dense = gaps <= dense_gap_min
    if dense.sum() < 3:
        return 0.0
    rates = []
    for p in PARAM_NAMES:
        v = np.asarray(series.values[p])[dense]
        if v.size >= 3:
            rates.append(float(mad_outlier_mask(v, 3.5).mean()))
    return float(np.mean(rates)) if rates else 0.0


def spike_risk(t_query, train_series, dense_gap_min: float = 45.0):
    """Return a per-query risk score in [0, 1] and a short summary dict."""
    t = np.asarray(t_query, dtype=float)
    n = t.size
    if n == 0:
        return np.array([]), {"n": 0, "high_risk": 0, "base_rate": 0.0}

    dense_q = (_local_gap_minutes(t) <= dense_gap_min).astype(float)
    astro, names = astro_features(t)
    eclipse = astro[:, names.index("eclipse_midnight")]
    eclipse_n = (eclipse - eclipse.min()) / (np.ptp(eclipse) + 1e-12)
    base_rate = training_burst_outlier_rate(train_series, dense_gap_min)

    # weighted blend, scaled by how spike-prone the series is historically
    risk = np.clip((0.6 * dense_q + 0.4 * eclipse_n) * (0.5 + base_rate), 0.0, 1.0)
    high = int((risk >= 0.5).sum())
    return risk, {"n": n, "high_risk": high, "base_rate": round(base_rate, 3)}
