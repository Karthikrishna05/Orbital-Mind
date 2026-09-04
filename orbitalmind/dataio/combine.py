"""Helpers to combine / subset SeriesData for fine-tuning on train + day-8."""
from __future__ import annotations

import numpy as np

from .. import PARAM_NAMES
from .loader import SeriesData


def subset(series: SeriesData, idx) -> SeriesData:
    idx = np.asarray(idx, dtype=int)
    return SeriesData(
        name=series.name, kind=series.kind,
        t=np.asarray(series.t)[idx],
        datetimes=np.asarray(series.datetimes, dtype=object)[idx],
        values={p: np.asarray(series.values[p])[idx] for p in PARAM_NAMES},
    )


def combine(a: SeriesData, b: SeriesData) -> SeriesData:
    """Concatenate two series and sort by time (for fine-tuning on train+day8)."""
    t = np.concatenate([np.asarray(a.t), np.asarray(b.t)])
    order = np.argsort(t)
    dts = np.concatenate([np.asarray(a.datetimes, dtype=object),
                          np.asarray(b.datetimes, dtype=object)])[order]
    vals = {p: np.concatenate([np.asarray(a.values[p]), np.asarray(b.values[p])])[order]
            for p in PARAM_NAMES}
    return SeriesData(name=f"{a.name}+ft", kind=a.kind, t=t[order], datetimes=dts,
                      values=vals, n_raw=t.size, n_duplicates=0)
