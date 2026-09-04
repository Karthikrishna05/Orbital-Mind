"""Load the irregular GNSS error CSVs.

Responsibilities (per the build plan):
  * normalize inconsistent headers (MEO files have ``y_error  (m)`` w/ 2 spaces),
  * parse ``M/D/YYYY H:MM`` timestamps,
  * DROP exact-duplicate rows (the MEO train/test files repeat whole blocks),
  * return per-orbit arrays sorted by time.

The residuals we ultimately score are on the four parameters x, y, z, clock.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import DATASETS
from .. import PARAM_NAMES

_EPOCH = datetime(1970, 1, 1)


def _epoch_seconds(dt: datetime) -> float:
    """Naive-UTC seconds since 1970-01-01 (timezone-independent)."""
    return (dt - _EPOCH).total_seconds()


def _normalize_header(cols):
    """Map raw header cells to canonical names x, y, z, clock, time."""
    out = []
    for c in cols:
        key = c.strip().lower().replace(" ", "")
        if key.startswith("utc") or key == "time":
            out.append("time")
        elif key.startswith("x_error") or key.startswith("xerror"):
            out.append("x")
        elif key.startswith("y_error") or key.startswith("yerror"):
            out.append("y")
        elif key.startswith("z_error") or key.startswith("zerror"):
            out.append("z")
        elif "clock" in key:
            out.append("clock")
        else:
            out.append(key)
    return out


@dataclass
class SeriesData:
    """A single satellite/orbit time series.

    Attributes
    ----------
    name : str
        Logical name (e.g. "GEO", "MEO1").
    kind : str
        Orbit kind, "GEO" or "MEO".
    t : np.ndarray
        Absolute epoch seconds (float), sorted ascending.
    datetimes : np.ndarray[object]
        Corresponding python datetimes (naive UTC).
    values : dict[str, np.ndarray]
        Per-parameter arrays for keys x, y, z, clock.
    n_raw : int
        Row count before de-duplication.
    n_duplicates : int
        Number of exact-duplicate rows removed.
    """

    name: str
    kind: str
    t: np.ndarray
    datetimes: np.ndarray
    values: dict = field(default_factory=dict)
    n_raw: int = 0
    n_duplicates: int = 0

    def __len__(self) -> int:
        return int(self.t.size)

    def matrix(self) -> np.ndarray:
        """(n, 4) array of [x, y, z, clock] columns."""
        return np.column_stack([self.values[p] for p in PARAM_NAMES])

    def t0(self) -> float:
        return float(self.t[0]) if self.t.size else 0.0

    def days_span(self) -> float:
        if self.t.size < 2:
            return 0.0
        return (self.t[-1] - self.t[0]) / 86400.0


def load_csv(path, name: str = "", kind: str = "") -> SeriesData:
    """Parse one CSV into a :class:`SeriesData`, de-duplicating exact rows."""
    path = Path(path)
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    header = _normalize_header(rows[0])
    idx = {col: i for i, col in enumerate(header)}
    for req in ("time", "x", "y", "z", "clock"):
        if req not in idx:
            raise ValueError(f"{path.name}: missing column {req!r} (header={header})")

    seen = set()
    parsed = []  # (t_seconds, dt, x, y, z, clock)
    n_raw = 0
    for r in rows[1:]:
        if len(r) < len(header):
            continue
        raw_key = tuple(cell.strip() for cell in r[: len(header)])
        n_raw += 1
        if raw_key in seen:  # exact-duplicate row -> drop
            continue
        seen.add(raw_key)
        ts = r[idx["time"]].strip()
        dt = _parse_time(ts)
        try:
            vals = [float(r[idx[p]]) for p in PARAM_NAMES]
        except ValueError:
            continue
        parsed.append((_epoch_seconds(dt), dt, *vals))

    n_dup = n_raw - len(parsed)
    parsed.sort(key=lambda row: row[0])

    t = np.array([p[0] for p in parsed], dtype=float)
    dts = np.array([p[1] for p in parsed], dtype=object)
    values = {p: np.array([row[2 + i] for row in parsed], dtype=float)
              for i, p in enumerate(PARAM_NAMES)}
    return SeriesData(
        name=name or path.stem,
        kind=kind or _infer_kind(path.name),
        t=t,
        datetimes=dts,
        values=values,
        n_raw=n_raw,
        n_duplicates=n_dup,
    )


def _parse_time(ts: str) -> datetime:
    """Parse timestamps like ``9/1/2025 6:00`` (single-digit month/day/hour)."""
    for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp: {ts!r}")


def _infer_kind(fname: str) -> str:
    return "GEO" if "GEO" in fname.upper() else "MEO"


def load_dataset(name: str):
    """Load a logical dataset from :data:`config.DATASETS`.

    Returns
    -------
    (train, test) : tuple[SeriesData, SeriesData]
    """
    spec = DATASETS[name]
    train = load_csv(spec["train"], name=name, kind=spec["kind"])
    test = load_csv(spec["test"], name=name + "_test", kind=spec["kind"])
    return train, test


def load_all():
    """Load every logical dataset -> {name: (train, test)}."""
    return {name: load_dataset(name) for name in DATASETS}
