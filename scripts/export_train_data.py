"""Export the REAL 7-day training observations to the frontend.

The frontend's Train Data page must describe the 7-day TRAINING window (what
the models are fit on), not the day-8 test period. This script mirrors the
existing ``evaluate_day8.py -> frontend/js/dynamic_data.js`` pattern: it reads
the actual training CSVs, derives per-point observations over the real 7-day
window, and dumps them to ``frontend/js/dynamic_train_data.js`` for the page
to plot. Training data is fixed, so this is a run-once export (re-run only if
the training files change).

GEO uses the GEO training file. MEO combines both MEO satellite training files
(MEO1 + MEO2) into a single time-ordered "MEO" series -- the frontend exposes a
single MEO category, and the MEO1/MEO2 split is only how the raw files are
named.

Usage:
    python scripts/export_train_data.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from orbitalmind import PARAM_NAMES
from orbitalmind.dataio.loader import load_dataset
from orbitalmind.dataio.combine import combine


ORBIT_TYPE = {
    "GEO": "Geostationary Orbit (GEO — ~35,786 km)",
    "MEO": "Medium Earth Orbit (MEO — ~20,200 km)",
}


def _series_for(orbit: str):
    """Return the training SeriesData for a frontend orbit category."""
    if orbit == "GEO":
        train, _ = load_dataset("GEO")
        n_sv = 1
        sv_note = "1 GEO SV"
    elif orbit == "MEO":
        t1, _ = load_dataset("MEO1")
        t2, _ = load_dataset("MEO2")
        train = combine(t1, t2)
        n_sv = 2
        sv_note = "2 MEO SVs"
    else:
        raise ValueError(orbit)
    return train, sv_note


def _export_one(orbit: str) -> dict:
    train, sv_note = _series_for(orbit)

    t = np.asarray(train.t, dtype=float)
    order = np.argsort(t)
    t = t[order]
    x = np.asarray(train.values["x"], dtype=float)[order]
    y = np.asarray(train.values["y"], dtype=float)[order]
    z = np.asarray(train.values["z"], dtype=float)[order]
    clk = np.asarray(train.values["clock"], dtype=float)[order]

    # 3D position-error magnitude per observation.
    eph = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Relative-day timeline labels ("D1 00:00" ... across the 7-day window).
    t0 = t[0]
    span_days = (t[-1] - t0) / 86400.0
    labels = []
    for ti in t:
        secs = ti - t0
        day = int(secs // 86400) + 1
        rem = secs - (day - 1) * 86400
        hh = int(rem // 3600)
        mm = int((rem % 3600) // 60)
        labels.append(f"D{day} {hh:02d}:{mm:02d}")

    rms3d = float(np.sqrt(np.mean(eph ** 2)))

    return {
        "orbitType": ORBIT_TYPE[orbit],
        "timestamps": labels,
        "ephemerisError": [round(float(v), 4) for v in eph],
        "clockBiasError": [round(float(v), 4) for v in clk],
        "stats": {
            "meanEphemeris": f"{float(np.mean(eph)):.2f} m (3D position error)",
            "maxEphemeris": f"{float(np.max(eph)):.2f} m peak",
            "meanClockBias": f"{float(np.mean(np.abs(clk))):.2f} m (|clock| mean)",
            "rmsError": f"{rms3d:.2f} m composite 3D RMS",
            "satellitesTracked": f"{sv_note} | {len(t)} train pts | "
                                 f"{span_days:.1f}-day training window",
        },
    }


def main():
    payload = {orbit: _export_one(orbit) for orbit in ("GEO", "MEO")}

    out = Path(__file__).resolve().parent.parent / "frontend" / "js" / "dynamic_train_data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "window.DYNAMIC_TRAIN_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    for orbit in ("GEO", "MEO"):
        s = payload[orbit]["stats"]
        print(f"[{orbit}] {s['satellitesTracked']}  |  {s['rmsError']}")
    print(f"Frontend training data synced: {out}")


if __name__ == "__main__":
    main()
