"""Produce a submission: fit on the 7-day training file, predict at supplied
timestamps, and emit predictions + the Priority-1/2/3 report.

Usage:
    python scripts/make_submission.py --train DATA_GEO_Train.csv \
        --kind GEO --model harmonic_robust_p2h2 \
        --timestamps query.csv --out submission_geo.csv

``--timestamps`` is a CSV/text file with one ``M/D/YYYY H:MM`` timestamp per line
(a header line is allowed). If ``--truth`` is given (a full test CSV) the script
also prints the residual normality report.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orbitalmind import PARAM_NAMES
from orbitalmind.dataio.loader import load_csv, _parse_time, _epoch_seconds
from orbitalmind.experiments.registry import build_model
from orbitalmind.evaluation import residuals, evaluate_residuals


def _read_timestamps(path):
    ts = []
    with open(path, "r", newline="") as fh:
        for line in fh:
            s = line.strip().split(",")[0].strip()
            if not s:
                continue
            try:
                dt = _parse_time(s)
            except ValueError:
                continue  # skip header / unparseable
            ts.append(dt)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--kind", default="GEO", choices=["GEO", "MEO"])
    ap.add_argument("--model", default="harmonic_robust_p2h2",
                    help="registered model name, or 'auto' for the per-channel selector")
    ap.add_argument("--timestamps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", default=None,
                    help="optional test CSV to score residual normality")
    ap.add_argument("--stat", default="swtest")
    args = ap.parse_args()

    train = load_csv(args.train, name="train", kind=args.kind)
    model_name = "per_channel_best" if args.model == "auto" else args.model
    model = build_model(model_name, args.kind).fit(train)
    print(f"Model: {model_name}", end="")
    if hasattr(model, "chosen_summary"):
        print(f"  (per-channel: {model.chosen_summary()})", end="")
    print()

    dts = _read_timestamps(args.timestamps)
    t_query = np.array([_epoch_seconds(d) for d in dts], dtype=float)
    pred = model.predict(t_query)

    # Priority-3-style diagnostic: per-timestamp spike-risk (never alters the forecast)
    from orbitalmind.evaluation.spike_risk import spike_risk
    risk, risk_summary = spike_risk(t_query, train)
    print(f"Spike-risk: {risk_summary['high_risk']}/{risk_summary['n']} query points "
          f"flagged high-risk (series burst-outlier rate {risk_summary['base_rate']})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utc_time", "x_error (m)", "y_error (m)",
                    "z_error (m)", "satclockerror (m)", "spike_risk"])
        for i, d in enumerate(dts):
            w.writerow([d.strftime("%m/%d/%Y %H:%M")] +
                       [f"{pred[p][i]:.9g}" for p in PARAM_NAMES] +
                       [f"{risk[i]:.3f}"])
    print(f"Wrote {len(dts)} predictions -> {out}")

    if args.truth:
        truth = load_csv(args.truth, name="truth", kind=args.kind)
        # align by exact timestamp
        idx = {t: i for i, t in enumerate(truth.t)}
        keep = [i for i, t in enumerate(t_query) if t in idx]
        if keep:
            actual = {p: np.array([truth.values[p][idx[t_query[i]]] for i in keep])
                      for p in PARAM_NAMES}
            pk = {p: pred[p][keep] for p in PARAM_NAMES}
            rep = evaluate_residuals(residuals(pk, actual), stat=args.stat,
                                     with_ci=True)
            print(f"\nResidual normality ({args.stat}) on {len(keep)} matched points:")
            print(f"  W_avg={rep.W_avg:.4f}  p_avg={rep.p_avg:.4f}  H_avg={rep.H_avg:.2f}")
            print(f"  mean|.|={rep.mean_abs_avg:.4f}  std={rep.std_avg:.4f}")
            for p in PARAM_NAMES:
                pr = rep.per_param[p]
                print(f"    {p:6} W={pr.W:.4f} p={pr.p:.4f} H={pr.H} "
                      f"CI=[{pr.ci_lo:.4f},{pr.ci_hi:.4f}]")


if __name__ == "__main__":
    main()
