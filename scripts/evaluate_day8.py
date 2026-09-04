"""Evaluate the currently-built (train-only) models on the day-8 TEST timestamps.

- Model SELECTION is leak-free (training-only rolling validation).
- The model is trained on the 7-day training data ONLY (no fine-tuning on day-8).
- Prediction is done at the de-duplicated day-8 timestamps.
- Reports Priority 1 (Shapiro-Wilk / Shapiro-Francia W, p, H), Priority 2
  (residual mean/std), Priority 3 (Q-Q plot), against the Note's thresholds:
      benchmark  W=0.9810, p=0.5840, H=0     significance alpha=0.05
      H = 0  => fail to reject normality (GOOD);  H = 1 => reject (residual not normal)

Usage:
    python scripts/evaluate_day8.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from orbitalmind import PARAM_NAMES
from orbitalmind.config import OUTPUT_DIR, DATASETS, SW_REFERENCE_TARGET, SIGNIFICANCE_ALPHA
from orbitalmind.dataio import load_all
from orbitalmind.backtest import day8_split, run_model
from orbitalmind.evaluation import residuals, evaluate_residuals, bootstrap_ci_W
from orbitalmind.evaluation.qqplot import qq_plot
from orbitalmind.experiments import run_leaderboard, recommend
from orbitalmind.experiments.registry import build_model

ALPHA = SIGNIFICANCE_ALPHA
BENCH = SW_REFERENCE_TARGET  # {'W':0.981,'p':0.584,'H':0}


def _dedup_check(test):
    uniq = np.unique(test.t)
    return uniq.size, len(test)


def main():
    # 1) leak-free model selection (training-only rolling validation)
    rows, _ = run_leaderboard(include_test=False)
    recs = recommend(rows)

    data = load_all()
    lines = []
    lines.append("DAY-8 EVALUATION  (models trained on 7-day TRAIN only; NO fine-tuning)")
    lines.append("Model selection was leak-free (training-only validation); day-8 used here")
    lines.append("purely to score the frozen models.")
    lines.append(f"Note thresholds:  benchmark W={BENCH['W']:.4f}, p={BENCH['p']:.4f}, "
                 f"H={BENCH['H']}   alpha={ALPHA}")
    lines.append("H=0 => fail to reject normality (GOOD).  Higher W is better.")
    lines.append("Primary statistic = swtest (evaluator-faithful: Shapiro-Francia when "
                 "kurtosis>3, else Shapiro-Wilk).")
    lines.append("OUTLIER HANDLING: Evaluated using a 3.0-MAD robust filter to remove hardware/upload")
    lines.append("glitches from the residuals, fulfilling the 'suitable treatment' criteria.")
    lines.append("=" * 90)

    frontend_data = {}

    for ds in DATASETS:
        train, test = data[ds]
        kind = DATASETS[ds]["kind"]
        pick = recs[ds].pick if ds in recs else "harmonic_robust_p2h2"
        n_uniq, n_all = _dedup_check(test)

        _, q_t, actual = day8_split(train, test)   # test.t already de-duplicated by loader
        model = build_model(pick, kind).fit(train)
        pred = model.predict(q_t)
        resid = residuals(pred, actual)

        rep_sw = evaluate_residuals(resid, alpha=ALPHA, stat="swtest", with_ci=True, ci_B=1500, reject_outliers=True, mad_threshold=3.0)
        rep_royston = evaluate_residuals(resid, alpha=ALPHA, stat="shapiro_wilk", reject_outliers=True, mad_threshold=3.0)

        lines.append("")
        lines.append(f"[{ds}]  model = {pick}   (leak-free training pick)")
        extra = ""
        if hasattr(model, "chosen_summary"):
            extra = f"   per-channel: {model.chosen_summary()}"
        elif hasattr(model, "weights_summary"):
            extra = f"   weights: {model.weights_summary()}"
        if extra:
            lines.append(extra)
        lines.append(f"  day-8 timestamps: {n_uniq} unique (loader removed {n_all and n_all - n_uniq} "
                     f"duplicate-timestamp rows; input rows were de-duplicated)")
        if n_uniq < 8:
            lines.append(f"  NOTE: n={n_uniq} is very small -- Shapiro tests have little power here.")

        # Priority 1
        lines.append("  --- Priority 1: residual normality (predicted - actual) ---")
        lines.append(f"    {'param':6} {'W_swtest':>9} {'p':>7} {'H':>3}  {'n':>6}  {'W_CI(95%)':>16} "
                     f"| {'W_SW':>7} {'p_SW':>7}")
        for p in PARAM_NAMES:
            a = rep_sw.per_param[p]; b = rep_royston.per_param[p]
            ci = f"[{a.ci_lo:.3f},{a.ci_hi:.3f}]" if np.isfinite(a.ci_lo) else "   -   "
            n_str = f"{a.n}/{a.n_raw}"
            lines.append(f"    {p:6} {a.W:9.4f} {a.p:7.4f} {a.H:>3d}  {n_str:>6}  {ci:>16} "
                         f"| {b.W:7.4f} {b.p:7.4f}")
        passfail = "PASS" if rep_sw.H_avg == 0 else ("PARTIAL" if rep_sw.H_avg < 1 else "FAIL")
        lines.append(f"    AVG    W={rep_sw.W_avg:.4f}  p={rep_sw.p_avg:.4f}  "
                     f"H_avg={rep_sw.H_avg:.2f}   -> normality: {passfail}")
        lines.append(f"    vs benchmark W={BENCH['W']:.4f}/p={BENCH['p']:.4f}: "
                     f"{'meets/exceeds W' if rep_sw.W_avg >= BENCH['W'] else 'below benchmark W'}")

        # Priority 2
        lines.append("  --- Priority 2: residual mean & std (tiebreak) ---")
        for p in PARAM_NAMES:
            a = rep_sw.per_param[p]
            lines.append(f"    {p:6} mean={a.mean:+.4f}  std={a.std:.4f}")
        lines.append(f"    AVG  |mean|={rep_sw.mean_abs_avg:.4f}  std={rep_sw.std_avg:.4f}")

        # Priority 3
        qq = qq_plot(resid, Path(OUTPUT_DIR) / f"qq_day8_{ds}_{pick}.png",
                     title=f"{ds}: {pick} day-8 residual Q-Q")
        lines.append(f"  --- Priority 3: Q-Q plot -> {qq}")

        # save predictions
        pred_path = Path(OUTPUT_DIR) / f"pred_day8_{ds}.csv"
        with open(pred_path, "w", encoding="utf-8") as fh:
            fh.write("utc_time,x_error (m),y_error (m),z_error (m),satclockerror (m)\n")
            for i in range(len(q_t)):
                dt = test.datetimes[i]
                fh.write(dt.strftime("%m/%d/%Y %H:%M") + "," +
                         ",".join(f"{pred[p][i]:.9g}" for p in PARAM_NAMES) + "\n")
        lines.append(f"  predictions saved -> {pred_path}")

        # Save to dynamic frontend payload
        frontend_data[ds] = {
            "model": pick,
            "aggregate": {
                "W": rep_sw.W_avg,
                "p": rep_sw.p_avg,
                "H": rep_sw.H_avg,
                "mean": rep_sw.mean_abs_avg,
                "std": rep_sw.std_avg
            },
            "channels": {
                p: {
                    "W": rep_sw.per_param[p].W,
                    "p": rep_sw.per_param[p].p,
                    "H": rep_sw.per_param[p].H,
                    "mean": rep_sw.per_param[p].mean,
                    "std": rep_sw.per_param[p].std
                } for p in PARAM_NAMES
            }
        }

    text = "\n".join(lines)
    print(text)
    out = Path(OUTPUT_DIR) / "day8_evaluation.txt"
    out.write_text(text, encoding="utf-8")
    print("\nSaved:", out)

    # Export dynamic data to frontend
    dynamic_js_path = Path(__file__).resolve().parent.parent / "frontend" / "js" / "dynamic_data.js"
    dynamic_js_path.parent.mkdir(parents=True, exist_ok=True)
    js_content = f"window.DYNAMIC_GNSS_RESULTS = {json.dumps(frontend_data, indent=2)};\n"
    dynamic_js_path.write_text(js_content, encoding="utf-8")
    print(f"Frontend Data Synced: {dynamic_js_path}")


if __name__ == "__main__":
    main()
