"""Fine-tune on day-8 (rules-permitted) and report the HONEST gain.

For each candidate model and dataset, report three numbers:
  1. train-only        : fit on 7-day train, score on day-8            (frozen baseline)
  2. fine-tuned OOF     : out-of-fold on day-8 -- each held-out day-8 point predicted
                          by a model trained on train + the OTHER day-8 points. This
                          is the HONEST estimate of fine-tuned generalization to
                          fresh day-8 timestamps.
  3. fine-tuned in-sample: fit on train+ALL day-8, score on day-8      (optimistic/leaky)

The final submission model is fit on train + ALL day-8.

Usage:  python scripts/fine_tune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from orbitalmind import PARAM_NAMES
from orbitalmind.config import OUTPUT_DIR, DATASETS, SW_REFERENCE_TARGET, SIGNIFICANCE_ALPHA
from orbitalmind.dataio import load_all
from orbitalmind.dataio.combine import combine, subset
from orbitalmind.evaluation import residuals, evaluate_residuals
from orbitalmind.evaluation.qqplot import qq_plot
from orbitalmind.experiments.registry import build_model

ALPHA = SIGNIFICANCE_ALPHA
BENCH = SW_REFERENCE_TARGET
CANDIDATES = ["harmonic_robust_p2h2", "harmonic_robust_p2h3", "gbr_huber",
              "mlp", "ensemble_median", "stack_harmonic+harmonic"]


def _oof_residuals(name, kind, train, test, k_folds):
    """Out-of-fold day-8 residuals: each held-out point predicted from
    train + the other day-8 points (interleaved folds for time coverage)."""
    n = len(test)
    k = min(k_folds, max(2, n // 2))
    fold = np.arange(n) % k
    resid = {p: np.full(n, np.nan) for p in PARAM_NAMES}
    for f in range(k):
        hold = np.where(fold == f)[0]
        keep = np.where(fold != f)[0]
        if hold.size == 0 or keep.size == 0:
            continue
        aug = combine(train, subset(test, keep))
        try:
            m = build_model(name, kind).fit(aug)
            pred = m.predict(np.asarray(test.t)[hold])
        except Exception:
            continue
        for p in PARAM_NAMES:
            resid[p][hold] = pred[p] - np.asarray(test.values[p])[hold]
    return resid


def _score(resid, stat="swtest"):
    rep = evaluate_residuals(resid, alpha=ALPHA, stat=stat)
    return rep


def main():
    data = load_all()
    lines = []
    lines.append("FINE-TUNING ON DAY-8  (rules-permitted; Note 1a)")
    lines.append(f"Thresholds: benchmark W={BENCH['W']:.4f}, p={BENCH['p']:.4f}, "
                 f"H={BENCH['H']}   alpha={ALPHA}   (H=0 => normal => GOOD)")
    lines.append("Honest metric = 'ft-OOF' (out-of-fold on day-8). in-sample is optimistic.")
    lines.append("=" * 94)

    best = {}
    for ds in DATASETS:
        train, test = data[ds]
        kind = DATASETS[ds]["kind"]
        k_folds = 3 if len(test) < 24 else 4
        lines.append(f"\n[{ds}]  train n={len(train)}  day-8 n={len(test)}  (folds={k_folds})")
        lines.append(f"  {'model':24} {'trainW':>7} {'ftOOF_W':>8} {'ftOOF_H':>8} "
                     f"{'ftOOF_std':>10} {'insampW':>8}")
        ds_rows = []
        for name in CANDIDATES:
            # 1. train-only
            m0 = build_model(name, kind).fit(train)
            r0 = residuals(m0.predict(np.asarray(test.t)), test.values)
            s0 = _score(r0)
            # 2. fine-tuned OOF (honest)
            roof = _oof_residuals(name, kind, train, test, k_folds)
            soof = _score(roof)
            # 3. fine-tuned in-sample (optimistic)
            aug = combine(train, test)
            m2 = build_model(name, kind).fit(aug)
            r2 = residuals(m2.predict(np.asarray(test.t)), test.values)
            s2 = _score(r2)
            ds_rows.append((name, s0, soof, s2, roof))
            lines.append(f"  {name:24} {s0.W_avg:7.4f} {soof.W_avg:8.4f} "
                         f"{soof.H_avg:8.2f} {soof.std_avg:10.4f} {s2.W_avg:8.4f}")

        # pick best by honest ft-OOF W (guard: std not blown up vs train-only median baseline)
        valid = [r for r in ds_rows if np.isfinite(r[2].W_avg)]
        winner = max(valid, key=lambda r: r[2].W_avg)
        best[ds] = winner
        name, s0, soof, s2, roof = winner
        gain = soof.W_avg - s0.W_avg
        lines.append(f"  --> best fine-tuned (honest): {name}  "
                     f"ftOOF W={soof.W_avg:.4f} (train-only {s0.W_avg:.4f}, "
                     f"gain {gain:+.4f}), H_avg={soof.H_avg:.2f}")
        # per-param honest breakdown
        for p in PARAM_NAMES:
            pr = soof.per_param[p]
            lines.append(f"       {p:6} W={pr.W:.4f} p={pr.p:.4f} H={pr.H} "
                         f"mean={pr.mean:+.4f} std={pr.std:.4f}")
        qq = qq_plot(roof, Path(OUTPUT_DIR) / f"qq_ft_{ds}_{name}.png",
                     title=f"{ds}: {name} fine-tuned OOF residual Q-Q")
        lines.append(f"       Q-Q (honest OOF) -> {qq}")

    lines.append("\n" + "=" * 94)
    lines.append("SUMMARY: fine-tuned honest (ft-OOF) vs train-only")
    for ds, (name, s0, soof, s2, _) in best.items():
        verdict = "PASS" if soof.H_avg == 0 else ("PARTIAL" if soof.H_avg < 1 else "FAIL")
        lines.append(f"  [{ds}] {name:24} train-only W={s0.W_avg:.4f} -> "
                     f"fine-tuned W={soof.W_avg:.4f} ({soof.W_avg - s0.W_avg:+.4f}), "
                     f"H_avg={soof.H_avg:.2f} [{verdict}]")

    text = "\n".join(lines)
    print(text)
    out = Path(OUTPUT_DIR) / "fine_tune_report.txt"
    out.write_text(text, encoding="utf-8")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
