"""Training-ONLY model comparison and pick. The day-8 test file is never touched.

Ranks every registered model by leak-free rolling-fold validation inside the
training data, and reports the defensible per-dataset pick.

Usage:
    python scripts/validate_train.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from orbitalmind.config import OUTPUT_DIR
from orbitalmind.experiments import run_leaderboard, recommend


def main():
    rows, _ = run_leaderboard(include_test=False)

    # group by dataset, rank by rolling-validation W
    datasets = []
    for r in rows:
        if r.dataset not in datasets:
            datasets.append(r.dataset)

    lines = ["TRAINING-ONLY validation (rolling folds; day-8 NEVER used)"]
    lines.append("=" * 80)
    header = f"{'model':26} {'W_val':>7} {'std_val':>8} {'|mean|':>8} {'H_val':>6} {'folds':>5}"
    for ds in datasets:
        drows = [r for r in rows if r.dataset == ds]
        drows.sort(key=lambda r: -(r.W_holdout if np.isfinite(r.W_holdout) else -1e9))
        lines.append(f"\n[{ds}]")
        lines.append(header)
        for r in drows:
            def f(v, nd=4):
                return "  -  " if not np.isfinite(v) else f"{v:.{nd}f}"
            lines.append(f"{r.model:26} {f(r.W_holdout):>7} {f(r.std_holdout,3):>8} "
                         f"{f(r.mean_abs_holdout,3):>8} {f(r.H_holdout,2):>6} {r.n_folds:>5}")

    recs = recommend(rows)
    lines.append("\n" + "=" * 80)
    lines.append("Defensible pick per dataset (leak-free; std + bias guarded):")
    for ds, rec in recs.items():
        lines.append(f"  [{ds}] -> {rec.pick}  "
                     f"(W_val={rec.pick_W_holdout:.4f}, std_val={rec.pick_std_holdout:.3f}, "
                     f"H_val={rec.pick_H_holdout:.2f})")
    text = "\n".join(lines)
    print(text)

    out = Path(OUTPUT_DIR) / "train_only_validation.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
