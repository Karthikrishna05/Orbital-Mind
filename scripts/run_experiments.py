"""Run the experiment leaderboard and generate Q-Q plots for the best model.

Usage:
    python scripts/run_experiments.py [--stat swtest|shapiro_wilk|shapiro_francia]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orbitalmind.config import OUTPUT_DIR, DATASETS
from orbitalmind.dataio import load_all
from orbitalmind.backtest import day8_split, run_model
from orbitalmind.experiments import (
    run_leaderboard, format_leaderboard, save_leaderboard,
    recommend, format_recommendations,
)
from orbitalmind.experiments.registry import build_model
from orbitalmind.evaluation import residuals
from orbitalmind.evaluation.qqplot import qq_plot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", default="swtest",
                    choices=["swtest", "shapiro_wilk", "shapiro_francia"])
    args = ap.parse_args()

    rows, diagnostics = run_leaderboard(stat=args.stat)
    text = format_leaderboard(rows, diagnostics, stat=args.stat)
    print(text)
    paths = save_leaderboard(rows, diagnostics, stat=args.stat)
    print("\nSaved:", paths["txt"], "and", paths["csv"])

    recs = recommend(rows)
    rec_text = format_recommendations(recs)
    print("\n" + rec_text)
    rec_path = Path(OUTPUT_DIR) / "recommendations.txt"
    rec_path.write_text(rec_text, encoding="utf-8")
    print("\nSaved:", rec_path)

    # Q-Q plot of the DEFENSIBLE pick per dataset (Priority 3).
    data = load_all()
    best_per_ds = {ds: rec.pick for ds, rec in recs.items()}
    plot_paths = []
    for ds, model_name in best_per_ds.items():
        train, test = data[ds]
        kind = DATASETS[ds]["kind"]
        _, q_t, actual = day8_split(train, test)
        res = run_model(build_model(model_name, kind), train, q_t, actual,
                        dataset=ds, stat=args.stat)
        resid = residuals(res.pred, actual)
        out = Path(OUTPUT_DIR) / f"qq_{ds}_{model_name}.png"
        plot_paths.append(qq_plot(resid, out,
                                  title=f"{ds}: {model_name} residual Q-Q"))
    print("Q-Q plots:", *plot_paths, sep="\n  ")


if __name__ == "__main__":
    main()
