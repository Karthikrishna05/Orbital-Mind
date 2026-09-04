"""Format and persist the leaderboard."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import OUTPUT_DIR


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  -  "
    return f"{v:.{nd}f}"


def format_leaderboard(rows, diagnostics=None, stat: str = "swtest") -> str:
    """Return a plain-text leaderboard grouped by dataset, best W first."""
    diagnostics = diagnostics or {}
    lines = []
    lines.append(f"Leaderboard  (statistic = {stat};  W = Priority-1 score, higher is better)")
    lines.append("=" * 92)
    datasets = []
    for r in rows:
        if r.dataset not in datasets:
            datasets.append(r.dataset)

    header = (f"{'model':22} {'W_avg':>7} {'p_avg':>7} {'H_avg':>6} "
              f"{'|mean|':>8} {'std':>8} {'W_hold':>7} {'W_pool':>7} {'n':>4}")
    for ds in datasets:
        diag = diagnostics.get(ds, {})
        diag_txt = ""
        if diag:
            diag_txt = (f"   (truth: W_raw={_fmt(diag.get('W_raw'))}, "
                        f"W_detrended={_fmt(diag.get('W_detrended'))})")
        lines.append("")
        lines.append(f"[{ds}]{diag_txt}")
        lines.append("-" * 92)
        lines.append(header)
        for r in rows:
            if r.dataset != ds:
                continue
            lines.append(
                f"{r.model:22} {_fmt(r.W_avg):>7} {_fmt(r.p_avg):>7} "
                f"{_fmt(r.H_avg,2):>6} {_fmt(r.mean_abs_avg,3):>8} {_fmt(r.std_avg,3):>8} "
                f"{_fmt(r.W_holdout):>7} {_fmt(r.W_pooled):>7} {r.n:>4}")
    lines.append("")
    lines.append("Notes: swtest = evaluator-faithful (Shapiro-Francia when kurtosis>3, else "
                 "Shapiro-Wilk).")
    lines.append("       H_avg is the mean reject-rate over 4 params (0 = all fail-to-reject = good).")
    lines.append("       W_hold = internal rolling validation (day-8 excluded); guards against "
                 "overfitting one realization.")
    lines.append("       truth W_raw/W_detrended = how non-Gaussian the day-8 ground truth is "
                 "(context, NOT a hard ceiling:")
    lines.append("       W rewards residual shape, so added spread can raise W above these -- "
                 "why Priority-2 std is the tiebreak).")
    return "\n".join(lines)


def save_leaderboard(rows, diagnostics=None, stat: str = "swtest",
                     out_dir=None) -> dict:
    """Write leaderboard.txt and leaderboard.csv; return paths."""
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = format_leaderboard(rows, diagnostics, stat=stat)
    txt_path = out_dir / "leaderboard.txt"
    txt_path.write_text(txt, encoding="utf-8")

    csv_path = out_dir / "leaderboard.csv"
    cols = ["dataset", "model", "W_avg", "p_avg", "H_avg", "mean_abs_avg",
            "std_avg", "W_holdout", "W_pooled", "n"]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(getattr(r, c)) for c in cols) + "\n")
    return {"txt": str(txt_path), "csv": str(csv_path)}
