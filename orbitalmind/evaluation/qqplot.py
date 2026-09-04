"""Q-Q plots for Priority 3 (visualizing residual outliers)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import PARAM_NAMES
from .shapiro import _ppnd


def _theoretical_quantiles(n: int) -> np.ndarray:
    return np.array([_ppnd((i - 0.375) / (n + 0.25)) for i in range(1, n + 1)])


def qq_plot(resid: dict, path, title: str = "") -> str:
    """Save a 2x2 Q-Q plot (one panel per parameter). Returns the file path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, p in zip(axes.ravel(), PARAM_NAMES):
        r = np.sort(np.asarray(resid[p], dtype=float))
        r = r[np.isfinite(r)]
        n = r.size
        if n < 3:
            ax.set_title(f"{p} (n={n}, too few)")
            continue
        tq = _theoretical_quantiles(n)
        # standardize sample for a comparable reference line
        mu, sd = np.mean(r), np.std(r)
        ax.scatter(tq, r, s=18, alpha=0.7)
        if sd > 0:
            lo, hi = tq.min(), tq.max()
            ax.plot([lo, hi], [mu + sd * lo, mu + sd * hi], "r--", lw=1)
        ax.set_title(f"{p}  (n={n})")
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("residual")
    fig.suptitle(title or "Residual Q-Q")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
