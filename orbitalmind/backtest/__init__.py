"""Query-based backtest that mirrors the real day-8 task."""
from .splits import day8_split, holdout_last_day, rolling_day_folds
from .harness import run_model, truth_diagnostics, rolling_validation, BacktestResult

__all__ = [
    "day8_split",
    "holdout_last_day",
    "rolling_day_folds",
    "run_model",
    "truth_diagnostics",
    "rolling_validation",
    "BacktestResult",
]
