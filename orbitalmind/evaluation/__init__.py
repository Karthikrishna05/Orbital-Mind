"""Evaluation: own Shapiro-Wilk / Shapiro-Francia, CIs, metrics, Q-Q plots."""
from .shapiro import (
    shapiro_wilk,
    shapiro_francia,
    swtest,
    ShapiroResult,
    STAT_FUNCS,
)
from .confidence import bootstrap_ci_W
from .metrics import (
    residuals,
    evaluate_residuals,
    ParamResult,
    ResidualReport,
)

__all__ = [
    "shapiro_wilk",
    "shapiro_francia",
    "swtest",
    "STAT_FUNCS",
    "ShapiroResult",
    "bootstrap_ci_W",
    "residuals",
    "evaluate_residuals",
    "ParamResult",
    "ResidualReport",
]
