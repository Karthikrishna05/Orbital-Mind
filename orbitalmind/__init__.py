"""OrbitalMind: GNSS clock/ephemeris error prediction, optimized for
Shapiro-Wilk residual normality (competition PS-08).

The package is organized as a model-agnostic experimentation platform:

    dataio       - load / de-duplicate / clean the irregular CSV data
    evaluation   - own Shapiro-Wilk implementation, confidence intervals, metrics, Q-Q
    features     - time / orbital-harmonic feature construction
    models       - candidate predictors behind one fit()/predict() interface
    backtest     - query-based backtest that mirrors the real day-8 task
    experiments  - run all models -> leaderboard ranked by the real metric

Subpackages are named ``dataio`` / ``evaluation`` (not ``io`` / ``eval``) to
avoid shadowing Python builtins.
"""

__version__ = "0.1.0"

PARAM_NAMES = ("x", "y", "z", "clock")
