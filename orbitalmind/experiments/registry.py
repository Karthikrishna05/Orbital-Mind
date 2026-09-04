"""Registry of candidate models.

Each entry maps a name to a factory ``build(kind) -> Model``. Add challengers
(GP, Kalman, ML, whitening wrappers) here; the leaderboard decides what ships.
"""
from __future__ import annotations

from ..models import (
    ZeroModel, MeanModel, PersistenceModel, HarmonicModel, AutoHarmonicModel,
    EnsembleModel, GPModel, GradientBoostingModel, MLPModel, KalmanLLTModel,
    SARIMAModel, StackedResidualModel, StackingMetaModel, PerChannelSelector,
    RegimeMatchedModel, PhysicsHarmonicModel, ClockKalmanModel, SegmentedClockModel,
    CompositeChannelModel,
)

MODEL_REGISTRY = {}


def register(name, factory):
    MODEL_REGISTRY[name] = factory
    return factory


def build_model(name, kind="GEO"):
    return MODEL_REGISTRY[name](kind)


# --- floor baselines ----------------------------------------------------------
register("zero", lambda kind: ZeroModel())
register("mean", lambda kind: MeanModel(robust=False))
register("median", lambda kind: MeanModel(robust=True))
register("persistence", lambda kind: PersistenceModel())

# --- bar-to-beat: harmonic + trend (robust and OLS, a couple of complexities) -
register("harmonic_ols_p1h1",
         lambda kind: HarmonicModel(kind=kind, n_poly=1, harmonics=1, robust=False))
register("harmonic_robust_p1h1",
         lambda kind: HarmonicModel(kind=kind, n_poly=1, harmonics=1, robust=True))
register("harmonic_robust_p2h2",
         lambda kind: HarmonicModel(kind=kind, n_poly=2, harmonics=2, robust=True))
register("harmonic_robust_p2h3",
         lambda kind: HarmonicModel(kind=kind, n_poly=2, harmonics=3, robust=True))

# --- challengers --------------------------------------------------------------
register("gp_matern_periodic", lambda kind: GPModel(kind=kind, n_restarts=4))
register("gbr_huber", lambda kind: GradientBoostingModel(kind=kind, n_poly=2, harmonics=2))
register("mlp", lambda kind: MLPModel(kind=kind, n_poly=2, harmonics=2))
register("kalman_llt", lambda kind: KalmanLLTModel(kind=kind))

# --- regime-matched training (route dense/coarse cadence to matched sub-models)
register("regime_harmonic", lambda kind: RegimeMatchedModel(
    lambda k: HarmonicModel(kind=k, n_poly=2, harmonics=2, robust=True),
    kind=kind, name="regime_harmonic"))
register("regime_gbr", lambda kind: RegimeMatchedModel(
    lambda k: GradientBoostingModel(kind=k, n_poly=2, harmonics=2),
    kind=kind, name="regime_gbr"))

# --- physics-informed features (solar geometry / eclipse proxies) -------------
register("physics_harmonic", lambda kind: PhysicsHarmonicModel(kind=kind))

# --- two-state clock Kalman (satclockerror only; harmonic for x/y/z) ----------
register("clock_kalman", lambda kind: ClockKalmanModel(kind=kind))

# --- change-point-aware clock (fits current segment after last upload reset) ---
register("segmented_clock", lambda kind: SegmentedClockModel(kind=kind))

# --- composite: strong stacked positions (x/y/z) + change-point clock ----------
# Principled composition (fixed by design, no selection leakage): the stacked
# harmonic model owns the position channels, the segmented clock owns the clock.
def _stack_positions(k):
    return StackedResidualModel(
        HarmonicModel(kind=k, n_poly=1, harmonics=1, robust=True),
        HarmonicModel(kind=k, n_poly=1, harmonics=3, robust=True),
        name="stack_harmonic+harmonic")

register("composite_pos_clock", lambda kind: CompositeChannelModel(
    {"x": "pos", "y": "pos", "z": "pos", "clock": "clk"},
    {"pos": _stack_positions, "clk": lambda k: SegmentedClockModel(kind=k)},
    kind=kind, name="composite_pos_clock"))

# --- data-driven periods (Lomb-Scargle) ---------------------------------------
register("auto_harmonic_np2h1", lambda kind: AutoHarmonicModel(kind=kind, n_periods=2, n_poly=2, harmonics=1))
register("auto_harmonic_np3h2", lambda kind: AutoHarmonicModel(kind=kind, n_periods=3, n_poly=2, harmonics=2))

# --- SARIMA (seasonal ARIMA on a resampled grid) ------------------------------
register("sarima", lambda kind: SARIMAModel(kind=kind, order=(1, 0, 1), seasonal=True))

# --- ensemble -----------------------------------------------------------------
_ENSEMBLE_MEMBERS = [
    ("harmonic_robust_p2h2", lambda k: HarmonicModel(kind=k, n_poly=2, harmonics=2, robust=True)),
    ("auto_harmonic_np3h2", lambda k: AutoHarmonicModel(kind=k, n_periods=3, n_poly=2, harmonics=2)),
    ("gbr_huber", lambda k: GradientBoostingModel(kind=k, n_poly=2, harmonics=2)),
]
register("ensemble_median", lambda kind: EnsembleModel(
    _ENSEMBLE_MEMBERS, kind=kind, combine="median"))

# --- learned stacking meta-model (per-channel NNLS weights) -------------------
register("stacking_meta", lambda kind: StackingMetaModel(
    _ENSEMBLE_MEMBERS, kind=kind))

# --- residual-whitening (stacking) wrappers -----------------------------------
register("stack_harmonic+gp", lambda kind: StackedResidualModel(
    HarmonicModel(kind=kind, n_poly=2, harmonics=2, robust=True),
    GPModel(kind=kind, n_restarts=1)))
register("stack_harmonic+harmonic", lambda kind: StackedResidualModel(
    HarmonicModel(kind=kind, n_poly=1, harmonics=1, robust=True),
    HarmonicModel(kind=kind, n_poly=1, harmonics=3, robust=True),
    name="stack_harmonic+harmonic"))

# --- meta-model: pick the best-whitening model per channel --------------------
# Curated candidate set (spread-gamers are allowed in but filtered by the std guard).
_SELECTOR_CANDIDATES = [
    ("median", lambda kind: MeanModel(robust=True)),
    ("harmonic_robust_p1h1", lambda kind: HarmonicModel(kind=kind, n_poly=1, harmonics=1, robust=True)),
    ("harmonic_robust_p2h2", lambda kind: HarmonicModel(kind=kind, n_poly=2, harmonics=2, robust=True)),
    ("harmonic_robust_p2h3", lambda kind: HarmonicModel(kind=kind, n_poly=2, harmonics=3, robust=True)),
    ("auto_harmonic_np3h2", lambda kind: AutoHarmonicModel(kind=kind, n_periods=3, n_poly=2, harmonics=2)),
    ("gbr_huber", lambda kind: GradientBoostingModel(kind=kind, n_poly=2, harmonics=2)),
    ("gp_matern_periodic", lambda kind: GPModel(kind=kind, n_restarts=2)),
    ("clock_kalman", lambda kind: ClockKalmanModel(kind=kind)),
    ("segmented_clock", lambda kind: SegmentedClockModel(kind=kind)),
    ("kalman_llt", lambda kind: KalmanLLTModel(kind=kind)),
]
register("per_channel_best",
         lambda kind: PerChannelSelector(_SELECTOR_CANDIDATES, kind=kind))
