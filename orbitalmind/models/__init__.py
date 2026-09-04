"""Candidate predictors behind one fit()/predict() interface."""
from .base import Model, PredictMixin
from .persistence import ZeroModel, MeanModel, PersistenceModel
from .harmonic import HarmonicModel
from .auto_harmonic import AutoHarmonicModel
from .ensemble import EnsembleModel
from .gp import GPModel
from .ml import GradientBoostingModel, MLPModel
from .kalman import KalmanLLTModel
from .sarima import SARIMAModel
from .stacking import StackedResidualModel
from .stacking_meta import StackingMetaModel
from .selector import PerChannelSelector
from .regime import RegimeMatchedModel
from .physics import PhysicsHarmonicModel
from .clock_kalman import ClockKalmanModel
from .segmented_clock import SegmentedClockModel
from .composite import CompositeChannelModel

__all__ = [
    "Model",
    "PredictMixin",
    "ZeroModel",
    "MeanModel",
    "PersistenceModel",
    "HarmonicModel",
    "AutoHarmonicModel",
    "EnsembleModel",
    "GPModel",
    "GradientBoostingModel",
    "MLPModel",
    "KalmanLLTModel",
    "SARIMAModel",
    "StackedResidualModel",
    "StackingMetaModel",
    "PerChannelSelector",
    "RegimeMatchedModel",
    "PhysicsHarmonicModel",
    "ClockKalmanModel",
    "SegmentedClockModel",
    "CompositeChannelModel",
]
