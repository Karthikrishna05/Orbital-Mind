"""Smoke tests: every registered model fits on 7-day history and predicts at
arbitrary query timestamps with the right shape and finite values."""
import numpy as np
import pytest

from orbitalmind import PARAM_NAMES
from orbitalmind.dataio import load_dataset
from orbitalmind.experiments.registry import MODEL_REGISTRY, build_model


@pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
def test_model_fit_predict_arbitrary_timestamps(name):
    train, test = load_dataset("MEO1")
    model = build_model(name, "MEO").fit(train)
    # predict at the (irregular) day-8 timestamps
    pred = model.predict(np.asarray(test.t, dtype=float))
    for p in PARAM_NAMES:
        assert pred[p].shape[0] == len(test)
        assert np.all(np.isfinite(pred[p]))


def test_predict_at_single_and_novel_timestamp():
    train, _ = load_dataset("GEO")
    model = build_model("harmonic_robust_p2h2", "GEO").fit(train)
    # a timestamp well beyond the training window
    future = np.array([train.t[-1] + 3 * 86400.0])
    pred = model.predict(future)
    for p in PARAM_NAMES:
        assert pred[p].shape == (1,)
        assert np.isfinite(pred[p][0])
