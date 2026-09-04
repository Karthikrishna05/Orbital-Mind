"""Loader must de-duplicate exact rows, parse irregular timestamps, and sort."""
import numpy as np

from orbitalmind.dataio import load_dataset


def test_meo_duplicates_removed():
    train, test = load_dataset("MEO1")
    # DATA_MEO_Train.csv repeats the whole 9/1-9/7 block twice.
    assert train.n_duplicates > 0
    assert len(train) == train.n_raw - train.n_duplicates
    # after dedup, timestamps within the train set are unique
    assert len(np.unique(train.t)) == len(train)


def test_geo_has_no_duplicates():
    train, _ = load_dataset("GEO")
    assert train.n_duplicates == 0
    assert len(train) == 142


def test_sorted_and_four_params():
    train, _ = load_dataset("MEO2")
    assert np.all(np.diff(train.t) >= 0)          # sorted ascending
    for p in ("x", "y", "z", "clock"):
        assert p in train.values
        assert train.values[p].shape[0] == len(train)


def test_irregular_sampling_present():
    train, _ = load_dataset("GEO")
    dt = np.diff(train.t)
    # a genuinely irregular series: many distinct gap sizes
    assert np.unique(np.round(dt / 60.0)).size > 5
