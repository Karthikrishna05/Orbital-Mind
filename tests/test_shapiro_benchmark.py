"""Validate the own SW / SF implementation against the organizer's benchmark
and against scipy (cross-check only)."""
import numpy as np
import pytest

from orbitalmind.config import SW_REFERENCE_FILE
from orbitalmind.evaluation import shapiro_wilk, shapiro_francia, swtest


def _load_reference():
    import openpyxl
    wb = openpyxl.load_workbook(SW_REFERENCE_FILE, data_only=True)
    ws = wb.active
    vals = []
    for row in ws.iter_rows(values_only=True):
        for c in row:
            if isinstance(c, (int, float)):
                vals.append(float(c))
    return np.array(vals)


def test_reference_has_45_values():
    x = _load_reference()
    assert x.size == 45


def test_swtest_matches_benchmark_pvalue():
    """The evaluator-faithful statistic reproduces the benchmark.

    Benchmark prints W=0.9810, p=0.5840, H=0. The p-value is the tight check
    (the printed W is a loose rounding of ~0.9814, confirmed by its own p-value).
    """
    x = _load_reference()
    r = swtest(x)
    assert r.method.endswith("shapiro_francia")   # kurtosis>3 branch
    assert abs(r.p - 0.5840) < 0.003
    assert abs(r.W - 0.9814) < 0.001
    assert r.H == 0


def test_shapiro_francia_equals_swtest_here():
    x = _load_reference()
    assert abs(shapiro_francia(x).W - swtest(x).W) < 1e-12


def test_shapiro_wilk_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    for n in (3, 6, 20, 50, 100, 300):
        s = rng.normal(size=n)
        ours = shapiro_wilk(s)
        W, p = scipy_stats.shapiro(s)
        assert abs(ours.W - W) < 1e-6, (n, ours.W, W)
        assert abs(ours.p - p) < 1e-5, (n, ours.p, p)


def test_hypothesis_flag_rejects_nonnormal():
    rng = np.random.default_rng(3)
    skewed = rng.exponential(size=200)   # clearly non-normal
    assert swtest(skewed).H == 1
