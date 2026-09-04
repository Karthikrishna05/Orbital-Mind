"""Guard the leak-free discipline: model selection must rely on internal
train-holdout validation, never on the day-8 test file."""
import numpy as np

from orbitalmind.dataio import load_dataset
from orbitalmind.backtest import rolling_validation, rolling_day_folds
from orbitalmind.experiments.registry import build_model
from orbitalmind.experiments import run_leaderboard, recommend


def test_rolling_folds_never_include_future_of_fit():
    train, _ = load_dataset("GEO")
    for fit_s, val_t, _ in rolling_day_folds(train, n_folds=3):
        # every validation timestamp is strictly after all fit timestamps
        assert val_t.min() > fit_s.t.max()


def test_rolling_validation_returns_metrics():
    train, _ = load_dataset("MEO2")
    val = rolling_validation(lambda k: build_model("harmonic_robust_p1h1", k),
                             train, kind="MEO", n_folds=3)
    assert val["n_folds"] >= 1
    assert np.isfinite(val["W"])


def test_std_control_blocks_spread_gamer():
    """A high-W model that wins only by inflating validation std must NOT be the
    recommended pick (Priority-4 requirement in the refined plan)."""
    from orbitalmind.experiments.runner import LeaderboardRow
    from orbitalmind.experiments import recommend
    rows = [
        LeaderboardRow(dataset="D", model="median", W_avg=float("nan"), p_avg=float("nan"),
                       H_avg=float("nan"), mean_abs_avg=float("nan"), std_avg=float("nan"),
                       W_pooled=float("nan"), n=0,
                       W_holdout=0.80, std_holdout=1.00, H_holdout=0.0, mean_abs_holdout=0.05),
        LeaderboardRow(dataset="D", model="honest", W_avg=float("nan"), p_avg=float("nan"),
                       H_avg=float("nan"), mean_abs_avg=float("nan"), std_avg=float("nan"),
                       W_pooled=float("nan"), n=0,
                       W_holdout=0.90, std_holdout=1.10, H_holdout=0.0, mean_abs_holdout=0.05),
        LeaderboardRow(dataset="D", model="spread_gamer", W_avg=float("nan"), p_avg=float("nan"),
                       H_avg=float("nan"), mean_abs_avg=float("nan"), std_avg=float("nan"),
                       W_pooled=float("nan"), n=0,
                       W_holdout=0.97, std_holdout=3.00, H_holdout=0.0, mean_abs_holdout=0.05),
    ]
    rec = recommend(rows)["D"]
    assert rec.pick == "honest"                 # not the higher-W spread_gamer
    assert rec.raw_leader == "spread_gamer"
    assert rec.raw_leader_is_spread_gamer is True


def test_recommendation_is_selected_on_holdout_not_test():
    # small model subset keeps this fast; the discipline is what we're testing
    rows, _ = run_leaderboard(
        model_names=["median", "harmonic_robust_p1h1", "gbr_huber"])
    recs = recommend(rows)
    assert recs
    # the pick must have a valid internal-validation score (selection basis)
    for ds, rec in recs.items():
        assert np.isfinite(rec.pick_W_holdout)
