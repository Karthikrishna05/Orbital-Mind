"""Turn the leaderboard into a *leak-free* submission pick.

Selection discipline: the recommended model is chosen using ONLY the internal
train-holdout metrics (last day of training held out; the day-8 test file is never
consulted to make the choice). The day-8 score is then reported purely as an
after-the-fact check -- unbiased precisely because we did not select on it.

Within the holdout, Priority 1 ranks by W, but W is inflatable by widening the
residual, so a moderate std guard keeps the pick honest and an egregious inflation
of the holdout leader is flagged as a "spread-gamer".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BASELINES = {"zero", "mean", "median", "persistence"}


@dataclass
class Recommendation:
    dataset: str
    pick: str
    # internal-validation (selection basis)
    pick_W_holdout: float
    pick_std_holdout: float
    pick_H_holdout: float
    baseline_std_holdout: float
    # day-8 after-the-fact check (NOT used for selection)
    pick_W_test: float
    pick_std_test: float
    pick_H_test: float
    # holdout leader for reference
    raw_leader: str
    raw_leader_W_holdout: float
    raw_leader_is_spread_gamer: bool
    reason: str


def _by_dataset(rows):
    out = {}
    for r in rows:
        out.setdefault(r.dataset, []).append(r)
    return out


def recommend(rows, std_tol: float = 0.25, gamer_tol: float = 0.50,
              bias_tol: float = 2.0):
    """Return {dataset: Recommendation}, selecting on internal holdout only.

    std_tol   : allowed fractional excess of a pick's holdout std over the holdout
                baseline std (the anti-spread-gaming guard -- std inflation is the
                only way to fake a higher W, so this is the guard that matters).
    gamer_tol : fractional excess above which the holdout leader is flagged.
    bias_tol  : a pick's |mean| may be up to bias_tol * baseline_std. Mean bias does
                NOT inflate W (W is location-invariant), so this only screens out
                catastrophically drift-prone models (e.g. a raw extrapolator whose
                bias is many times the noise), not modestly-biased good models.
    """
    recs = {}
    for ds, rlist in _by_dataset(rows).items():
        # candidates must have a valid internal-validation score
        cand = [r for r in rlist if np.isfinite(r.W_holdout)]
        if not cand:
            continue

        base_std = min((r.std_holdout for r in rlist
                        if r.model in _BASELINES and np.isfinite(r.std_holdout)),
                       default=np.nan)
        raw_leader = max(cand, key=lambda r: r.W_holdout)

        def within(r, tol):
            # std guard
            if np.isfinite(base_std) and np.isfinite(r.std_holdout):
                if r.std_holdout > base_std * (1.0 + tol):
                    return False
            # mean-bias guard: only screens out catastrophic drift (|mean| many x
            # the noise). Mean bias does not inflate W, so this is deliberately loose.
            if np.isfinite(base_std) and np.isfinite(r.mean_abs_holdout):
                if r.mean_abs_holdout > base_std * bias_tol:
                    return False
            return True

        qualified = [r for r in cand if within(r, std_tol)]
        pool = qualified if qualified else cand
        pick = max(pool, key=lambda r: r.W_holdout)

        gamer = not within(raw_leader, gamer_tol)

        if pick.model == raw_leader.model:
            reason = ("holdout leader also keeps holdout std near baseline "
                      "-> selected on internal validation, no test peeking.")
        else:
            reason = (f"holdout leader '{raw_leader.model}' inflates holdout std; "
                      f"picked the highest holdout-W model within {int(std_tol*100)}% "
                      f"of baseline std. Selection used internal validation only.")

        recs[ds] = Recommendation(
            dataset=ds, pick=pick.model,
            pick_W_holdout=pick.W_holdout, pick_std_holdout=pick.std_holdout,
            pick_H_holdout=pick.H_holdout, baseline_std_holdout=base_std,
            pick_W_test=pick.W_avg, pick_std_test=pick.std_avg, pick_H_test=pick.H_avg,
            raw_leader=raw_leader.model, raw_leader_W_holdout=raw_leader.W_holdout,
            raw_leader_is_spread_gamer=gamer, reason=reason)
    return recs


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  -  "
    return f"{v:.{nd}f}"


def format_recommendations(recs) -> str:
    lines = ["Leak-free submission picks  (selected on INTERNAL train-holdout; "
             "day-8 shown only as an after-the-fact check)"]
    lines.append("=" * 92)
    for ds, rec in recs.items():
        lines.append(f"\n[{ds}]  holdout baseline std={_fmt(rec.baseline_std_holdout,3)}")
        lines.append(f"  PICK              : {rec.pick}")
        lines.append(f"    selected on     : W_holdout={_fmt(rec.pick_W_holdout)} "
                     f"std_holdout={_fmt(rec.pick_std_holdout,3)} "
                     f"H_holdout={_fmt(rec.pick_H_holdout,2)}")
        lines.append(f"    day-8 check     : W={_fmt(rec.pick_W_test)} "
                     f"std={_fmt(rec.pick_std_test,3)} H={_fmt(rec.pick_H_test,2)}  "
                     f"(NOT used to choose)")
        flag = "  <-- SPREAD-GAMER" if rec.raw_leader_is_spread_gamer else ""
        lines.append(f"  holdout leader    : {rec.raw_leader} "
                     f"(W_holdout={_fmt(rec.raw_leader_W_holdout)}){flag}")
        lines.append(f"  reason            : {rec.reason}")
    lines.append("\nNote: the day-8 'check' is an honest generalization estimate "
                 "because selection never used it. The real final round uses fresh "
                 "timestamps, so treat day-8 numbers as indicative, not exact.")
    return "\n".join(lines)
