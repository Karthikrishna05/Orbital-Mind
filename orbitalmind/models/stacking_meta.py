"""Learned stacking meta-model.

Instead of a plain median of base models, this learns *per-channel weights* by
fitting the base models out-of-fold on the training data (rolling folds) and
solving a non-negative least-squares combination against the held-out truth. The
learned weights are then applied to the base models refit on the full history.

Note: least-squares stacking optimizes *accuracy* (closeness to the true value),
which is the competition's Priority-2 lever and only an indirect help to the
Priority-1 normality score. Judged on training folds like everything else.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .. import PARAM_NAMES
from ..backtest.splits import rolling_day_folds
from .base import Model


class StackingMetaModel(Model):
    def __init__(self, members, kind: str = "GEO", n_folds: int = 3,
                 name: str | None = None):
        self.members = members            # list of (label, factory(kind)->Model)
        self.kind = kind
        self.n_folds = n_folds
        self.name = name or "stacking_meta"
        self._weights = {}                # channel -> weight vector
        self._full = []

    def fit(self, series):
        m = len(self.members)
        # collect out-of-fold base predictions vs truth, per channel
        oof = {p: {"P": [], "a": []} for p in PARAM_NAMES}
        for fit_s, val_t, val_actual in rolling_day_folds(series, n_folds=self.n_folds):
            fitted = []
            for _, factory in self.members:
                try:
                    fitted.append(factory(self.kind).fit(fit_s))
                except Exception:
                    fitted.append(None)
            for p in PARAM_NAMES:
                cols = []
                ok = True
                for mdl in fitted:
                    if mdl is None:
                        ok = False; break
                    cols.append(np.asarray(mdl.predict(val_t)[p], dtype=float))
                if not ok:
                    continue
                oof[p]["P"].append(np.column_stack(cols))
                oof[p]["a"].append(np.asarray(val_actual[p], dtype=float))

        for p in PARAM_NAMES:
            if oof[p]["P"]:
                P = np.vstack(oof[p]["P"]); a = np.concatenate(oof[p]["a"])
                try:
                    w, _ = nnls(P, a)
                    if w.sum() <= 1e-9:
                        w = np.ones(m) / m
                    else:
                        w = w / w.sum()   # convex combination (avoids scale blow-up)
                except Exception:
                    w = np.ones(m) / m
            else:
                w = np.ones(m) / m
            self._weights[p] = w

        # refit all members on the full training history
        self._full = []
        for _, factory in self.members:
            try:
                self._full.append(factory(self.kind).fit(series))
            except Exception:
                self._full.append(None)
        return self

    def predict(self, t_seconds):
        out = {}
        preds = [m.predict(t_seconds) if m is not None else None for m in self._full]
        n = np.asarray(t_seconds).size
        for p in PARAM_NAMES:
            w = self._weights[p]
            acc = np.zeros(n)
            wsum = 0.0
            for wi, pr in zip(w, preds):
                if pr is None:
                    continue
                acc += wi * np.asarray(pr[p], dtype=float)
                wsum += wi
            out[p] = acc / wsum if wsum > 0 else acc
        return out

    def weights_summary(self) -> str:
        labels = [lbl for lbl, _ in self.members]
        parts = []
        for p in PARAM_NAMES:
            w = self._weights.get(p, [])
            parts.append(f"{p}:[" + ",".join(f"{lbl}={wi:.2f}" for lbl, wi in zip(labels, w)) + "]")
        return " ".join(parts)
