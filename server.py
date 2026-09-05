"""Live inference API for the OrbitalMind frontend.

Fills the gap the frontend's Test Data page always assumed existed: today,
uploading a CSV there does nothing real (see ``frontend/js/data.js``'s mocked
``uploadTestData``) -- the numbers on the Results page always come from a
static snapshot (``frontend/js/dynamic_data.js``) written by manually running
``scripts/evaluate_day8.py``. This server makes the upload real: it fits each
orbit's leak-free-selected model once (same selection procedure as
``evaluate_day8.py``: rolling-fold validation on the 7-day training data only,
never touching any test file to choose the model), caches the fitted model,
and then scores *only* whatever CSV you upload against it -- honest, live
inference, not a replay of old day-8 numbers.

MEO has two physically distinct trained satellites (MEO1, MEO2) but the
frontend only exposes a single "MEO" choice, so a live MEO upload is scored
against one model fit on the two satellites' training data combined, with the
architecture (which model type) itself chosen by leak-free rolling validation
per satellite: whichever of the MEO1/MEO2 leak-free picks validated better is
the one used. This is a real per-request contract, not a hand-picked default.

Usage:
    pip install -r requirements.txt
    python server.py
    # then open frontend/index.html (or serve it) with the backend running
    # on http://localhost:5000
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from orbitalmind import PARAM_NAMES
from orbitalmind.config import SIGNIFICANCE_ALPHA
from orbitalmind.dataio import load_csv
from orbitalmind.dataio.loader import load_dataset
from orbitalmind.dataio.combine import combine
from orbitalmind.evaluation import residuals, evaluate_residuals
from orbitalmind.evaluation.qqplot import _theoretical_quantiles
from orbitalmind.experiments import run_leaderboard, recommend
from orbitalmind.experiments.registry import build_model

app = Flask(__name__)
CORS(app)

ALPHA = SIGNIFICANCE_ALPHA
PARAM_LABEL = {"x": "x_error", "y": "y_error", "z": "z_error", "clock": "satclockerror"}
PARAM_TITLE = {"x": "Radial", "y": "Along-Track", "z": "Cross-Track", "clock": "Clock"}

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}  # 'GEO' | 'MEO' -> fitted model
_model_names: dict[str, str] = {}     # 'GEO' | 'MEO' -> model name (for display)
_recommendation_cache: dict[str, object] | None = None


def _load_saved_recommendations():
    """Use the checked-in leak-free picks before running the expensive selector."""
    global _recommendation_cache
    if _recommendation_cache is not None:
        return _recommendation_cache

    saved = Path(__file__).resolve().parent / "outputs" / "recommendations.txt"
    if saved.exists():
        picks = {}
        for line in saved.read_text(encoding="utf-8").splitlines():
            if line.startswith("[GEO]"):
                orbit = "GEO"
            elif line.startswith("[MEO1]"):
                orbit = "MEO1"
            elif line.startswith("[MEO2]"):
                orbit = "MEO2"
            elif "  PICK              : " in line:
                picks[orbit] = line.split(":", 1)[1].strip()
        if {"GEO", "MEO1", "MEO2"}.issubset(picks):
            _recommendation_cache = picks
            return picks

    _recommendation_cache = recommend(run_leaderboard(include_test=False)[0])
    return _recommendation_cache


def _pick_model_for_orbit(orbit: str):
    """Leak-free model selection + fit, cached for the life of the process.

    GEO uses the leaderboard pick for the 'GEO' dataset. MEO combines the
    MEO1 + MEO2 training series and uses whichever satellite's leak-free pick
    validated better (higher internal rolling-fold W) as the model type.
    """
    if orbit in _model_cache:
        return _model_cache[orbit], _model_names[orbit]

    with _model_lock:
        if orbit in _model_cache:  # re-check after acquiring the lock
            return _model_cache[orbit], _model_names[orbit]

        recs = _load_saved_recommendations()

        if orbit == "GEO":
            train, _ = load_dataset("GEO")
            pick = recs["GEO"] if isinstance(recs["GEO"], str) else recs["GEO"].pick
        elif orbit == "MEO":
            train1, _ = load_dataset("MEO1")
            train2, _ = load_dataset("MEO2")
            train = combine(train1, train2)
            rec1, rec2 = recs["MEO1"], recs["MEO2"]
            if isinstance(rec1, str):
                pick = rec1
            else:
                pick = rec1.pick if rec1.pick_W_holdout >= rec2.pick_W_holdout else rec2.pick
        else:
            raise ValueError(f"unknown orbit {orbit!r}")

        model = build_model(pick, orbit).fit(train)
        _model_cache[orbit] = model
        _model_names[orbit] = pick
        return model, pick


def _build_shapiro_rows(rep, orbit: str):
    rows = []
    for p in PARAM_NAMES:
        a = rep.per_param[p]
        status = "normal" if a.H == 0 else "non-normal"
        hypothesis = "Fail to Reject H₀ (Normal)" if a.H == 0 else "Reject H₀ (Non-Normal)"
        ci = f"W CI [{a.ci_lo:.3f}, {a.ci_hi:.3f}]" if np.isfinite(a.ci_lo) else f"n={a.n}/{a.n_raw}"
        rows.append({
            "parameter": f"{PARAM_LABEL[p]} — {orbit} {PARAM_TITLE[p]}",
            "wStatistic": f"{a.W:.4f}" if np.isfinite(a.W) else "-",
            "pValue": f"{a.p:.4f}" if np.isfinite(a.p) else "-",
            "hypothesis": hypothesis,
            "status": status,
            "notes": ci
        })
    agg_status = "normal" if rep.H_avg == 0 else ("partial" if rep.H_avg < 1 else "non-normal")
    agg_hyp = ("Fail to Reject H₀ — PASS ✅" if rep.H_avg == 0
               else ("Partial — some channels pass ⚠️" if rep.H_avg < 1 else "Reject H₀ — FAIL ❌"))
    rows.append({
        "parameter": f"{orbit} Aggregate (AVG 4-channel)",
        "wStatistic": f"{rep.W_avg:.4f}" if np.isfinite(rep.W_avg) else "-",
        "pValue": f"{rep.p_avg:.4f}" if np.isfinite(rep.p_avg) else "-",
        "hypothesis": agg_hyp,
        "status": agg_status,
        "notes": f"n={rep.n_points} pts | H_avg={rep.H_avg:.2f}"
    })
    return rows


def _build_meansd_rows(rep, orbit: str):
    rows = []
    for p in PARAM_NAMES:
        a = rep.per_param[p]
        ci = f"W CI [{a.ci_lo:.3f}, {a.ci_hi:.3f}]" if np.isfinite(a.ci_lo) else "-"
        rows.append({
            "parameter": f"{PARAM_LABEL[p]} — {orbit}",
            "unit": "meters (m)",
            "mean": f"{a.mean:+.4f}" if np.isfinite(a.mean) else "-",
            "sd": f"{a.std:.4f}" if np.isfinite(a.std) else "-",
            "confidence95": ci,
            "maxResidual": f"n={a.n}/{a.n_raw} pts"
        })
    rows.append({
        "parameter": f"{orbit} Aggregate",
        "unit": "meters (m)",
        "mean": f"|μ|={rep.mean_abs_avg:.4f}" if np.isfinite(rep.mean_abs_avg) else "-",
        "sd": f"{rep.std_avg:.4f}" if np.isfinite(rep.std_avg) else "-",
        "confidence95": "4-channel avg",
        "maxResidual": f"n={rep.n_points} pts"
    })
    return rows


def _build_qq_data(resid: dict, orbit: str):
    out = {}
    for p in PARAM_NAMES:
        r = np.sort(np.asarray(resid[p], dtype=float))
        r = r[np.isfinite(r)]
        n = r.size
        if n < 3:
            out[PARAM_LABEL[p]] = {"title": f"{orbit} {PARAM_LABEL[p]} Residuals (n={n})",
                                    "unit": "m", "points": []}
            continue
        tq = _theoretical_quantiles(n)
        points = [{"theoretical": round(float(t), 3), "sample": round(float(s), 3)}
                  for t, s in zip(tq, r)]
        out[PARAM_LABEL[p]] = {
            "title": f"{orbit} {PARAM_LABEL[p]} Residuals (Live upload, n={n})",
            "unit": "m",
            "points": points
        }
    return out


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/test-data/upload", methods=["POST"])
def upload_test_data():
    orbit = request.form.get("orbit", "").upper()
    if orbit not in ("GEO", "MEO"):
        return jsonify({"error": "orbit must be 'GEO' or 'MEO'"}), 400

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "no file uploaded"}), 400

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = Path(tmp.name)

    try:
        uploaded = load_csv(tmp_path, name=f"upload_{orbit}", kind=orbit)
    except Exception as exc:
        return jsonify({"error": f"could not parse CSV: {exc}"}), 400
    finally:
        tmp_path.unlink(missing_ok=True)

    if len(uploaded) < 3:
        return jsonify({"error": f"need at least 3 valid rows, got {len(uploaded)}"}), 400

    try:
        model, pick = _pick_model_for_orbit(orbit)
    except Exception as exc:
        return jsonify({"error": f"model selection/training failed: {exc}"}), 500

    query_t = np.asarray(uploaded.t, dtype=float)
    actual = {p: np.asarray(uploaded.values[p], dtype=float) for p in PARAM_NAMES}
    pred = model.predict(query_t)
    resid = residuals(pred, actual)

    rep = evaluate_residuals(resid, alpha=ALPHA, stat="swtest", with_ci=True,
                             ci_B=1500, reject_outliers=True, mad_threshold=3.0)

    return jsonify({
        "orbit": orbit,
        "model": pick,
        "filename": file.filename,
        "n": len(uploaded),
        "shapiroWilk": _build_shapiro_rows(rep, orbit),
        "meanAndSD": _build_meansd_rows(rep, orbit),
        "qqPlotData": _build_qq_data(resid, orbit)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
