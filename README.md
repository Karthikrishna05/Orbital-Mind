# OrbitalMind

GNSS clock and ephemeris error prediction for GEO and MEO satellites.

OrbitalMind learns from irregularly sampled satellite history and predicts X, Y,
Z position error and clock error at arbitrary timestamps. It combines orbital
features, harmonic models, Gaussian processes, Kalman filtering, gradient
boosting, and model selection into one evaluation workflow.

## What It Provides

- Satellite error prediction in metres for position and clock channels
- Support for GEO, MEO1, and MEO2 datasets
- Irregular-timestamp handling and duplicate-observation cleanup
- Physics-aware time, astronomical, spectral, and change-point features
- Multiple forecasting models and per-channel model selection
- Rolling validation and backtesting
- Residual normality metrics, confidence intervals, and Q-Q diagnostics
- Experiment registries, leaderboards, reports, and submission generation
- A browser frontend for training trends, model progression, test uploads, and
  statistical results
- A live inference API for uploaded telemetry files

## Applications

OrbitalMind is designed for satellite operations, navigation-system monitoring,
orbit and clock quality assessment, research workflows, and GNSS analytics.
Its modular design supports new model evaluation, satellite-regime comparison,
and integration with operational engineering tools.

## Project Layout

```text
orbitalmind/
  dataio/       Dataset loading, cleaning, and combination
  evaluation/   Metrics, confidence intervals, normality, and Q-Q analysis
  features/     Time, astronomical, spectral, and change-point features
  models/       Forecasting and ensemble model implementations
  backtest/     Query-based validation and backtesting
  experiments/ Model registry, runners, reports, and recommendations
frontend/       Browser dashboard
scripts/        Evaluation, experiment, export, and submission utilities
tests/          Automated tests
outputs/        Reports, leaderboards, plots, and generated submissions
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset in a `Data_PS-08/` directory at the repository root. The
backend discovers this location automatically, so no source changes are needed
on a teammate's machine.

For a dataset stored elsewhere, set `ORBITALMIND_DATA_DIR`:

```bash
export ORBITALMIND_DATA_DIR=/path/to/Data_PS-08
```

The dataset should contain the GEO and MEO training and test CSV files.

The complete dataset is required for the project workflow. The backend fits its
models from the training CSVs when it starts processing inference requests and
caches those models in memory for subsequent uploads. The test CSVs are used by
the evaluation, backtesting, and submission workflows. The reference files are
used by the validation tests. Because trained models are not stored as permanent
artifacts, keep the training data available whenever the backend is restarted.

## Run Experiments

```bash
python scripts/run_experiments.py
```

Results are written to `outputs/`, including leaderboards, recommendations,
reports, and diagnostic plots.

Run the test suite with:

```bash
python -m pytest tests/ -q
```

## Generate a Submission

```bash
python scripts/make_submission.py \
  --train DATA_GEO_Train.csv \
  --kind GEO \
  --model auto \
  --timestamps query.csv \
  --out outputs/submission_geo.csv \
  --truth DATA_GEO_Test.csv
```

The automatic selector evaluates available model families and chooses a model
for each error channel.

## Run the Frontend

Export the measured training window for the dashboard:

```bash
python scripts/export_train_data.py
```

Start the inference API:

```bash
python server.py
```

Keep the backend running while uploading test files. It trains and caches the
appropriate GEO or MEO model from the available training data, then reuses that
model for additional uploaded test files during the same session.

In another terminal, serve the frontend:

```bash
python3 -m http.server 8000 --directory frontend
```

Open <http://localhost:8000> in a browser. The frontend connects to the API at
`http://localhost:5000` and uses bundled evaluation data when the API is
unavailable.

## Add a Model

Implement the `fit(series)` and `predict(t_seconds)` interface from
`orbitalmind/models/base.py`, register the model in
`orbitalmind/experiments/registry.py`, and run the experiment suite to compare
it with the existing methods.
