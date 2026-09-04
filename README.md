# OrbitalMind — GNSS clock/ephemeris error prediction (PS-08)

Predict the **8th-day** GNSS satellite error (X, Y, Z position + clock, all in
metres) at **arbitrary timestamps**, from 7 days of irregularly-sampled history —
optimised for the way the competition actually scores: **residual normality**.

## The scoring, and why it drives the design

The evaluator forms the residual `r = predicted − actual` per parameter (equal
weight) and measures how **Gaussian** it is:

- **Priority 1** — Shapiro-Wilk `W` (higher = better), p-value, hypothesis `H`
  (0 = fail to reject normality at α=0.05), **averaged over the 4 parameters**.
- **Priority 2** (tiebreak) — mean & std of the residual.
- **Priority 3** (tiebreak) — Q-Q plot.

So the objective is **not** small residuals — it is *Gaussian-shaped* residuals.
A model wins by removing the *systematic* structure (drift + orbital periodicity +
autocorrelation) so the leftover is iid-Gaussian, while staying robust to the
outlier bursts that wreck `W`.

## Two verified facts that change the build

1. **The benchmark statistic is Shapiro-Francia, not Shapiro-Wilk.** The reference
   `SW_ReferenceData.xlsx` is 45 standard-normal values; standard Shapiro-Wilk
   (our code ≡ scipy to 1e-10) gives **W=0.9852**, but the organizer reports
   **W=0.9810, p=0.5840**. That matches the **Shapiro-Francia** branch of MATLAB's
   `swtest` (used when kurtosis > 3): our `swtest()` reproduces **W=0.98139,
   p=0.58383**. The p-value is the tight check — plugging the printed W=0.9810 into
   the SF transform gives p≈0.568, *not* 0.584, so the printed W is just a loose
   rounding of ~0.9814. **We therefore score against `swtest` (evaluator-faithful),
   not pure SW.** On small samples the branch changes the answer a lot.
2. **The MEO CSVs contain exact duplicate blocks** (MEO1 train 90→46 rows, MEO2
   244→143) and sampling is genuinely irregular. The loader de-duplicates.

There is also **no clean "oracle ceiling"**: `W` rewards residual *shape*, so
adding Gaussian spread can *raise* `W` by masking real outliers. GEO day-8 truth
carries extreme outlier bursts (±75 m) that land in any model's residual, capping
GEO `W`≈0.78 with H=1. This is why **Priority-2 std is the essential tiebreak** and
why the report shows truth-diagnostics instead of a fake ceiling.

## Layout

```
orbitalmind/
  dataio/       load + de-duplicate + robust outlier utils
  evaluation/   own Shapiro-Wilk (Royston AS R94) + Shapiro-Francia + swtest,
                bootstrap CI, metrics (Priority 1/2), Q-Q (Priority 3)
  features/     time / orbital-harmonic Fourier features (any timestamp)
  models/       zero, mean/median, persistence, harmonic(+trend, robust)
  backtest/     query-based backtest mirroring the real day-8 task
  experiments/  registry + runner + report -> leaderboard by the real metric
scripts/        run_experiments.py, make_submission.py
tests/          SW benchmark + loader dedup
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Data path defaults to `C:\Users\pc\Downloads\Data_PS-08`; override with the
`ORBITALMIND_DATA_DIR` environment variable.

## Run

```bash
.venv\Scripts\python -m pytest tests/ -q                 # SW benchmark + loader
.venv\Scripts\python scripts/run_experiments.py          # leaderboard + Q-Q plots
```

The leaderboard (and CSV + Q-Q PNGs) is written to `outputs/`, along with
`recommendations.txt` — the defensible submission pick per dataset (max W with std
held near the irreducible baseline, flagging any raw-W "spread-gamer"). Use
`--stat shapiro_wilk` to also see the Note's literal statistic.

### Make a submission

```bash
.venv\Scripts\python scripts/make_submission.py \
  --train DATA_GEO_Train.csv --kind GEO --model auto \
  --timestamps query.csv --out outputs/submission_geo.csv [--truth DATA_GEO_Test.csv]
```

`--model auto` uses the per-channel selector; `--timestamps` is a file with one
`M/D/YYYY H:MM` per line. With `--truth`, it also prints the residual normality
report (W, p, H, and confidence interval) per channel.

## Models implemented

Baselines: `zero`, `mean`, `median`, `persistence`. Bar-to-beat: robust/OLS
harmonic+trend (`harmonic_*`). Challengers: Gaussian Process (`gp_matern_periodic`
— Gaussian residuals + predictive std for the CI), gradient boosting with Huber
loss (`gbr_huber`), an MLP (`mlp`), a self-contained local-linear-trend Kalman
filter (`kalman_llt`, irregular-`dt` aware), residual-whitening stacking wrappers
(`stack_harmonic+gp`, `stack_harmonic+harmonic`), and a **per-channel meta-model**
(`per_channel_best`) that picks the best-whitening model separately for each of
X/Y/Z/clock (the score is a 4-channel average, so this maximizes it honestly).

## Reading the leaderboard (a real tradeoff the platform exposes)

Under the Note's literal rules, ranking is by Priority-1 `W` alone; Priority-2
(std) only breaks *ties*. The experiments make a genuine tension visible: flexible
models (notably `mlp`) can post the **highest W by injecting Gaussian-like spread
that masks the ground-truth outliers** — e.g. on GEO `mlp` reaches W≈0.90 but with
residual std ≈17 vs the harmonic's ≈15. So the leaderboard reports `std`,
`W_holdout` (internal rolling validation, day-8 excluded), and truth-diagnostics
alongside `W`, so you can tell a model that genuinely *whitens* the residual from
one that just widens it. On MEO1 the robust harmonic reaches W≈0.94 with H=0 (all
four params fail-to-reject normality) at low std — the cleanest kind of win.

## Extending (the experimental loop)

Add a model behind `models/base.Model` (`fit(series)` / `predict(t_seconds)`),
register it in `experiments/registry.py`, and re-run `run_experiments.py`. Still
open as challengers: Kalman/state-space for the clock drift, and a proper GEO
outlier-treatment stage to lift its residual normality past the outlier cap.
```
