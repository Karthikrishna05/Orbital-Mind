# How other SIH25176 teams handle overfitting and the Shapiro-Wilk objective

Surveyed the ~20 teams found earlier working this same ISRO problem statement.
For each claim below, the label tells you how solid it is: **[verified in code]**
(I downloaded and read the actual source), **[claimed]** (their README says so,
not independently re-run), or **[inferred]** (my reading of what's visible).

---

## Summary table

| Team | Overfitting mitigation | Shapiro-Wilk strategy | Legitimate? |
|---|---|---|---|
| `codewithRahul01/GNSS-Error-prediction` | Winsorization (train-only), TimeSeriesSplit CV, low-capacity GP kernels, **regime-matched training window** (GEO trained only on Sep 6-7 to match test's 15-min sampling mode) | Kernel engineering per satellite type; **no residual transform** | ✅ Yes — closest in spirit to our approach |
| `gankit-aiml/TeamID_59634_SUB2` | Physics-informed features (Beta angle, eclipse), Lomb-Scargle periodicity detection, GBRT + GPR (bounded capacity) | Data-driven periods + physical feature engineering, not post-hoc correction | ✅ Yes |
| `Tamaghna1/SIH2025` | Time-based split (no shuffling), **dropout**, **Huber loss**, **EarlyStopping** | Model sophistication only — "capture systematic pattern, leave white noise" — **no transform** | ✅ Yes |
| `Krishna-mishra-26/SIH-NAVPREDICT` | LSTM + Dropout(0.2) | Diagnostic-only "Normality Score >0.85" threshold, no gaming | ✅ Yes (basic) |
| `awasthi108/NavAi-backend` | Ridge regression (L2) | Not addressed | ✅ Yes (basic, no real SW strategy) |
| **`Mapicx/TimeForge_2.0`** | "aggressive regularization" **[claimed]**, otherwise unspecified | **[verified in code]** A "Stage 2 Gaussianization" step that directly reshapes predictions toward the training data's own ground truth via a Shapiro-p-maximizing grid search, with a Yeo-Johnson+QuantileTransformer fallback that force-Gaussianizes the residual | ⚠️ **Methodologically unsound** — see below |

---

## The one substantive finding: `Mapicx/TimeForge_2.0`'s "Stage 2 Gaussianization"

This is the only team whose approach to the Shapiro-Wilk objective goes beyond
normal modelling, so it's worth documenting precisely. I downloaded and read
`Model/residual_gaussianization_stage2.py` in full (266 lines) — **[verified in
code]**, not summarized secondhand.

### What it actually does
1. Loads a **training** file only (`DATA_GEO_Train.csv` / `DATA_MEO_Train.csv` /
   `DATA_MEO_Train2.csv` — confirmed no `*_Test.csv` is ever touched by any of the
   four gaussianization scripts in the repo).
2. Takes the model's own prediction on that training data (`y_pred_stage1`) and
   the **known true values** (`y_true`, from the same training file).
3. Computes `residuals = y_true - y_pred_stage1`, then **grid-searches** over
   blend factors, clamp widths, and mixing weights, at each step nudging the
   prediction toward a spline-smoothed, Gaussian-quantile-matched target —
   explicitly optimizing `score = Shapiro_p + 0.1·R²` (Shapiro weighted 10× R²).
4. If Shapiro p is still < 0.05 after that search, it **falls back to fitting a
   Yeo-Johnson power transform + a `QuantileTransformer(output_distribution=
   "normal")` directly on the residual itself**, then reconstructs the "corrected"
   prediction as `y_pred + transformed_residual`. A `QuantileTransformer` set to
   `"normal"` output **forces any residual into a Gaussian shape by construction**
   — it will report a passing Shapiro score on literally any input.

### What the numbers show (from `stage2_eval.txt`, their own output files)

| Series | Stage-1 R² | Stage-1 Shapiro p | Final R² | Final Shapiro p |
|---|---|---|---|---|
| GEO | **0.874** (good fit) | 0.0000425 (fail) | **0.677** (much worse fit) | 0.439 (pass) |
| MEO1 | 0.929 | 0.0002 (fail) | **−0.72** (worse than predicting the mean!) | 0.839 (pass) |
| MEO2 | 0.999 | 0.0023 (fail) | 0.997 | 0.903 (pass) |

**On GEO and especially MEO1, they bought a passing Shapiro score by destroying
prediction accuracy** — MEO1's R² of −0.72 means the "corrected" predictions are
worse than simply guessing the training mean for every point, yet the residual
looks Gaussian because it was built by construction to look Gaussian.

### Why this doesn't (and can't) generalize
- It only ever runs on **training** data — no `Test.csv` is loaded anywhere, so
  this is not literal test-answer leakage.
- But it **is** a category of the same underlying flaw our own project's other
  competitor's `DECISIONS.md` explicitly names as a past incident (a
  "manufactured p=0.9999" by reshaping residuals to match a target): **the
  correction functions require `y_true` as a direct input.** There is no fitted,
  frozen transform saved anywhere that could be applied to day-8 predictions
  where the true values are unknown — the technique is mathematically undefined
  at real evaluation time. So even taken at face value, "Stage 2" cannot be part
  of an actual submission; it can only ever report a self-referential number on
  data it already has the answer key for.
- If their README's headline "Shapiro p = 0.439" **[claimed]** is this GEO
  "Final" number, it is measuring how well a truth-informed post-hoc correction
  can reshape a known residual — not anything about their model's real predictive
  skill on unseen day-8 timestamps.

### The direct parallel to our own project
This is exactly the category of technique we tested and **rejected** in Phase 6/7
of our own build: an additive bias correction, and later a blanket
Yeo-Johnson transform on the target. We measured that a transform fit this way
either didn't transfer to genuinely held-out data or made Priority-2 (mean) worse
on most series, and we removed it — see `PROGRESS.md` Phase 6 point on the
bias-correction rejection, and the honest fine-tuning study in Phase 8. The
difference: we measured the honest (out-of-fold) generalization before deciding,
and rejected the technique when it failed; `TimeForge_2.0`'s repo reports the
in-sample number as the result, with no held-out check visible in the code.

---

## What the *legitimate* teams do that overlaps with (or extends) our approach

- **Regime-matched training data** (`codewithRahul01`): training GEO only on the
  1-2 days whose sampling cadence matches the test day, rather than all 7 days.
  This is the same insight as our GEO burst-window finding (Phase 6) — burst-mode
  and calm-mode are different statistical regimes — but they act on it by
  *filtering which days to train on*, where we quantified it as a diagnostic. This
  is worth trying on our system as a genuine, honest lever we have not yet tested.
- **Physics-informed features** (`gankit-aiml`): Beta angle / eclipse-intensity
  features for GEO. We do not model orbital physics directly (only
  time/harmonic features) — this is a legitimate idea we don't currently have.
- **Winsorization + TimeSeriesSplit** (`codewithRahul01`), **Huber loss +
  EarlyStopping** (`Tamaghna1`): both equivalent in spirit to our robust/Huber
  fitting and multi-fold validation — mutually confirms the robust-fitting
  approach is the sound default, independently arrived at by multiple teams.
- **None of the legitimate teams reported beating the 0.9810 benchmark on GEO**
  (best reported: `codewithRahul01`'s 0.7865 grand-average, `TimeForge`'s
  legitimate/non-gamed baseline never shown). This corroborates our own
  conclusion that GEO's outlier bursts are a genuine, shared, unsolved ceiling
  across every team who approached it honestly.

## Bottom line
Most teams mitigate overfitting with standard, legitimate tools (dropout,
winsorization, robust losses, time-aware splits, regularization) — consistent
with what we already do. **Exactly one team's repo contains code that games the
Shapiro-Wilk score by using the ground truth to reshape predictions**, and the
same repo's own output files show it does so at a severe, sometimes catastrophic,
cost to actual accuracy — for GEO and MEO1 it produces R² scores far worse than a
naive mean predictor. We correctly avoided this pattern in our own build (and
explicitly tested and rejected the milder version of it).
