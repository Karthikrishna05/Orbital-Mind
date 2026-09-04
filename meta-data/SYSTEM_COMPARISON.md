# OrbitalMind: Two Systems Compared — Mallhar03/OrbitalMind vs. this build

**Scope.** "Theirs" = the `Mallhar03/OrbitalMind` GitHub repo (main + the
`feature/inference-entrypoint-and-shaping` branch) plus its "Complete Solution
Report". "Ours" = the system in this workspace. Everything about *ours* was
executed and measured this session; facts about *theirs* come from reading its
code and the data — labelled **[code]**, **[data]**, or **[reported]** (their
report's own numbers, which I did not re-run).

---

## 0. Executive summary

- **Both teams independently found the single most important insight**: the
  scorer is **Shapiro-Francia**, not `scipy.stats.shapiro` (which returns 0.9852
  vs the required 0.9810 on the reference). Both implement it by hand. This is
  the crux of the whole task and both got it right. **[code]** for theirs,
  **[measured]** for ours.
- **The systems score comparably** on the real day-8 data; neither can make GEO
  Gaussian, both pass the clean MEO series. The differences are in **data
  correctness, breadth of modelling, validation rigour, scorer fidelity, and
  software/process maturity** — not in a large accuracy gap.
- **The biggest hard-fact divergence:** their pipeline treats an **exact-duplicate
  data block as a second satellite**; ours de-duplicates it. I verified the block
  is 100 % identical **[data]**, so on this point ours is correct and theirs
  double-counts.
- **Where theirs is clearly ahead:** engineering discipline — an ADR decision log,
  frozen contracts, provenance tracking, a real NASA-CDDIS data-fetch path,
  predictive-interval output, and a formalised anti-fraud test culture.
- **Where ours is clearly ahead:** data de-duplication, a wider and more
  systematically-compared model set (incl. data-driven Lomb-Scargle periods,
  robust/Huber fitting, SARIMA, per-channel selection), multi-fold validation,
  an honest fine-tuning (out-of-fold) analysis, a kurtosis-branching scorer, and
  a lean codebase with no superseded pipeline, no broken tests, no doc drift.

---

## 1. The one hard data fact both systems must reconcile

Each MEO file contains a block that repeats after a single backward time-jump.
**Verified this session [data]:**

| File | rows | block1 | block2 | block2 rows that are EXACT copies of block1 | block2 rows with *different* values |
|---|---|---|---|---|---|
| `DATA_MEO_Train.csv` | 90 | 46 | 44 | **44 / 44** | **0** |
| `DATA_MEO_Train2.csv` | 244 | 143 | 101 | **101 / 101** | **0** |

Two genuinely different satellites cannot have byte-identical float values on
shared timestamps. This is a **copy-paste duplication**, not a second satellite.

- **Theirs [code]:** `split_stacked_blocks()` cuts at the backward jump and emits
  the duplicate block as a separate series (`MEO_Train-b1`, `MEO_Train2-b1`),
  explicitly *without* de-duplicating. Result: "5 series", 2 of which are
  duplicates. Consequence visible in their own report: `MEO_Train2-b0` **fails**
  (W=0.814) while its duplicate `MEO_Train2-b1` **passes** (W=0.948) — the same
  satellite lands on both sides of the pass line purely because the duplicated
  test rows split differently. That inflates the "3 / 5 pass" headline.
- **Ours [measured]:** the loader drops exact-duplicate rows (MEO1 90→46,
  MEO2 244→143) and scores **3 unique series**. No double-counting.

**Verdict: ours is correct here.** (Caveat of fairness: if the organiser's hidden
data intentionally stacks two *distinct* satellites, their splitter is the right
shape — but the supplied files do not; the blocks are identical.)

---

## 2. Architecture, head to head

| Dimension | Theirs | Ours |
|---|---|---|
| **Scorer** | Shapiro-Francia only (Blom + Royston-1993) **[code]** | Shapiro-Francia **+** Royston Shapiro-Wilk **+** `swtest` kurtosis-branch dispatcher (matches MATLAB) **[measured]** |
| **Own-SW requirement (Note)** | Met | Met |
| **Benchmark reproduced** | W=0.9814/p=0.5838 **[code/reported]** | W=0.9814/p=0.5838 **[measured]** |
| **Data de-duplication** | No (splits duplicate as satellite) | Yes |
| **Irregular timestamps** | Native (predict at supplied times) | Native |
| **Models for the deliverable** | Harmonic (OLS, 12h+24h, 3 harmonics), GP (ExpSineSq+Matérn+White), DeepResidual (harmonic + zero-init MLP), Persistence baseline | Baselines, Harmonic OLS **+ robust/Huber**, **auto-harmonic (Lomb-Scargle periods)**, GP, GradientBoosting, MLP, **Kalman**, **SARIMA**, ensemble (median), stacking (residual + learned NNLS meta), **per-channel selector** |
| **Period choice** | Hardcoded 12h/24h **[code]** | Hardcoded **and** data-driven (Lomb-Scargle) |
| **Robustness to outliers** | OLS (not robust); GP/Deep add some | Huber/robust fitting, MAD outlier tooling |
| **Model selection** | Leak-free single 75/25 chronological split **[code]** | Leak-free **rolling 3-fold** validation + **std & bias guards** |
| **Validation of "no peeking"** | Time-split ranking | Rolling folds; day-8 held strictly out; spread-gamer flagging |
| **Fine-tuning study** | Not explored | **Out-of-fold day-8 fine-tuning** quantified (no honest gain) |
| **Predictive intervals** | Yes — ±1.96σ columns in submission **[code]** | Bootstrap CI on W; GP std available but not emitted per-row |
| **Q-Q (Priority 3)** | Yes | Yes |
| **Confidence interval (Note)** | Calibration σ | Bootstrap percentile CI on W |
| **Legacy baggage** | Large superseded neural pipeline (LSTM/TCN-LSTM/TFT/NeuralODE/diffusion/normalizing-flow/LightGBM) still present | None |
| **Software scaffolding** | ADR log, contracts doc, provenance JSON, CDDIS fetch, device/paths abstraction, PR history | Lean, single-purpose; README + PROGRESS log |

---

## 3. Scorer fidelity — a subtle but real difference

Both reproduce the benchmark (kurtosis of the 45-sample reference is 3.33, so the
Shapiro-Francia branch fires). The difference shows up on **other** residuals:

- **MATLAB's `swtest`** (the most likely origin of the organiser's benchmark)
  **branches on kurtosis**: >3 → Shapiro-Francia, ≤3 → Shapiro-Wilk.
- **Theirs** always applies Shapiro-Francia. For a residual with kurtosis < 3 it
  would diverge from a branching evaluator.
- **Ours** replicates the branch (`swtest`) *and* keeps pure SW and pure SF
  available, so we match whichever the organiser used and can report all three.

If the evaluator is pure-SF, both are identical. If it is the branching `swtest`,
ours is the faithful one. **Edge: ours, conditional on the evaluator.**

---

## 4. Measured results (best honest, leak-free, per unique series)

Not perfectly apples-to-apples (different splits/selection), so read directionally.

| Unique series | Theirs (leak-free) **[reported]** | Ours (train-only leak-free) **[measured]** |
|---|---|---|
| GEO | W≈0.787, H=1 (FAIL) | W≈0.898, H=1 (FAIL) — pick=mlp |
| MEO_Train (=MEO1) | W≈0.915, H=0 (PASS) | W≈0.934, H=0 (PASS) — ensemble |
| MEO_Train2 (=MEO2) | W≈0.814, H=1 (FAIL) | W≈0.872, H≈0.5 (PARTIAL) — stack |
| **Mean (3 unique)** | **≈0.839** | **≈0.901** |

**Important honesty caveat on GEO:** our headline 0.898 comes from `mlp`, which
raises W partly by widening the residual (we flagged this as "spread-gaming"). Our
own *defensible, std-guarded* pick on GEO is closer to ~0.79 — i.e. **on GEO the
two systems are effectively tied**, both failing normality because the day-8 truth
diverges (±58 m outlier bursts nothing trained on the calm week can predict). On
the clean MEO series ours is marginally higher, and MEO2 is partial-vs-fail in our
favour. Net: **ours edges the raw scores, but the gap on GEO is not real skill.**
Both systems correctly conclude GEO is unfixable.

---

## 5. Where THEIRS is better

1. **Engineering & process maturity.** A numbered ADR **decision log**
   (`DECISIONS.md`), a **frozen-contracts** doc, **provenance JSON**, a device/
   paths abstraction, and an actual PR history. Ours is leaner but less formal.
2. **Real data-acquisition path.** `fetch_data.py` pulls live multi-GNSS orbit/
   clock products from **NASA CDDIS** with a synthetic-fallback safety rail. Ours
   only consumes the supplied CSVs.
3. **Predictive intervals as a first-class output.** Their submission emits
   ±1.96σ bounds per prediction. Ours computes CI on the *metric* but does not
   surface per-row intervals (easy to add; theirs already ships it).
4. **Formalised anti-fraud culture.** Explicit tests pin that the point forecast
   is bit-identical after calibration and that residual transforms cannot inflate
   p (a real past incident of manufactured p=0.9999 is on record and guarded).
   Ours enforces integrity through leak-free selection and guards, but with fewer
   codified invariants.
5. **DeepResidual design.** Harmonic base + zero-init neural residual head is a
   clean small-data idea (reverts to harmonic if no signal). Ours has an analogous
   stacking wrapper, but theirs is a tidy single model.

## 6. Where OURS is better

1. **Data correctness (de-duplication).** We drop the exact-duplicate MEO block;
   they score it as a second satellite (§1). Most concrete correctness win.
2. **Breadth + systematic search.** ~15 models incl. **data-driven Lomb-Scargle
   periods**, **robust Huber** harmonic, GBR, **SARIMA**, ensemble, learned
   stacking, and a **per-channel selector**, all ranked on the real metric.
   Theirs fields three for the deliverable.
3. **Validation rigour.** **Rolling multi-fold** validation with **std + bias
   guards** vs their single 75/25 split — we showed single small holdouts are
   noisy and pick unstable models; multi-fold fixes that.
4. **Honest fine-tuning analysis.** We ran **out-of-fold** day-8 fine-tuning and
   proved it yields no honest gain (and can hurt). Theirs does not explore it.
5. **Scorer completeness.** Kurtosis-branching `swtest` + pure SW (matches scipy
   to 1e-10) + SF; theirs is SF-only (§3).
6. **Codebase hygiene.** No superseded pipeline, no broken tests, no doc drift
   (see §7 for theirs). 33 passing tests, all relevant to the deliverable.
7. **Robustness to outliers in fitting** (Huber/MAD) — theirs leans on OLS.

## 7. Where they are roughly EQUAL

- The decisive Shapiro-Francia insight (both correct).
- Leak-free selection principle (both refuse to pick on the answer).
- Arbitrary/irregular-timestamp prediction (both native).
- Own-implemented scorer per the Note (both compliant).
- Honest acknowledgement that GEO can't be made Gaussian (both).
- Broadly similar headline scores on the unique series.

---

## 8. Document & repo consistency audit (their repo)

Requested second task — inconsistent / irrelevant / stale artifacts found:

1. **`README.md` (main) — stale/inconsistent.** Describes the **old CDDIS real-
   data + 4-neural-model** pipeline and nanosecond performance numbers. **No
   mention** of PS-08, Shapiro-Francia, or the `predict.py` entrypoint that is the
   actual deliverable. Points newcomers at the superseded `run_pipeline.py`.
2. **`ARCHITECTURE.md` (main) — stale.** Documents the **fixed 96-step** neural
   pipeline; silent on `predict.py`/`shaping.py`/arbitrary timestamps/Shapiro-
   Francia. Even notes internal contradictions (MSE loss vs claimed Gaussian
   likelihood; orphaned `features/` and `diffusion.py` "never imported").
3. **`docs/DECISIONS.md` — branch-only + self-contradiction.** Exists **only on
   the feature branch**, not on `main`, yet the report and other docs cite it.
   **Decision 004 rejects Gaussian Process modelling**, but `gp_predictor.py`
   ships **and is the selected model for two series** — the rejection is never
   reconciled for the small-data regime. Multiple "deck correction" entries (011
   GPyTorch nowhere, 012 EMD-not-EWT, 016 FFT constant features, 017 diffusion
   dropped, 018 C-05) show the judge-facing deck asserted things the code does not
   do.
4. **`latest.pdf` (repo root) — irrelevant/inconsistent binary.** A stray slide
   deck committed at root. Given the "deck correction" decisions above, it likely
   still contains claims (GPyTorch, diffusion-as-ensemble-member, 96 lag features,
   Gaussian-likelihood loss) that the code contradicts. Binaries like this don't
   belong tracked at root and are stale relative to the code.
5. **`tests/test_score_day8.py` — broken.** Imports a module `score_day8` from
   `scripts/`, but **`scripts/score_day8.py` does not exist** (tree has only
   ablation/fetch_data/generate_submission/train_and_rank). Fails at collection.
6. **`tests/test_causal_preprocess.py` — broken [reported].** Imports `MIN_LIMIT`
   from `preprocessing.pipeline`, which is undefined. Fails at collection.
7. **Large orphaned/superseded code** relative to the PS-08 deliverable:
   `models/{lstm,tcn_lstm,tft,neural_ode,diffusion,normalizing_flow}.py`,
   `ensemble/lightgbm_meta.py`, and all of `features/` — built and tested but not
   used by `predict.py`. Dead weight (and CI-less, so silent rot risk).
8. **Process smell (not a document, but relevant).** `main` has **no branch
   protection** and PRs #1–#3 are "closed, not merged" yet their commits are on
   `main` — i.e. direct pushes. This is how the stale docs and the (separately
   removed) duplicate project folder got in unreviewed.

*(Legit and consistent: `data/` competition files incl. `Note.pdf`,
`SIH_Data_Discription.pdf` — the latter's filename is misspelled "Discription"
but it is the organiser's file; `docs/CONTRACTS.md`; the PR template.)*

---

## 9. Bottom line

The two systems agree on the hard part (Shapiro-Francia, leak-free selection,
GEO's irreducibility) and land at similar scores. **Ours is the more correct and
more thoroughly-searched *modelling* effort** (de-dup, robust fitting, data-driven
periods, multi-fold validation, wider model set, honest fine-tuning study, richer
scorer). **Theirs is the more mature *software/process* effort** (decision log,
contracts, provenance, real data fetch, predictive intervals, anti-fraud tests) —
but it carries stale docs, two broken tests, a superseded pipeline, and a
data-duplication misread that our approach avoids. The strongest possible system
would merge ours' data hygiene, model breadth, multi-fold validation and
kurtosis-branch scorer into theirs' engineering scaffolding and interval output.
