# Validation: EKF+ODE integrator, and Transfer-Learning pre-training

**Verdict up front:** Both proposals are built on one core misunderstanding of
what the data is. Neither, as described, will produce a tangible improvement in
the *scored* metric (residual normality). One (EKF+ODE) is the wrong tool for
this task entirely; the other (transfer learning) is partly implementable but
cannot clear the ceiling that actually limits us, and we already have direct
measured evidence that added model capacity does not help here. I'll show the
reasoning, then list what *would* legitimately help.

---

## 0. The core misconception both ideas share

> "predict a satellite orbit 24 hours into the future" / "learn the laws of physics"

**We are not predicting a satellite orbit. We are predicting the *error* in
someone else's already-physics-based prediction.**

The columns are `x_error, y_error, z_error, satclockerror`. Per ISRO's own
`SIH_Data_Discription.pdf`, these are the *"error build-up between uploaded and
modelled values"* — i.e. `broadcast/modelled ephemeris − precise/true ephemeris`.
The deterministic orbital physics has **already been removed**; what remains is the
*residual* the physics model could not capture.

That residual is dominated by:
1. **Ground-segment upload events** — when the control segment uploads a fresh
   ephemeris/clock correction, the error jumps discontinuously. These are
   *operational human/scheduling decisions*, not orbital dynamics. (This is
   exactly the ±35–58 m GEO day-8 spikes we saw, and another team,
   `codewithRahul01`, independently labelled them *"operationally driven
   ground-segment decisions, not predictable from orbital mechanics."*)
2. **Clock stochastic noise** — atomic-clock random-walk / flicker processes.
3. **Small unmodelled forces** — the leftover after the model, near-zero-mean.

There is no hidden orbital ODE inside the *error* signal to integrate. If there
were, the ground segment's own model would have integrated it out already — that's
what produced the "modelled value" in the first place.

---

## 1. Extended Kalman Filter + Numerical ODE Integrator

**What EKF+ODE is genuinely for:** orbit *determination and propagation* —
given position/velocity state and a force model (gravity, drag, SRP), integrate
the equations of motion to predict where the satellite *will be*. It is excellent
at that, and that is a real, near-perfect-accuracy technology.

**Why it does not apply here:**

1. **We have no orbital state.** EKF/ODE needs a state vector (position, velocity)
   and a dynamics model. We have four *scalar error residuals* per timestamp — not
   a state, not a trajectory. There is nothing to numerically integrate.
2. **The target is the residual, which is not ODE-governed.** Even if we
   reconstructed an orbit, propagating it forward gives us a *better orbit*, not
   the *error of the broadcast orbit*. The error is the gap between two models plus
   exogenous upload jumps — an ODE integrator has no term for "when will the ground
   station next upload."
3. **"Near-perfect 24 h accuracy" is impossible for this signal**, because
   broadcast ephemeris validity is only ~2–4 h; the error is deliberately reset by
   uploads several times a day. No propagator predicts the reset schedule.
4. **Accuracy is not even the scored metric** (see §3).

**The one legitimate grain:** a Kalman filter *is* a standard, sound model for the
**clock** channel specifically — a 2–3 state clock model (bias, drift, drift-rate)
with process noise tuned to the clock's Allan variance. We already ship a
local-linear-trend Kalman (`kalman_llt`); a physically-motivated clock-state
version is a *modest, honest* refinement for one of four channels. It will not
touch the GEO position spikes that are our actual failure.

**Verdict on EKF+ODE: not implementable against this data as framed, and
conceptually mismatched. Do not build it.** (A clock-specific Kalman refinement is
fine but small.)

---

## 2. Transfer learning: pre-train on CDDIS, fine-tune on 143 rows

This is more serious and deserves a real evaluation on three axes.

### (a) Is it implementable? Partly.
- **The data pipeline is real** (I outlined it in `DATASET_SEARCH.md`): pull
  broadcast RINEX + precise SP3/CLK from CDDIS/IGS, difference them, and you get
  thousands of rows of the *same kind* of error signal. Feasible, multi-day work.
- **The fatal gap: GEO.** Precise reference ephemerides for GEO/IGSO satellites
  are sparse-to-nonexistent publicly (documented, and the reason our GEO series is
  the hard one). So you can build a large **MEO** (GPS/Galileo/BeiDou) error
  corpus, but **you cannot build a large GEO corpus** — meaning the transfer target
  that needs help most is the one you can't pre-train for. You'd be transferring
  from other constellations' MEO to ISRO's GEO: a large domain shift on the
  precise case we most need to fix.

### (b) Would pre-training "learn the laws of physics"? No — and it doesn't need to.
- There is no physics left in the residual to learn (§0). What a model *can* learn
  from a large error corpus is the **statistical structure** of GNSS errors:
  orbital-harmonic periodicity, autocorrelation scales, noise colour, upload-cycle
  patterns. That is legitimate.
- **But that systematic structure is low-dimensional and we already capture it.**
  The predictable part of our series is essentially *trend + a couple of orbital
  harmonics* — a ~6–14 parameter fit that is well-determined on 143 rows. After we
  remove it, the residual is already close to white/heavy-tailed. A "massive
  Transformer/Neural ODE" has nothing deep left to extract; the bottleneck is not
  model capacity.

### (c) Would it improve the *scored* metric? Almost certainly not.
This is the decisive point.
- The metric is **residual normality** (Shapiro-Francia W), whose ceiling is set by
  the **irreducible** part of the truth: the upload spikes and heavy-tailed noise.
  A perfectly pre-trained model still produces `residual = its_prediction − truth`,
  and the truth's spikes/outliers land in that residual regardless of how good the
  model is. **No pre-training removes an outlier that lives in the answer key.**
- We have **direct measured evidence** this is capacity-independent:
  - Our Phase-8 study fine-tuned on day-8 itself (the strongest possible
    adaptation) via honest out-of-fold and got **no gain — slightly negative**
    (GEO −0.002, MEO-1 −0.009, MEO-2 −0.018).
  - Simple robust models already match or beat flexible ones on rolling
    validation; the MLP only "wins" by widening variance, not by predicting
    better.
  - `codewithRahul01`, an independent team, also concluded the GEO spikes are
    unpredictable from history.
- So the expected effect of a pre-trained Transformer on Priority-1 is ~0 on MEO
  (already near ceiling / already passes MEO-1) and ~0 on GEO (spikes are
  exogenous). It might modestly improve *point accuracy* on MEO — but that is
  **Priority 2 (tiebreak only)**, not the headline score.

### (d) Cost/risk
Weeks of data engineering + training infrastructure + a massive model fine-tuned on
143 rows (extreme overfitting surface), to chase a metric whose ceiling we've
already shown is set by exogenous events. High cost, high risk, ~zero expected
Priority-1 gain.

**Verdict on transfer learning: implementable in part, but not favourable. It
attacks a capacity problem we do not have, cannot build data for the case that
needs it (GEO), and cannot cross the noise/upload ceiling that actually caps the
score. Not worth building for this competition.**

---

## 3. The point that reframes everything

> "predicts ... with near-perfect accuracy"

The competition does **not** reward accuracy. Priority 1 is the *normality* of the
residual; Priority 2 (mean/std) is only a tiebreak. A model can be near-perfectly
accurate and still score badly if the truth's residual is non-Gaussian, and can be
inaccurate-but-Gaussian and score well. Every "make it more accurate with bigger
physics/AI" plan optimizes the wrong axis. The winning move is *capturing the
simple systematic signal and leaving clean Gaussian residuals* — which is what our
current robust harmonic / GP / ensemble already do, and which is why MEO-1 already
passes honestly.

---

## 4. What WOULD legitimately help (small, honest, worth trying)

Ranked by expected value, all buildable on the data we have:

1. **Regime-matched training** (borrowed from `codewithRahul01`): train GEO only on
   the days whose sampling cadence matches the query regime (burst vs calm),
   instead of all 7 days. We *diagnosed* this regime split in Phase 6 but haven't
   *acted* on it. Cheapest real lever; may lift GEO/MEO-2 a little. **Try first.**
2. **Physics-informed features for GEO** (borrowed from `gankit-aiml`): Beta angle,
   eclipse flags, sun position — deterministic ephemerides, cheap to compute, may
   explain some of GEO's structure the harmonic misses. Honest and additive.
3. **A proper clock-state Kalman** (the salvageable grain of idea #1): 2–3 state
   clock model with Allan-variance-tuned process noise, for the `satclockerror`
   channel only.
4. **Accept the ceiling and optimise Priority 2** where Priority 1 is capped
   (GEO): keep residual std tight so we win the tiebreak, since W is bounded there.

None of these are glamorous, but they are real, they're days not weeks, and they
target the axis that's actually scored. The EKF/ODE and transfer-learning plans are
not "the one non-trash solution left" — they're impressive-sounding solutions to a
problem this competition is not posing.
