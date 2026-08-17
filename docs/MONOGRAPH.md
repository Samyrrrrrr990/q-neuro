# Q-Neuro: A Monograph

### The complete history of a research program that falsified itself, era after era

**Scope: synthetic, analytic, nonclinical computational research only.** Nothing here has been
evaluated on patients. Nothing here is a medical device, a diagnostic system, a clinical
decision-support tool, or a demonstration of physical quantum computation.

---

## How to read this

Every substantive idea is presented at three levels:

> **Intuition** — what the idea feels like, in words a colleague would use at a whiteboard.
> **Formal** — the precise statement, with its derivation.
> **Experimental** — what happened when we tested it, including when the answer was no.

Failures are not appendix material here. They are the spine. Of the ideas in this document, the
overwhelming majority are dead, and several of the dead ones were more beautiful than anything that
survived. Where that is true, it is said.

The three-part documentation set: `docs/PAPER.md` is the short version, `docs/TECHNICAL_BREAKDOWN.md`
is the reproduction manual, and this is the narrative — including the parts the other two omit.

**Table of contents**

**The four eras.** Helix (the complex-valued hypothesis, overturned by exact controls) → Sentinel
(the falsification and equivalence framework) → Pulse (one adaptive-compute result that survived,
with its boundaries mapped) → **Nova** (a clean-slate architecture search that returned no).

- Part I — Q-Neuro 1.0 / Helix: the complex-valued hypothesis
- Part II — The falsification, and the falsification of the falsification
- Part III — Q-Neuro 2.0: the equivalence compiler
- Part IV — The discovery lane: eleven attempts, eleven closures
- Part V — Q-Neuro 3.0: architecture from first principles
- Part VI — Cycle 2: the fix, and thirteen predictions later
- Part VII — The ceiling moves: a runtime, and a second boundary
- Part VIII — The final phase: real data, strong baselines, three dead branches
- Part IX — Nova: the search for a new computational principle
- Part X — What the program actually produced
- Part XI — The beautiful results that had to be killed
- Appendix A — The failure register
- Appendix B — Rules that earned their cost

---

# Part I — Q-Neuro 1.0: the complex-valued hypothesis

## I.1 The original idea

> **Intuition.** A real-valued network stores a number at each unit. A complex-valued one stores a
> number *and a phase*. Phase lets contributions interfere — reinforce or cancel — which is how
> waves carry structured information through noisy media. If a task requires composing evidence in
> an order-sensitive way, phase seems like exactly the right substrate, and a complex recurrent
> operator ought to be more robust than a real one of the same size.

> **Formal.** Model a hidden state `ψ ∈ ℂ^d` evolving under a learned structured operator
> `ψ ← U(x)ψ`, with a readout that is sensitive to relative phase. The claim under test: for
> matched parameters, the complex operator family attains higher top-1 accuracy under distribution
> shift than real families of the same budget.

> **Experimental.** In the original NeuroWorld simulator, it worked. `QN-000008`: the complex model
> beat every control across five unseen worlds and four severities, with world-level paired
> confidence intervals excluding zero, and **+0.054 to +0.063** over a two-channel real control.
> `QN-000016` isolated apparent mechanisms: ordered state-conditioned composition contributed
> **+0.232** shifted top-1 over a commutative accumulator, and phase-sensitive readout contributed
> **+0.104**.

That is a good-looking result. It is also, in retrospect, the most instructive thing in this
repository, because almost none of it survived contact with better controls.

## I.2 The cracks, in the order they appeared

Each of the following was found by the program itself, and each is recorded in `docs/CLAIMS.md`
with its status.

**Sample efficiency was a comparator artifact.** `QN-000004` suggested operator states were the
most sample-efficient mechanism. `QN-000006` tuned the GRU properly, and the GRU reached **0.920**
at 250 cases against 0.774 real and 0.699 complex. Status: *refuted under tested setup.* The
original claim had been measured against an under-tuned baseline.

**Calibration transfer was false for everyone.** In-domain temperature scaling, fitted on
validation, **worsened** shifted ECE for every model and sometimes severely worsened NLL. Not a
complex-specific finding, but it removed one of the reasons to like the complex model.

**Ambiguity representation was backwards.** The story was that complex hypothesis states represent
irreducible ambiguity better. `QN-000010`: complex pair NLL **2.581** against 1.148 real and 1.453
two-channel, with valid-twin mass only 0.212. Refuted.

**The representation claim inverted.** `QN-000019` probed for hierarchical structure and found it —
mechanism, localization, temporality, context recovered at 0.932/0.933/0.907/0.918. But GRU and
state-space probes were *generally stronger*, and complex-minus-GRU was negative on every factor.
Refuted.

**The interpretability correlation was decorative.** Probe accuracy correlated with shift robustness
at `r = +0.45` across models — a descriptive correlation over 18 non-independent architectures,
where the strongest probe state (GRU) was among the weakest under shift. Not supported.

By this point the complex hypothesis was surviving on one leg: robustness under simulator shift.

## I.3 The decisive control

> **Intuition.** Before asking whether complex arithmetic *helps*, ask whether the complex model is
> even doing something a real model cannot do. Any complex computation can be written in real
> arithmetic by splitting into real and imaginary parts. So: build the exact real implementation of
> the same computation, and compare.

> **Formal.** Let `ℛ: ℂ^d → ℝ^{2d}` be realification. For a complex linear map `M = A + iB`, the
> realified operator is the block matrix `[[A, −B], [B, A]]`. Any complex network has an exact real
> counterpart computing the identical function. Call it `exact_real_block_operator`.

> **Experimental.** `QN-000042`: top-1 **matches in all 1,920 held-out cells.** Not approximately —
> matches. And the wider discovery run `QN-000040` found **0 of 2,880** positive effects for
> complex against the best-real envelope, mean **−0.03695**. The held-out family/world/seed
> hierarchical bootstrap interval for complex-minus-best-real: **[−0.01325, −0.00457]** — entirely
> below zero.

The complex advantage was gone. Recorded as `FAIL-005`.

## I.4 The frozen law that also failed

> **Intuition.** Even if complex models are not better, maybe the *size* of the shift-robustness
> gap follows a law you could predict from task and model properties.

> **Formal.** `QN-LAW-001`, fitted on discovery families and frozen with thresholds on held-out R²
> and MAE, before the confirmation families were opened.

> **Experimental.** Held-out **R² = −30.94**, MAE 0.03126. Both thresholds failed. The sign
> transferred; the magnitude did not. No refit, no threshold change. Recorded as `FAIL-003`.

And the grand benchmark, `QN-GRAND-001`, never ran: six mandatory readiness gates failed at
preflight and the sealed benchmark stayed sealed (`FAIL-004`). It is still sealed.

---

# Part II — The falsification, and the falsification of the falsification

## II.1 The uncomfortable audit

The negative result in Part I was clean, published, and — it turned out — partly built on the same
mistake it was correcting.

`docs/QE_AUDIT_MEMO_001.md` asked eight questions about the exact-real control. The answer to the
central one was uncomfortable:

> **The parameter map between the complex model and its "exact real" control is the identity on
> shared coordinates.**

> **Intuition.** We had built the real control by *reinterpreting the same numbers*, not by building
> a genuinely separate model. Of course the outputs matched — there was only ever one model, wearing
> two labels.

> **Formal.** A pair is **transport-degenerate** when the parameter map `T` is the identity on
> shared coordinates. Then every transport level `T0`…`T5` is vacuously satisfied, and the pair
> carries no information about how equivalence behaves under training.

> **Experimental.** The measured forward discrepancy between the two was **5.245e-06** — nonzero,
> but attributable entirely to numerical implementation, not to any structural difference.

## II.2 What this cost, stated plainly

The 1,920 held-out cells were not 1,920 independent architecture comparisons:

| Best-real winner | Cells | Mean effect | Exactly zero |
|---|---:|---:|---:|
| `exact_real_block_operator` | 1,478 (77%) | +0.00000 | **1,478 (100%)** |
| `real_polar_operator` | 442 (23%) | −0.03978 | 60 (14%) |

Those 1,478 exact zeros are **equivalence-induced**. They are the same model twice, and presenting
them as independent wins would be wrong.

## II.3 Why the negative result survived anyway

> **Intuition.** The complex hypothesis had to beat two different things: an exact re-encoding of
> itself, and genuinely different real architectures. The first comparison was vacuous. The second
> was not, and it is the one that matters.

> **Experimental.** The informative slice is `QN-000040`, where genuinely distinct real models win
> **52%** of discovery cells: `state_space` (637 cells, −0.0906), `real_polar_operator` (485,
> −0.0469), `gru` (367, −0.0707).

So the falsification stands, on stronger ground and for a better reason. The complex-arithmetic
claim fails on two independent grounds: exact realification reproduces the implemented computation,
**and** genuinely distinct real controls outperform it.

The history was **not rewritten**. `docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md` preserves the
superseded wording in §1 and records the downgrade in §3. The rule that produced that choice is in
Appendix B.

## II.4 The lesson that started Q-Neuro 2.0

> The comparison you are making is only as good as your ability to say precisely **what is being
> compared**. "The real version of this model" is not a specification. It is four unstated choices.

Which is where the whole second phase came from.

---

# Part III — Q-Neuro 2.0: the equivalence compiler

## III.1 The reframing

> **Intuition.** Stop treating "equivalent" as a yes/no property of two models. Treat it as a
> *typed claim about a map*, which either can or cannot be certified. Make the certificate a real
> object in the code, and make it refuse.

> **Formal.** An equivalence claim is a tuple: a **map**, an **equivalence level** (E0 symbolic
> through E4 metric-only), a **domain** (with exclusions), and a **transport level** (T0 parameters
> only through T5 including weight decay). Certificates may be downgraded, never upgraded.

> **Experimental.** Three refusals are enforced at construction or call time, not in prose:
> declaring E0/E1 with a domain restriction raises; upgrading a certificate raises; a map that
> cannot transport gradients raises `NotImplementedError` rather than approximating.

This sounds like software engineering rather than science. It is both, and the engineering is what
made the science checkable.

## III.2 The transport-covariance conjecture

> **Intuition.** Two equivalent models drift apart under training because the optimizer does not
> commute with the map relating them. If you could *measure* that non-commutation — a "defect" —
> you should be able to predict how far apart they end up.

> **Formal.** For a map `T`, an optimizer step `U`, and a state `θ`, define a defect measuring the
> failure of `T ∘ U = U ∘ T`. **Conjecture:** an accumulated defect statistic predicts final
> predictive divergence across equivalence families, with a single calibration.

> **Experimental.** *False.* This is Gate D, §III.5.

The name was chosen deliberately: **`transport_covariance_conjecture`**, no eponym, so that killing
it would cost nobody's name.

## III.3 The map families, and what each taught

**Permutation.** Discrepancy **1.192e-07** — one float32 ULP from second-layer reduction order.
Nothing deeper. The lesson is that a family can be *conjugate*, meaning there is nothing left to
predict, which matters enormously later.

**Scaling orbit.**

> **Formal.** Under uniform scale `s`, gradients scale as `s^{-1}`. Optimizer state scales by the
> gradient's power: first moments by 1, second moments by 2. The learning rate exponent is 2 for
> SGD and **1 for Adam**, because Adam's update is scale-free in the gradient.
>
> **Experimental.** With `η → η s²`, the SGD gradient step transports **exactly** — bitwise zero
> discrepancy, derived first and then confirmed.

**And then weight decay broke it, structurally.** The gradient step needs `η s²`; the decoupled
decay term needs `η s⁰`. No single learning-rate policy satisfies both. Discrepancies: 3.405e-03
(SGD), 1.312e-04 (AdamW). This is not a bug. It is a true statement about the pair.

**Dense ↔ factorized.** No transport exists. Factor descent preconditions the product: the gradient
with respect to `(U,V)` induces a different effective metric on `UV` than a dense gradient does.
The map sets `supports_optimizer_transport = False` and **refuses**. Being able to refuse is the
feature.

**Native complex.** A genuinely new model family with real `complex64` leaves — introduced
prospectively, never as a modification of historical evidence. AdamW transports **exactly zero**;
SGD leaves 1 ULP from complex kernel arithmetic.

**Complex ↔ exact-real.** Certified **E2 on a declared domain**, never E0 or E1, with the excluded
region `min_k |δ − i(2k+1)π/2| ≤ ρ_c` and a **measured** `ρ_c` of 1.55e-03 (float32) / 3.16e-08
(float64) — several times larger than the naive `sqrt(eps/2)` estimate. Reachability was probed
once: closest approach 1.425, a margin of 919×. One configuration only, and we say so.

## III.4 Gate C: the bound, and the real finding

> **Intuition.** If you bound how far two models can drift using Lipschitz constants, is the bound
> useful or is it astronomically loose?

> **Formal.** Compose per-step Lipschitz constants of the update maps to bound accumulated
> divergence. Gate C requires bound/observed ≤ 100.

> **Experimental.** 405 cells, 360 scored. **Zero violations.** Median ratio 2.90, worst **58.74**.
> Passed.

**But the finding is not that the inequality is good.** The same inequality, with the Lipschitz
constant obtained by the triangle inequality `‖I − ηH‖ ≤ ‖I‖ + η‖H‖`, gives a worst ratio of
**1.472e+259** — vacuous by more than 250 orders of magnitude. The triangle inequality discards
exactly the cancellation that makes gradient descent contractive.

> **The bound's usefulness is a property of how you obtain the constant, not of the bound.**

And the scope is narrow, stated at the time and not retrofitted: affine maps with *exactly computed*
constants are the most favorable possible setting. Gate C says nothing about nonlinear models.
Part IV turns that caveat into a measured failure.

## III.5 Gate D: the conjecture dies

> **Formal.** 216 transport traces over four families, scored by **leave-one-family-out** held-out
> R² on log₁₀ final predictive divergence. Pass requires a candidate to beat every baseline on ≥2
> families.

> **Experimental.**

| Feature | Kind | Mean held-out R² | Best within-family |
|---|---|---:|---|
| `amplified_defect` | candidate | −20.47 | 0.819 (scaling) |
| `cumulative_defect` | candidate | −31.71 | **0.962** (factorization) |
| `one_step_predictive_divergence` | baseline | −92.50 | 0.460 |
| `learning_rate` | baseline | −380.51 | **0.896** (factorization) |
| `mean_amplification` | baseline | −898.28 | 0.348 |

`cumulative_defect` beat every baseline on **one** family. The gate requires two. **FAIL.**

> **The failure mode is calibration, not absence of signal.** Within family, the candidate is the
> strongest feature available on both families with real spread. Across families it collapses,
> because family medians span about **6.5 orders of magnitude** — permutation near 1e-7 (conjugate,
> nothing to predict), scaling orbit near 1e-0.6. The ranges *chain* rather than clustering, so
> there is no single gap to bridge; a global slope and intercept is simply the wrong object.

Two honesty notes that cost us something:

**A candidate was disqualified mid-analysis, and it was ours.** `one_step_defect` was listed as a
candidate. With a mapped initialization `e₀ = 0`, the first re-coupled step's defect is bit-for-bit
the first step's predictive divergence — **identical in all 216 rows**. It was a baseline wearing a
candidate's label. Removed, reason recorded in source, and no features were added after the outcome
was seen.

**A kill condition is partially live.** `learning_rate` *alone* reaches R² 0.896 within
factorization and 0.509 within native-complex, beating both candidates there. Any revival of this
conjecture must show added value *after* controlling for the learning rate. That is written into the
open obligations.

**The gate is enforced in code.** `run_qe_000010.py` refused to freeze an estimator, which keeps
ladder rungs 5–8 sealed and blocks `QE-000012`. A gate you can talk your way past is not a gate.

## III.6 The outcome of Q-Neuro 2.0

Of the seven pre-declared outcomes A–G, the program landed on **Outcome A**: the transport law
failed, and the Equivalence Compiler is the durable result.

That is a real deliverable. It is also much smaller than what we set out to build, and this document
does not inflate it.

---

# Part IV — The discovery lane: eleven attempts, eleven closures

Lane B exists so that speculative work can run without contaminating confirmatory evidence.
Nothing in this part may be cited by any claim. Everything in it is preserved.

## IV.1 DISCOVERY-001 — the phase boundary

This is the most beautiful thing the program found, and it is dead in the direction that mattered.

> **Intuition.** Take two models that are the *identical predictor*, written in different
> coordinates, and train both with identical data, optimizer and hyperparameters. One converges.
> The other blows up. Not because of noise — because of the coordinates.

> **Formal.** Under uniform scale `s` with an untransported learning rate, the target's update
> operator is `I − (η/s²)H`. Its effective step is `η/s²`, so it is stable exactly when
>
> ```
> ρ = η·λ_max(H) / (2s²) < 1
> ```
>
> The source is stable when `ρ s² < 1`. Therefore **for `s < 1` there is an open window in which a
> model converges and its exact equivalent diverges.** The control parameter is dimensionless and
> was derived before any sweep.

> **Experimental.** Across a **1.4%** change in `s`, paired divergence moves ~**14 orders of
> magnitude**, with the source stable throughout.
>
> | | SGD | AdamW |
> |---|---:|---:|
> | Cells | 1,476 | 1,476 |
> | Prediction accuracy | **0.9912** | 0.5041 |
> | False alarms (ρ ≤ 1 yet diverged) | **0** | — |
> | Misses away from ρ = 1 | **0** | — |
> | Misses exactly at ρ = 1 | 13 | — |
> | Diverged cells | 720 | **1** |
>
> All 13 disagreements sit at `ρ = 1.0` itself — marginal stability, where the spectral radius is
> exactly 1 and neither verdict is defined.

**The differential prediction is what makes this science rather than curve-fitting.** Adam's update
is scale-free in the gradient, so its effective step never acquires the `1/s²` factor and the
boundary should be *absent* at the same `ρ`. Measured: **1 divergent cell out of 1,476**, against
720 for SGD.

**The boring explanation, stated first and stated plainly.** This *is* textbook gradient-descent
stability. Reparameterization changes the effective Hessian, hence the effective step, hence
stability. Coordinate dependence of sharpness is established (Dinh et al. 2017; Kristiadi, Dangel
and Hennig 2023). No novelty is claimed for the mechanism. What it contributes is an exactly
predictable *location* for equivalence breaking — and a mechanistic account of the Gate D failure:
the discovery families straddle this boundary, so they are not one population.

### IV.1.1 And then it failed twice

Promotion requires a frozen prediction opened against an untouched nonlinear system. Two attempts
were made. Both are preserved.

**`DISCOVERY-001-P1` — VACUOUS.** The frozen grid used learning rates 0.05–0.2 against a measured
curvature range of 0.51–10.9, so `ρ` never reached 1.1. There were **197 chances to false-alarm and
zero chances to miss.** It printed `passes: True`, and that reading is meaningless.

This is worth sitting with. A frozen, hashed, prospective prediction *passed*, and the pass was
worthless, because the grid could not express the failure. Preserved, not re-tuned. P2 added a
non-vacuity guard requiring cells on both sides of the band.

**`DISCOVERY-001-P2` — FAILED.** With `η` placed per cell so `ρ` equalled its target by
construction, **96 of 96** cells at `ρ ≥ 1.1` converged. SGD divergence rate **0.0000** even at
`ρ = 3.0`, growth ratios 1.03–1.27 against a threshold of 2.0.

> **Why.** Exactly the mode frozen in P2's `anticipated_failure_modes` beforehand. `ρ` is computed
> from the Hessian at the mapped initialization, and a ReLU network under cross-entropy **changes
> the geometry it occupies** — it relocates to flatter regions and the loss saturates, so an
> initially over-large step does not compound.

**What survived:** `ρ < 1 ⟹ stable`, with **zero false alarms across 269 scored cells** over both
attempts. That half has never produced a counterexample.

**What died:** `ρ > 1 ⟹ diverges`, completely, outside quadratic objectives.

No third prediction was issued. Two attempts are consumed, and a third would require a materially
different estimator — which is a new discovery, not a rescue. Recorded as `FAIL-007`.

> **The generalizable lesson:** *a nonlinear learner cannot be understood from the geometry at
> initialization.* That single sentence redirected the entire rest of the program.

## IV.2 DISCOVERY-002 — canalized quotient dynamics

> **Intuition.** Borrow from developmental biology. An embryo reaches the same adult form from many
> perturbed starting points — Waddington's canalization. Maybe training does the same: perturbed
> learners fall into the same *functional* attractor. If so, a trajectory-aware measurement taken
> *early* should predict which attractor a run will reach.

> **Formal.** Define a phenotype map `Π: S → F` from full training state to predictors —
> **never from raw weights.** Declared representation: logits on a frozen audit batch. Declared
> metric: max-norm logit distance (primary), Jensen–Shannon divergence (secondary). Study dynamics
> on the quotient `S/∼` by the group of semantics-preserving transformations. Manifold language is
> avoided deliberately: these actions are non-free, have stabilizers, and produce singular strata.
>
> **Hypothesis:** a representation-invariant, trajectory-aware basin quantity measured before
> training finishes predicts final functional agreement better than initialization curvature,
> sharpness, learning rate, parameter distance, or loss statistics.

> **Experimental.** Falsifier 1 fired. **Basin coherence added nothing beyond early loss.**
> (`FAIL-008`.)

The preregistration listed thirteen falsifiers up front, precisely so this could not be argued
around. It also stated up front, in §8, that every ingredient was assumed old — Waddington
landscapes, cell-fate attractors, neural cellular automata, contraction analysis, transition-path
theory — and that **inspiration is not novelty**.

**The committor was untestable, not wrong.** `q(s) = Pr_s[τ_B < τ_A]` requires two distinguishable
functional attractors `A` and `B`. There weren't two. (`FAIL-009`.) You cannot measure the
probability of reaching one of two fates in a system with one fate.

## IV.3 DISCOVERY-003 — the hunt for real bifurcation

Since the committor needed two fates, the next step was to find a system that genuinely has them.

> **Experimental.** Five system families searched. **None found.** (`FAIL-010`.)

And the search produced two textbook self-inflicted wounds:

**The threshold artifact.** Counting crossings of an arbitrary 0.9 accuracy line suggested **6 of
16** systems were bifurcating. A proper max-gap/range bimodality check gave **0.20–0.47 — all
unimodal.** The "bifurcation" was the threshold, not the system.

**The pre-asymptotic exponent.** An apparent nonequilibrium scaling `rms(M) ~ T^0.625`, against the
equilibrium `0.5`. Beautiful, and false: `M/√T` was not constant (drifting 0.52 → 1.02), and long
runs were still non-stationary at `t = 64000`. We had fitted a power law inside a transient.
(`FAIL-011`.)

## IV.4 DISCOVERY-004 and 005 — geometric phase

> **Intuition.** Move a system around a closed loop in hyperparameter or curriculum space and
> return to the start. In physics, the state can come back *rotated* — a Berry phase. If training
> did this, curriculum order would leave a measurable geometric signature.

> **Experimental.** No phase. (`FAIL-012`, `FAIL-013`.)

**And the first measurement was broken in a way that would have produced a positive result.** The
endpoint metric `D_F(end, start)` was dominated by ordinary convergence drift: the *stay-control*
— a system that never traversed any loop — scored **4.19**, exceeding every actual loop. Replaced
with paired clockwise/counter-clockwise traversal at identical noise, which is the only comparison
that isolates orientation. Curriculum holonomy then failed all six discriminators under a fully
deterministic ordering.

## IV.5 DISCOVERY-006 through 010 — functional navigation

> **Intuition.** Among all parameter settings that fit the training data equally well, some
> generalize better out of distribution. If you could find directions that preserve training
> behavior while moving toward better OOD behavior, you could *navigate* to a better model for free.

> **Formal.** Find `v` with `J_train v ≈ 0` (training predictions preserved) and `J_OOD v ≠ 0`
> (something else changes). Move along `v`.

> **Experimental.** Seven consecutive failures. Then the explanation, which is the actual result.

**Closure 1 — the subspace was empty.**

> **Formal.** `rank(J_train) = min(n(C−1), P − g_arch)` because the gauge kernel is structural and
> shared by every distribution. So
>
> ```
> d_free = max(0, P − g_arch − n(C−1))
> ```

> **Experimental.** Every navigation search ran at `n = 600` with `P − g = 193`, giving
> **`d_free = 0` exactly.** There was nothing to find. The searches were **structurally doomed, not
> unlucky.**
>
> Stated before measurement, then confirmed exactly in 9 of 9 cells including the transition at
> `n = 193`:
>
> | n | 50 | 100 | 150 | 180 | 190 | **193** | 200 | 250 | 400 |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
> | predicted | 143 | 93 | 43 | 13 | 3 | **0** | 0 | 0 | 0 |
> | measured | 143 | 93 | 43 | 13 | 3 | **0** | 0 | 0 | 0 |

**Closure 2 — where a subspace existed, it was gauge.** Principal cosines between the measured free
subspace and the predicted softmax-gauge subspace: **1.0000000000 across all 25 directions.**
Traversing **472%** of `‖θ‖` along it changes the maximum logit by **4.26e-11**. It is not an
approximate symmetry. It is *the* symmetry, to ten decimal places.

**Closure 3 — the boring alternative won on every axis.** Joint training on the combined data beat
every navigation method simultaneously: `dL` 1.86e-2, test 0.9845, `dOOD` **+0.0758**, held-out
**+0.0423**. If the OOD gradient is available at all, the simplest possible use of it dominates
anything clever.

**And the novelty was dead before the performance comparison.** Gate 1 was prior art, run first by
design. The navigator's update is `(JᵀJ + λI)⁻¹g` — the proximal step of EWC with full curvature.
Cosine **0.66** with OGD, **0.23** with diagonal proximal-EWC. Marked dead on substance and retained
as a strong baseline. **No existing method was rebranded as Q-Neuro.** (`FAIL-016`.)

## IV.6 DISCOVERY-011 — three frozen attempts at an exact law

The dimension law explained everything. So we tried to freeze it as a law. Three attempts, three
failures, and the third one hurts.

**`DFREE-LAW-P1` — COMPROMISED.** Substance held **126/126** on the direct rank measurement. But
the measurement procedure was underspecified: a 400-sample gauge probe cannot saturate rank when
`P − g > 400(C−1)`. That produced exactly **21 mismatches = 3 configurations × 7 n-values** — a
diagnosis so clean it is essentially proof.

**And the freeze is still recorded as invalid.** Changing a measurement after observing failure
invalidates a freeze regardless of how good the diagnosis is. That rule cost us a result we could
have defended, and it is the right rule. (`FAIL-019`.)

**`DFREE-LAW-P2` — FAILED, 118/360.** The assumption `g = h_last + 1` universally is **false**;
`g` is activation-dependent.

> **Formal.** If `φ(cx) = cφ(x)` for `c > 0`, then `(W₁,W₂) → (cW₁, W₂/c)` is an exact
> function-preserving symmetry — one scaling per hidden unit, contributing `h` more directions. So
> tanh gives `h+1` and ReLU gives **`2h+1`**. Confirmed 8/8 in a follow-up diagnostic.

**`DFREE-LAW-P3` — FAILED, 2 of 48.** And this one contains a genuine prospective success buried
inside a failure: `leaky_relu` and `abs` were **derived as positively homogeneous before
measurement** and gave exactly `2h+1` in every cell. But two ELU cells at `h = 15` measured **17**
against a predicted **16** — off by exactly one, most plausibly a singular value sitting near the
`1e-9` rank tolerance. The frozen criterion was exactness. One mismatch fails. Two did.

> **Honest status:** an elementary and known relation, imperfectly confirmed under our own exactness
> standard. Useful as a diagnostic, not as a discovery. The form is right; the integer count is not
> controllable to the standard we froze.

## IV.7 What the discovery lane produced

Not a discovery. Three reusable controls, each of which would have saved months if we had them at
the start:

1. **Check `d_free` before searching for free directions.** If it is zero, stop.
2. **Check integrability before trusting a first-order subspace.** A direction that exists
   pointwise may not integrate into a path.
3. **Check whether the simplest use of the same information already dominates.** It usually does.

---

# Part V — Q-Neuro 3.0: architecture from first principles

## V.1 The setup

Q-Neuro 3.0 was a clean-slate architecture program, built hardware-first for one specific machine —
a fanless Apple M2 MacBook Air, 8.0 GiB unified memory — with Q-Neuro 2.0's machinery acting as an
immune system auditing every claim.

The hardware profile is **measured, not assumed**: 2.08 GiB available, 8 physical cores, 4 torch
threads, MPS with `complex64`, CPU↔MPS crossover at **65,536 elements**, working budget 1.04 GiB at
a deliberately conservative 50% of available memory. Sustained swapping on a fanless machine costs
more than a smaller model does.

## V.2 Cycle 1: does thinking longer, sometimes, beat thinking long always?

> **Intuition.** A fixed-depth model pays the worst case on every input. Easy examples get the same
> compute as hard ones. If a task's difficulty genuinely varies per example, a model that decides
> its own depth should win.

### V.2.1 Getting the task right took two tries

**The first task was unsolvable, and the models said so in unison.** All ten candidate models
returned **exactly 0.1441**. The target was ambiguous given the input — the model was never given
the information needed to disambiguate it. Ten architectures agreeing to four decimal places is not
a tie; it is a message about the task.

**The replacement.** `chase_to_goal`: a permutation defines a **single cycle** through 24 nodes; the
model starts somewhere on it and reports how many hops away node 0 is, up to 8. Difficulty is a
property of the datum. The answer is discoverable *only by following the chain*. Guessing gives
0.136.

### V.2.2 A bug that produced a perfectly plausible null result

Everything sat at or below the 0.136 guessing baseline. The cause:

> **Formal.** Chain following is an associative lookup — match the current node against its
> *identity* (a key), read its *successor* (a value). `Core.advance` used **one embedding for
> both**, which makes the lookup impossible.

Separate `key` and `value` embeddings, and depth-8 reached **1.0000 by epoch 2**.

Had we stopped one step earlier, we would have concluded that recurrent chain-following does not
work. It is worth naming how ordinary that near-miss was.

### V.2.3 The ladder

All variants share one `Core`, so any difference is attributable to the halting mechanism rather
than capacity.

| Model | Halting rule | Params | Accuracy | Steps |
|---|---|---:|---|---:|
| `Q0Fixed` | always 8 steps | 28,360 | 1.0000 | 8.00 |
| `Q1Elastic` | PonderNet-style mixture | 28,425 | 0.6241 | 3.27 |
| `Q2Commit` | hard commit, straight-through | 28,425 | 0.9999 | 8.00 |
| `Q3Arrival` | halt on detected arrival | **27,970** | 0.9994–1.0000 | **4.54** |
| `Q4Grounded` | Q3 + training-only grounding | 27,970 | 0.6322–0.9500 | 4.50–5.33 |

**Q1 collapsed.** Ponder collapse: the halt head shuts depth down before the core can use it. A
negative halt-bias initialization (`−5.0`, then `−2.0`) did not cure it — so collapse is not merely
an initialization artifact.

**Q2 isolated the cause.** Changing *only* the halting rule from a mixture to a hard
straight-through commit lifted the same core from 0.6241 to **0.9999**. The mixture was the defect,
not the halting. Single-variable intervention with the seed held fixed — this result stands.

**But Q2 bought nothing:** 0.9999 at full depth is just Q0 with extra machinery.

**Q3 was the idea.**

> **Formal.** Halt when arrival is *detected*, and let the halt step itself be the answer. Compute
> a first-arrival distribution in log space:
>
> ```
> log_not   = log1p(−p)
> cum       = [0, cumsum(log_not)[:−1]]
> log_first = log(p) + cum          # P(first firing at step k)
> ```
>
> trained by `−log_first[distance − 1]`.

> **Experimental.** **0.9995 accuracy at 4.53 average steps** against Q0's 1.0000 at 8.00 —
> **1.77× less inference compute**, with **fewer parameters** (27,970 vs 28,360) and 28.8 s of
> training.

That is a genuine, matched, favorable result. And it is not reportable.

## V.3 Attacking it

Two objections were written down before testing: (a) the task may be rigged because the answer
equals the step count; (b) the 1.77× may be nothing but `max/mean = 8/4.5 = 1.78`.

Objection (b) was frozen as a prediction — `QNEURO3-Q3-P1`, sha256 `a29900e8…`:

> "Q3's inference-compute saving is set entirely by the difficulty distribution, not by the
> architecture: saving = `max_depth / E[distance]`, and average steps = `E[distance] + ε` with
> `|ε| ≤ 0.25`."
>
> "If true, Q3 is an **optimal** allocator rather than a clever one, and the size of any advertised
> speedup is a property of the workload. Stating that in advance prevents quoting a 1.77× number as
> though the architecture produced it."

**Opened once against three untouched distributions. Failed 3 of 4.**

A disclosure was made before scoring: the frozen record quoted idealized `E[d]` of 3.00/6.00/7.00,
while the distributions as sampled give 3.38/5.72/6.98. Each cell was scored against its
**measured** `E[d]` — the reading most favorable to the prediction.

| Distribution | E[d] | steps | \|err\| | accuracy | passes |
|---|---:|---:|---:|---:|---|
| uniform | 4.55 | 6.12 | 1.577 | 0.4356 | no |
| hard-skewed | 5.72 | 7.28 | 1.560 | 0.2461 | no |
| easy-skewed | 3.38 | 3.39 | **0.007** | **0.9959** | yes |
| narrow | 6.98 | 7.71 | 0.730 | 0.3334 | no |

**What broke was accuracy, not step count** — which meant the prediction had not really been tested
at all. So the boring explanation got tested first, as the rules require.

## V.4 The boring explanation was right

Re-run the **original** task at the **original** training budget across seeds:

| | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| original, 500 batches | 0.9994 / 4.54 | **0.4308 / 5.96** | 0.9998 / 4.54 | 1.0000 / 4.54 | **0.4913 / 5.75** |
| original, 400 batches | 0.9998 / 4.54 | **0.4181 / 6.00** | 0.9997 / 4.54 | 0.9998 / 4.54 | **0.5105 / 5.55** |

**7 of 20 runs reach ≥ 0.99.** The distribution is **bimodal** — nothing between 0.5664 and 0.9994.
Training volume is irrelevant; the seed decides. The headline was drawn from a successful seed. It
was not fabricated, and it reproduces exactly on seeds 0, 2 and 3 — but as a statement about the
architecture it was wrong, because it omitted that four runs in ten do not get there.

**The matched control settles it.** Q0 under identical conditions: **10 of 10** at ≥ 0.99, minimum
0.9919, on both task constructions. The task is not flaky, the optimizer is not flaky, the budget is
sufficient. The unreliability belongs to the architecture.

> Q3 buys 1.77× less compute in exchange for a **40% chance of a silently broken model**, against a
> baseline that never breaks. Expected accuracy across seeds: **0.78 for Q3, 1.00 for Q0.**

## V.5 The finding worth carrying elsewhere

> **A collapsed Q3 run reports 5.2–6.1 average steps.**

That is a plausible, non-degenerate, adaptive-looking allocation, comfortably below the fixed depth
of 8. Read the step counter alone and **all thirteen failed runs look like working elastic models
delivering a 1.3–1.5× saving.** Only the accuracy column reveals the model is wrong more than half
the time.

> **Any adaptive-compute system whose halting signal is also its answer can fail this way.** A
> compute-saving figure reported without a matched accuracy figure *and* a matched seed-reliability
> rate cannot distinguish a working elastic model from a broken one.

If one sentence from this program should outlive it, it is that one.

## V.6 The repair, and why it failed instructively

`QNEURO3-Q4-P1` was frozen — hashed — **before its code was written**:

> **Mechanism hypothesis.** Q3 is bimodal because nothing in its loss forces the recurrent state to
> track *position on the chain*. The arrival head is the state's only consumer and is supervised by
> a single scalar per example, so a competing "fire at a typical depth" solution is available and
> the same loss also rewards it.
>
> **Intervention.** Add a training-only per-step readout predicting the currently occupied node.
> Zero inference cost, identical halting rule — verified by a test that loads the same weights into
> both models and asserts bit-identical halting.
>
> **Pass:** ≥9/10 and ≥8/10 seeds at ≥0.99, with steps within 0.25 of 4.54.
> **Kill:** ≤6/10 on the original task means the hypothesis is false, no second variant, cycle
> closes.

**Result: 0 of 10 on both tasks. Kill condition fired as written.**

> **It did exactly half of what was predicted, and the useless half.** The collapse mode is
> genuinely **gone** — no Q4 run falls below 0.6322, where four Q3 runs sat at 0.42–0.51. But the
> good mode went with it: Q4's best run anywhere is 0.9500. Grounding the state cured the variance
> by pulling both tails to the middle.

> **Proposed explanation** (offered as explanation, not established): a 24-way cross-entropy at
> every one of eight steps dominates the single scalar arrival signal, so the state is optimized to
> *name the node* rather than to make the one binary decision that matters. An easier auxiliary task
> can absorb the representation.

## V.7 Cycle 1 verdict, and what was not attempted

**Negative.** On `chase_to_goal`, adaptive depth loses to fixed depth once reliability is counted
alongside compute.

Two questions are **open and stated as open**: whether Q3's result depends on the answer coinciding
with the step count (the decoupling control was specified and deliberately not run, because it would
have measured training reliability a second time under a different name); and why two task
constructions inducing the same distance distribution give 6/10 versus 1/10.

**Branches of the 3.0 directive not attempted**, recorded as not done rather than quietly dropped:
complex-field and optical/spectral computation (deprioritized — `FAIL-005` already showed the
complex advantage vanishes against a best-real envelope); sub-agent and collective-field
computation; homeostasis and self-repair; distillation; multi-objective Pareto training beyond the
accuracy/compute frontier measured here.

Cycle 1 consumed its budget establishing that the first architecture family fails. That is the
outcome the program is designed to surface early rather than late.

---

# Part VI — Cycle 2: the fix, and thirteen predictions later

## VI.1 Reading the failure properly

Cycle 1 closed with "adaptive depth loses once reliability is counted." That is a verdict, not a
diagnosis. The diagnosis took one measurement that should have been taken first: **accuracy
conditioned on true distance.**

> **Experimental.** A failing Q3 run scores `1.00 1.00 0.34 0.12 0.03 0.01 0.25 0.74` across
> distances 1–8. Perfect for two hops, then collapse. It is not halting at a typical depth; the
> recurrent state stops carrying position after two steps.

> **Intuition.** The state is a residual accumulation with no scale control. Its magnitude drifts
> with depth, so a halting head with a fixed sigmoid threshold is effectively asking a different
> question at step 6 than at step 2.

> **Formal.** RMS-normalise after each hop: `h ← h · rsqrt(mean(h²) + ε)`.

> **Experimental.** 11 of 24 seeds became **6 of 6**, every success landing on exactly 1.0000 at
> 4.54 steps. Confirmed at 20 of 20. Three other single-variable interventions run at the same time
> did nothing: a goal-match feature (3/6), dense per-step halting supervision (3/6), both (2/6).

Layer normalisation in recurrent networks is textbook. **No novelty is claimed and none is
available.** Two things about it here are worth the ink.

**It is an interaction, not a main effect.** The same normalisation *destroys* the fixed-depth model
on the same task, 1.0000 → 0.1281–0.2483. An unnormalised residual state carries magnitude
information that a distance readout uses; removing it removes the readout's substrate. Predicate
halting reads a *direction* and is helped; confidence halting reads a *magnitude* and is hurt.

**And the first version of that inference was wrong.** The normalised mixture-halting runs were
compared against unnormalised runs at a different ponder weight. Checking the cycle-1 records showed
unnormalised mixture halting scored the same 0.25 at those weights, so the comparison said nothing.
It was withdrawn and re-run at matched ponder before anything was concluded.

## VI.2 A committor, at last — and it still dies

`DISCOVERY-002`'s committor branch died as `FAIL-009` because it needed two distinguishable
functional attractors and none existed. Cycle 1 produced exactly that: an outcome that is either
0.9994–1.0000 or 0.42–0.57, with almost nothing between.

> **Formal.** Instrument 24 runs with 21 mechanistic observables logged every 100 optimizer steps:
> linear decodability of the current node, of hops remaining, of the goal indicator; halting-
> distribution entropy and per-step profile; attention entropy; state effective rank. Ask whether
> any separates the eventual mode earlier than validation accuracy does.

> **Experimental.** **None does.** First perfect separation is step 1900 of 4000 for accuracy,
> decodability, loss and halting entropy alike. The best early AUC at step 500 is 0.731 for
> `dec_remaining` against accuracy's 0.521 — inside multiple-comparison noise for 21 observables
> across 7 timepoints.

Recorded as `FAIL-027`. The branch closed without a prediction being issued: early validation
accuracy is the operative predictor, and kill-and-restart on it is standard practice.

Note what this costs. Cycle 1's variance record described the outcome as bimodal "with nothing
between 0.57 and 0.99" on the strength of ten runs. Twenty-four runs produced a 0.7283. The gap is
real but not empty, and the earlier wording was too strong.

## VI.3 The control that everything passed

Cycle 1 left one question explicitly open: does the result depend on the answer being the step count?

The first attempt at the control kept the goal at node 0 and asked for the goal's label.
**Everything scored 1.0000** — including runs whose halting was 30% correct.

> A control that every configuration passes is not a control. The tell was the contradiction: a
> model halting at the wrong step cannot read the right node's label, unless it never needed to
> walk. A fixed, known goal identity makes its label directly addressable by attention at any step.

Preserved as `FAIL-026`. Rebuilt content-addressed: walk to the **first** node whose label matches a
per-example query, then report **which node** that is. Shortcut audit — distance alone 0.064,
guessing any node carrying the query label 0.291, chance 0.042.

Then nobody could solve it, for two reasons at once (`FAIL-028`). The label of the node just moved
to was never available to the halting head, so the predicate was unanswerable; and the head was a
linear map on a concatenation, which cannot express *"these two are equal"*. The second defect is
the identical expressivity gap the goal-match variant had exposed in §VI.1, reintroduced from
scratch in a new task a few hours later.

## VI.4 The separation, and three frozen attempts to make it a principle

With the task honest, the result was dramatic.

| Model | Answer read from | Steps | Answer acc | Step-id acc |
|---|---|---:|---|---|
| fixed | final state | 8.00 | 0.221–0.223 | — |
| fixed, 3× training | final state | 8.00 | 0.256–0.273 | — |
| fixed_supervised | final state + per-step loss | 8.00 | 0.232–0.243 | 0.13–0.31 |
| gated | final state + explicit latch | 8.00 | 0.222–0.239 | 0.16–0.18 |
| mean_pooled | mean of all states | 8.00 | 0.259–**0.847** | 0.16–0.86 |
| **select** | **input-selected step** | 8.00 | **1.0000** | **1.0000** |
| **arrival** | **first step the predicate fires** | **4.45** | **1.0000** | **1.0000** |

Five distinct fixed-depth alternatives fail where input-selected readout succeeds on every seed. And
the sharpest detail: `fixed_supervised` and `select` receive an *identical* per-step match signal,
yet `select` identifies the match step at 1.0000 while `fixed_supervised` manages 0.13–0.31.
**Where the answer is read from decides whether the per-step head can learn at all.**

That is the kind of finding that feels like a principle. It was frozen three times, and falsified
three times.

**`QNEURO3-ATTRIB-P1` — the mechanism was false.** The account said the fixed model must transport
the matched identity through the remaining iterations and degrades in proportion. Accuracy by
distance is **flat at ~0.22**; the required `d=max_depth` minus `d=1` margin was 0.30 and measured
0.02 and 0.07. It fails uniformly, *including where nothing is carried at all.* Kill triggered,
exactly as the frozen anticipated failure modes said in their first line.

Diagnosing it exposed a confound in our own comparison — the attribution models were told which step
was the match and the fixed model was not. The matched-supervision control removed it. The
separation survived. The mechanism did not.

**`QNEURO3-TRANSFER-P1` — it did not generalise.** An untouched, qualitatively different family:
streaming threshold-crossing, one token per step, an arithmetic predicate, **no attention anywhere
in the core**. Required: final-state readout at least 0.20 below selection. Measured **0.9351
against 0.9425 — a gap of 0.007.** Kill triggered.

The best model on the new family is the **explicit latch** at 0.9760, one of the controls that
failed on lookup. The two families reward opposite designs, and that fact does more damage to the
principle than the failed threshold does.

**`QNEURO3-EXTRAP-P1` — it buys no capability.** The one thing halting could uniquely offer:
running past the depth it was trained at, since its stopping rule is a local condition rather than a
count. Trained at 12, evaluated at 16: **0.8328 / 0.3528 / 0.6548** against a required 0.80 on every
seed.

And E2 *inverted*. The **unnormalised** model extrapolates at 0.9136 against the normalised model's
0.6135. Normalisation costs 0.30 of extrapolated accuracy while buying about 0.05 in distribution —
which bounds the fix that opened the whole cycle. It is a family-specific engineering trade-off, not
a principle, and the final architecture exposes it as a flag rather than a default.

## VI.5 The ceiling

One clause survived, and it had already transferred prospectively as T3 of a prediction that
otherwise failed: halting on a supervised predicate costs nothing in accuracy and returns the
workload's full saving. Mean steps track `E[predicate index]` to within 0.1 across six settings.

The first attempt to bank it failed, and failed *usefully*.

> **Experimental.** `QNEURO3-PARETO-P1` measured batched wall-clock latency at depth 32 and found
> **1.0×**, despite a 6.5× step-count saving.

> **Why.** The batched forward runs every step and *then* selects. The saving was nominal. Nobody
> had checked, because the step counter said 4.91 and the step counter is exactly the instrument
> cycle 1 warned could not be trusted alone.

Implementing genuine early exit and benchmarking a *trained* model — three separate measurement
errors were made and corrected getting to this, including benchmarking an untrained model whose
halting head never fires — exposed the real structure:

> **Formal.** A batch cannot exit until its slowest member does. Batched cost tracks
> `E[max halt over the batch]`, not `E[halt]`:
> `E[max] = Σ_k k · (F(k)^n − F(k−1)^n)`.

> **Experimental.** For `P(k) ∝ 0.8^k` on 1..32: E[max] = 4.97 at batch 1, 12.53 at 8, 20.96 at 64,
> 29.42 at 1024. The realisable saving decays from 6.43× to 1.09×.

This is arithmetic, and it applies to every per-example adaptive-compute method — ACT, PonderNet,
early-exit transformers, depth-routed mixtures of experts. It is, plausibly, why adaptive-depth
methods keep failing to deliver at serving scale.

## VI.6 The thirteenth prediction

`QNEURO3-NICHE-P1` was frozen on the streaming family's measurements and opened, once, on the
associative-lookup family at depth 24 with a heavy-tailed difficulty distribution never used before.

| Clause | Required | Measured | |
|---|---|---|---|
| N1 accuracy matched | within 0.02 of `select` | 1.0000 vs 1.0000 | ✓ |
| N2 optimal allocation | steps within 0.5 of E[d] | 6.14 vs E[d] **6.14** | ✓ |
| N3 small-batch win | ≥ 2.5× at batch 1 | **2.78×** | ✓ |
| N4 **the ceiling reappears** | ≤ 1.2× at batch 256 | **0.97×** | ✓ |

**The first frozen prediction in the programme to pass as written.** Twelve preceded it.

N4 is why it counts. A claim that only predicts its own success is weak. This one predicted **where
it stops working** — on a family it was not derived from — and the entire crossover curve came with
it: 2.99× at batch 1, 1.58× at 4, 1.18× at 16, 0.99× at 64, 0.97× at 256.

## VI.7 What Q-Neuro 3.0 is

`qneuro3/adaptive.py`. Small, and every part carries the measurement that justifies it.

- **first-arrival objective** — mixture halting caps at 0.6241 where this reaches 1.0000; commit
  halting reaches 0.9999 but only at full depth, buying nothing. No ponder weight to tune.
- **answer read at the halt step** — 0.22 → 1.00 on lookup tasks; scoped there, per `TRANSFER-P1`.
- **normalised state, as a flag** — 11/24 → 20/20 reliability on lookup; −0.30 extrapolation on
  streaming.
- **genuine early exit** — without it the saving measures 1.0×.

The M2 modes are not presets. Each is the regime where a measurement says something different is
correct: **Eco** (batch 1, early exit, the full 2.8–4.9×), **Balanced** (batch < 32 while ≥1.15×
remains), **Throughput** (batch ≥ 32, early exit **off**, because there it is a measured 0.97–0.99×
— a penalty).

---

# Part VII — The ceiling moves: a runtime, and a second boundary

## VII.1 The over-claim, corrected before anything else

The confirmed result came with a ceiling, and the ceiling was written up as though it were a fact
about adaptive computation. It was a fact about one execution policy.

> **Intuition.** If you advance every example in a batch in lockstep until the last one halts, the
> batch pays the slowest member. That is a scheduling decision, not a property of halting.

> **Formal.** Lockstep executes `n · max_i d_i` example-steps where the useful work is `Σ_i d_i`.
> At batch 256 on the lookup family: 5888 rows executed, 1415 useful — a **4.16× waste ratio**.

The corrected statement: *heterogeneous halt depths create a straggler effect under lockstep
batching, so per-example savings translate strongly into low-batch latency and can vanish at large
batch.* The open question that creates is whether a better runtime recovers it.

## VII.2 Prior art, and it is decisive

Run first, by rule. **Active-set compaction is not new** — it is the standard early-exit loop, it is
what MoE dispatch does at every layer, and it is what sequence packing does for variable-length
attention. **Continuous batching is not new** — it is iteration-level scheduling as deployed in LLM
serving. **Length bucketing predates transformers.**

So no novelty was available and none is claimed. All four were implemented as baselines
(`qneuro3/runtime.py`), each required to reproduce lockstep's answers exactly. That requirement
immediately earned itself: with deferred compaction, a row that fired kept being advanced until the
next gather and overwrote its own answer with a later step's logits. The equivalence check caught it
in 13 rows at batch 16 and 215 at batch 256. Nobody would have noticed it in a timing table.

What the audit *did* find is that the assumptions differ even where the mechanisms don't.
Continuous batching assumes a request stream to backfill from, and the confirmed niche is
single-stream. Compaction assumes the gather is cheap relative to a step, which is true on a GPU
matmul and is exactly the open question on a fanless M2. And bucketing assumes depth is known before
execution — true for sequence length, **false for adaptive halting**, where the depth is the output
of the computation.

## VII.3 What compaction recovers

> **Experimental.** On the associative-lookup family, against the same matched-accuracy full-depth
> baseline, both at 1.0000 accuracy and matched parameters:
>
> | batch | lockstep | **compacted** |
> |---:|---:|---:|
> | 1 | 3.64× | 3.27× |
> | 16 | 1.04× | **1.28×** |
> | 64 | 1.10× | **1.59×** |
> | 256 | 1.01× | **1.95×** |

The advantage stops decaying and starts growing with batch size. At batch 1 and 4 compaction is a
small *loss* — there is nothing to compact and the gather is pure overhead — so the two policies are
complementary rather than ordered, and the planner dispatches between them.

## VII.4 Two more frozen predictions, two more failures

**The cost model missed.** `QNEURO3-RUNTIME-P1` froze
`T = c_step·rows + c_launch·iterations + c_compact·compactions`, with all three constants measured on
the target family's raw forward and the *form* derived on a different family. It predicted the
crossover at batch 45. Measured: **below 16**.

> **Why.** The model is accurate where compute dominates — 1.0% error at batch 128, 11.5% at 256 —
> and wrong where overhead does, 55% at batch 16. It over-charged compaction at small batch: 15
> modelled compactions against 10 measured, using a `c_compact` taken from a synthetic gather of all
> six state tensors including the large keys tensor.

Kill condition applied. **No predictive runtime equation is claimed**, and the equation was not
patched and re-issued — that would be a new identifier and a fresh prospective test.

*And a procedural defect surfaced before it opened.* The hash first recorded could not be
re-verified from disk, because integer keys sort numerically in memory and lexicographically after a
JSON reload. Caught with no evidence in existence, and re-issued by hashing the round-trip. The rule
it establishes belongs with the others: **a frozen prediction whose hash cannot be re-verified from
disk is not frozen.**

**And the recovery does not transfer.** `QNEURO3-RUNTIME-P2` predicted it would appear on the
streaming family. Measured **1.065×** against a required 1.5×. Compaction over lockstep was 1.289×
against a required [1.3, 2.2] — missing by 0.011. Accuracy was not even matched: the baseline scored
0.9336 against arrival's 0.9189, so the comparison flattered arrival and it still failed.

The reason was written down before the run: *at 0.33 µs per example-step, a removed row saves less
than the gather that removes it.* The streaming core is attention-free and eight times cheaper.

## VII.5 The second boundary

> **Compaction removes the lockstep straggler ceiling when per-step cost is large relative to gather
> cost.** Measured: 1.95× at 2.66 µs/example-step, 1.07× at 0.33 µs/example-step.

This has the same shape as the first boundary, and that repetition is the most honest thing in the
programme. The advantage of adaptive computation is real. Every time it is measured under a stronger
control, it turns out to be bounded — first by the batch maximum, then by the ratio of step cost to
gather cost. Neither bound is a property of the mechanism. Both are properties of the machine it
runs on, and both had to be measured rather than reasoned about.

---

# Part VIII — The final phase: real data, strong baselines, three dead branches

## VIII.1 The two questions that could still overturn everything

Sixteen predictions in, one had passed, and the niche had survived every control thrown at it. Two
questions remained that could reverse the headline, and both had been deferred:

1. Does the mechanism beat *strong* adaptive-compute baselines — not fixed-depth strawmen?
2. Does it work on data I did not design?

Both were answered in the last phase. Both answers are no.

## VIII.2 Real data

> **Intuition.** Every task so far was one I built. A task I build can accidentally be a task my
> method is good at. The only way out is a dataset whose structure, split and difficulty I did not
> choose.

> **Formal.** UCI Human Activity Recognition: 9 inertial channels × 128 timesteps, 6 activity
> classes, delivered in 16 chunks of 8. Early classification — emit the class as soon as possible.
> The split is the dataset's **own canonical subject-disjoint partition**: 17 training subjects,
> 4 held out for validation by lowest ID, 9 test subjects never seen. A genuine distribution shift
> by person, and not mine to tune.
>
> The protocol was frozen and hashed before the test subjects were read. Five arms over one
> identical core, identical parameter counts (63,271), three seeds.

> **Experimental.**
>
> | arm | test accuracy | mean chunks | p95 | train s |
> |---|---:|---:|---:|---:|
> | fixed depth | **0.9127** | 16.00 | 16.00 | 3.6 |
> | **ACT** (Graves 2016) | **0.9006** | **3.61** | 10.33 | 4.5 |
> | confidence exit | 0.8811 | 2.57 | 13.00 | 3.4 |
> | confidence exit @ matched compute | 0.8747 | 2.28 | 10.00 | 3.4 |
> | **supervised halting (ours)** | 0.8112 | 2.39 | 15.67 | **8.4** |
> | PonderNet | 0.5220 | 16.00 | 16.00 | 4.3 |

**Fourth of five.** Beaten by a method from 2016 and by a softmax threshold, at 2.3× the training
cost.

The cost is structural, and it was declared before the run. Real data supplies no ground-truth halt
step. My mechanism is *supervised* halting; with nothing to supervise it, the target has to be
distilled from a teacher's earliest confident-correct chunk — which is early-exit distillation,
prior art, and requires training the teacher first.

> **The scope condition.** Supervised halting earns its place only where the task supplies a
> ground-truth halt step. Where it does — `chase_to_goal`, `query_chase`, threshold-crossing — it
> attains the optimal allocation at matched accuracy with 10/10 seed reliability. Where it does not,
> methods that need no teacher win.

That sentence is the honest headline of Q-Neuro 3.0, and it is a narrowing, not a result.

**What did transfer, completely, is the runtime characterisation**: 6.38× at batch 1, 0.70× at batch
256 under lockstep, 1.63× once compaction is used. The execution-policy findings hold on real data
even though the mechanism does not win. That is worth noticing — the durable part of this programme
was never the architecture.

One honesty note: PonderNet collapsed to 0.5220 using all 16 chunks. ACT working well on the same
core suggests the harness is fair, so I read this as a defect in my implementation rather than in
the method. A correct PonderNet would rank *above* me, which is why the error does not favour me.

## VIII.3 Three branches, three controls, three deaths

**Complex fields.** The branch the whole programme is named after, tested last.

> **Formal.** A genuinely complex state carried as `complex64`, Hermitian attention so that phase
> affects the halting decision rather than decorating it, at matched **real** parameter count —
> 56,545 against the real model's 55,001.

> **Experimental.** 1.0000 accuracy, 1.0000 halt accuracy, 6.15 mean steps, on 3 of 3 seeds. The
> real control: 1.0000 at 6.14 steps. Indistinguishable.

Q-Neuro 1.0 believed in phase. Q-Neuro 2.0 proved realification exact, which removed expressivity
as a route. This closes the last one: optimisation outcome is identical too. Nothing about the
complex formulation survives into the final architecture.

**Adaptive width.** The most promising untested idea: let the model choose depth *and* width, so the
resource spent is `C(x) = T(x)·N(x)`.

> **Experimental.**
>
> | config | cost | accuracy | params |
> |---|---:|---:|---:|
> | adaptive depth, full width | 6.15 | 1.0000 | 57,065 |
> | adaptive depth, **routed** 4/8 | 3.07 | 1.0000 | 57,065 |
> | adaptive depth, **static** 4/8 | 3.07 | 1.0000 | **32,425** |
> | adaptive depth, **static** 2/8 | **1.54** | **1.0000** | **20,105** |
> | adaptive depth, routed 2/8 | 2.13 | 0.8143 | 57,065 |

For a few minutes this looked like a clean 2× Pareto win. It was an accounting bug: the fixed-depth
arms were being charged only up to their *selected* step rather than the full depth they actually
execute, understating their cost fourfold. Fixed before any conclusion was drawn.

Then the real control killed it. A **statically** narrow model matches the router at identical cost
with 43% fewer parameters, and beats it at the aggressive setting. At binding capacity (`d = 16`)
routing is strictly worse — 0.032 against static's 0.115 at equal cost. The router is pure overhead.

**Homeostasis, self-repair, developmental specialisation, distillation.** Recorded as **not
attempted**, with reasons, rather than dressed up as negative results. Normalisation had already
taken reliability to 20/20, leaving a homeostatic controller no headroom to demonstrate; the others
would defend a mechanism that had just been restricted.

## VIII.4 What these deaths have in common

Each branch was killed by the **simplest available control**: a smaller model, an exactly equivalent
real model, a 2016 baseline. None required a subtle argument. That is the most useful thing to
notice about the whole search — the ideas that felt most like discoveries were dispatched by
controls that took a few hours to run and should have been run first.

---

# Part IX — Nova: the search for a new computational principle

## IX.1 A clean slate, and a much harder question

Nova inherited the falsification machinery and nothing architecturally. No mechanism from the
earlier eras had privileged status: complex numbers, waves, agents, halting — all had to earn their
place from zero. The objective was set deliberately out of reach of an ordinary architecture search:
find a principle that changes what capability a given amount of compute can buy.

The previous era's failure mode was named explicitly and designed against. Q-Neuro 3.0 spent weeks
developing beautiful theories about one architecture. Nova built a **discovery lab** first: tiered
screening, matched budgets, and a registry that makes rediscovering a dead idea impossible.

## IX.2 Build the instrument before the theory

> **Intuition.** A benchmark score tells you a model did well. It does not tell you whether the
> model learned the *procedure*. The cleanest way to ask is to train on short inputs and test on
> long ones: a model that learned the algorithm keeps working, a model that fitted the training
> lengths does not.

> **Formal.** Eight algorithmic tasks, each with a known optimal procedure, uniform interface
> `(B, L)` tokens in and `(B, L)` targets out, trained at lengths 8–16 and evaluated at 16, 32
> and 64.

> **Experimental — and this is the part that mattered.** Before any candidate was compared, a
> shortcut audit asked what three degenerate predictors could score. Position alone reaches
> **0.887 on `cummax`** and **0.598 on `sort`** at length 64.

Both were dropped. Had they stayed, the deliberately weak `causal_mlp` control's 0.917 on sort would
have looked like a discovery. Five tasks remained, with degenerate ceilings within 0.03 of chance.

![Shortcut audit](../research/figures/generated/nova_shortcut_audit.png)

## IX.3 The baselines refused to be a strawman

Ten architectures at matched parameters: transformers with three position schemes, GRU, LSTM,
diagonal and selective state-space models, a causal MLP, linear attention, and retention. The
capability matrix at 4× the trained length showed a clean complementarity nobody had to invent:

**Recurrence tracks state and extrapolates perfectly on it** — LSTM reaches 1.000 on parity and
0.992 on mod-sum. **Attention retrieves and extrapolates on that** — 0.764 on needle. **Neither does
both**, and nothing at all extrapolates on ordered memory.

The obvious answer to that gap is the linear-attention family: recurrent state that is also
content-addressable, the idea behind fast-weight programmers, linear transformers, RetNet and
Mamba. It was implemented **as a baseline, deliberately, so that Nova could not rediscover it** —
and it came out mediocre at both: 0.55 parity, 0.30 mod-sum, 0.58 needle. The gap was real and the
established answer did not close it. That is what made the search worth running.

## IX.4 Three hypotheses, three deaths

**H-DILUTION.** Softmax attention is not length-invariant — non-matching keys take probability mass,
so a read learned at length 16 is a different operation at 64. A read that ignores them should
extrapolate.

> **The operator-level property is real.** Read drift when 24 distractors are inserted: 0.236 for
> max and threshold, against softmax's 0.724.

> **The task-level effect is not.** The confound control — ordinary softmax with the *same* post-read
> normalisation the unnormalised readers require — moves copy from 0.172 to 0.305, capturing
> essentially the whole apparent gain. Max lands at 0.321, inside noise.

Two bugs were caught getting there. The first `max` normaliser divided by the sum at the end, which
is algebraically *exactly softmax*; it matched the control to three decimals, and the hypothesis had
not been tested at all. **An operator can have a property without the property mattering. Those are
different questions and only a control separates them.**

**H-INTERFERENCE.** An LSTM alone reaches 0.992 on mod-sum. Add attention and it collapses,
identically for three normalisers. A test-time branch ablation asked whether attention was drowning
out a working recurrence: turning attention off makes it **worse** (0.291 → 0.157), and turning the
recurrence off is equally bad. Neither branch works alone. The model found a joint solution that
needs both and does not extrapolate.

The frozen prediction said a handicap would fix it. All four clauses failed. Dropout does not
de-conflict the routes — it slides the model along a trade-off until it simply *is* an LSTM again.

*And a validity threat surfaced that touched every number in Nova.* The dramatic version of this
effect was an 800-step undertraining artifact; at 2400 steps mod-sum reads 0.776, not 0.291.
Everything was re-measured. The prediction still fails as scored; its narrative was corrected.

**H-COMPOSE.** If the competition is an artifact of which two routes were paired, a model with all
three should approach the per-task best everywhere.

> Mean 0.692 against a required 0.75 — the best of any architecture tested. And reverse fell to
> **0.146**, chance, from 0.348 for the cursor alone.
>
> The third clause *passed*: adding attention relieved the state-tracking conflict exactly as
> predicted, mod-sum 0.776 → 0.998 with needle at 0.977. Ordered memory died in the same change.

**Capability competition is conserved. Relieving it between one pair reintroduces it elsewhere.**

![Capability competition](../research/figures/generated/nova_competition.png)

## IX.5 The winner was a 2014 paper

The single best mechanism Nova produced is the `cursor`: an LSTM controller emitting a distribution
over relative shifts `{−1, 0, +1}` that moves a read pointer over memory, with a soft window read.

That is Neural Turing Machine location-based addressing (Graves, Wayne & Danihelka, 2014), §3.3.2 —
and copy with generalisation to longer sequences is the NTM paper's *first experiment*. Nova
reproduces it at 0.398, more weakly than the original, with a read-only memory that makes it
strictly less capable than an NTM.

The prior-art firewall caught this **before** any novelty was claimed. That is the whole reason it
runs before the comparison rather than after.

## IX.6 Verdict

**No — no new superior architecture survived.** NOVA-1 on the ladder: repeatable behaviour worth
recording, no capability edge that is not prior art. Two capabilities remain unsolved by everything
tested — copy at 0.470 and reverse at 0.371 against a chance level of 0.126 — and that gap is left
open rather than papered over.

---

# Part X — What the program actually produced

## X.1 The scoreboard

**Nineteen frozen, hashed, prospective predictions. One passed as written** — the thirteenth, after
twelve failures had narrowed the claim to something small enough to be true. The three that followed
it all failed, and each narrowed the claim further rather than being absorbed into it.

Two contained genuine prospective successes inside failures (`DFREE-LAW-P3`'s homogeneity component;
`TRANSFER-P1`'s third clause). One *passed* and the pass was worthless (`DISCOVERY-001-P1`, vacuous).
Thirty-nine failures preserved with mechanisms.

## X.2 What survives

**1. The equivalence compiler.** A typed, refusing certificate system that makes "these two models
are equivalent" into a checkable claim with a level, a domain, and a transport class. It caught its
own program's central error (transport-degeneracy) and it refuses to certify what it cannot verify.
This is the durable artifact.

**2. The dimension law as a diagnostic.** `d_free = max(0, P − g_arch − n(C−1))`, with `g_arch`
decomposed into softmax common mode (`h_last + 1`) and activation homogeneity (`h` per positively
homogeneous layer). Elementary, known, and it converts a string of empirical failures into a
predicted consequence.

**3. The stability boundary, scoped to quadratics.** `ρ = ηλ_max/(2s²)`, exact for quadratic
objectives with zero false alarms in 1,476 cells, plus a confirmed differential prediction against
Adam. Dead for nonlinear models.

**4. The adaptive-compute reporting standard.** §V.5. Cheap to apply, and it invalidates a class of
results that look fine.

**5. A catalogue of measurement defects that produce plausible wrong answers.** §XI.2. This is
worth more than it looks.

**6. Supervised predicate halting, with its ceiling.** Optimal per-example allocation — mean steps
equal `E[predicate index]` to within 0.1 across six settings — a 2.8–4.9× wall-clock saving at
batch 1, and an analytic ceiling that removes the advantage above batch ≈ 32. Confirmed
prospectively. The mechanism is prior art; the measured boundary is the contribution.

## X.3 What does not survive

The transport-covariance conjecture. Canalized quotient dynamics. Functional bifurcation. Geometric
phase in training. Curriculum holonomy. Functional navigation of the near-optimal set. The
navigator's novelty. The nonlinear stability boundary. The exact dimension law under our own
exactness standard. Adaptive depth on `chase_to_goal` as originally built. Early prediction of the
training mode. The carry-distance mechanism. The readout-location principle as anything general.
Depth extrapolation from halting.

## X.4 What is not claimed

No state-of-the-art result. No new capability. No claim that adaptive depth cannot win on some other
workload. No claim that the transport conjecture is false in every possible form. No clinical
validity. No connection to quantum cognition. No general superiority for any method described here.

---

# Part XI — The beautiful results that had to be killed

Collected deliberately, because a program that reports only its survivors teaches nothing about how
often the survivors are wrong.

## XI.1 Results that were beautiful and false

**The 14-order-of-magnitude phase boundary.** Two models, provably the same predictor, on opposite
sides of a stability boundary because of coordinates. 0.9912 prediction accuracy, zero false alarms
in 1,476 cells, a confirmed differential prediction against Adam, and a clean dimensionless control
parameter. It is real — for quadratic objectives. Against a nonlinear system it failed 96 of 96.

**`rms(M) ~ T^0.625`.** A clean nonequilibrium exponent, visibly distinct from the equilibrium 0.5,
with a good-looking fit. It was a transient. `M/√T` drifted 0.52 → 1.02 and the process was still
non-stationary at `t = 64000`.

**Six of sixteen systems bifurcate.** A discrete count, in a program actively hunting for two
distinguishable fates. The count was of crossings of an arbitrary 0.9 accuracy line. A proper
bimodality check: 0.20–0.47, **all unimodal**.

**Curriculum holonomy.** Loops in curriculum space appeared to leave a geometric signature. The
stay-control — which traversed no loop at all — scored 4.19, higher than every loop.

**The complex hypothesis, finally buried.** Q-Neuro 1.0 was built on it. Q-Neuro 2.0 proved
realification exact, which removed expressivity as a route but left optimisation dynamics open. The
last experiment of the programme closed that too: 1.0000 accuracy, 1.0000 halting, 6.15 steps —
identical to the real control at matched parameters, on every seed. The idea the project is named
after contributes nothing to its final architecture.

**Adaptive width's 2× Pareto win.** Real-looking, and an accounting bug: fixed-depth arms were
charged only to their selected step, not the depth they execute. Corrected, then killed properly by
a statically narrow model that matched it with 43% fewer parameters.

**Q3's 1.77× speedup.** Matched accuracy, fewer parameters, 28.8 s training, and it reproduces
exactly on the seeds where it works. It works 6 times in 10, against a baseline that works 10 times
in 10, and it fails silently.

**The readout-location separation.** 0.22 against 1.00, with five distinct fixed-depth alternatives
failing — matched supervision, an explicit latch, mean pooling, triple training. It looked like a
principle about where answers must be read from. On an untouched family the gap was 0.007, and the
best model there was one of the controls that had failed.

**A 6.5× compute saving at depth 32.** Real in step counts, confirmed to two decimals against the
optimal allocation, and worth exactly 1.0× in wall-clock until someone implemented early termination
and discovered the batch maximum eats it anyway.

**A frozen prediction that passed.** `DISCOVERY-001-P1` returned `passes: True` on a hashed,
prospective, one-attempt test. Its grid gave 197 chances to false-alarm and **zero** chances to
miss. The strongest methodological safeguard in the program produced a meaningless pass, and only
an audit of the grid's coverage caught it.

## XI.2 The defects, in one table

Each produced a plausible, reportable, wrong answer.

| Defect | What it claimed | Actual cause | What caught it |
|---|---|---|---|
| NaN misclassification | runaway runs **converged** | a norm overflows before its entries do → `inf/inf = nan`, and `nan > threshold` is `False` | an exact-ρ probe disagreed with the sweep |
| Transport-bound sign error | bound ratios **below 1.0** | `S⁻¹` applied once too many times | the invariant was written as a test |
| Threshold artifact | 6/16 bifurcating | crossings of an arbitrary line | a proper bimodality statistic |
| Pre-asymptotic exponent | `T^0.625` | fitting inside a transient | `M/√T` not constant |
| Holonomy endpoint metric | geometric phase | convergence drift | a stay-control that beat every loop |
| Tautological candidate | perfect within-family agreement | `e₀ = 0` makes two features identical | identical to 16 significant figures |
| Gauge-probe saturation | law violated in 21 cells | a 400-sample probe cannot saturate rank | exact arithmetic: 3 configs × 7 n-values |
| Unsolvable task | ten architectures tie | the target was ambiguous | all ten returned exactly 0.1441 |
| Shared key/value embedding | recurrent chain-following fails | associative lookup impossible | accuracy pinned at the 0.136 baseline |

**Four of these were caught only by a control we nearly did not run.** That ratio is the argument
for the whole apparatus.

---

# Appendix A — The failure register

Thirty-nine preserved failures, `research/failures.json`, narrated in `docs/FAILED_IDEAS.md`. No
failed idea has been renamed and rerun.

| ID | Subject | Failure |
|---|---|---|
| FAIL-001 | `QN-000027` | simulator leakage via metadata and order tokens |
| FAIL-002 | `QN-000038` | smoke profile produced too few law cells |
| FAIL-003 | `QN-000042` | `QN-LAW-001` failed held-out R² and MAE |
| FAIL-004 | `QN-GRAND-001` | six readiness gates failed; benchmark still sealed |
| FAIL-005 | `QN-000040/42` | best-real envelope eliminates the complex advantage |
| FAIL-006 | `QE-000009` | Gate D: no estimator beat baselines on ≥2 families |
| FAIL-007 | `DISCOVERY-001-P2` | nonlinear stability boundary failed 96/96 |
| FAIL-008 | `DISCOVERY-002` | basin coherence added nothing beyond early loss |
| FAIL-009 | `DISCOVERY-002` | committor untestable — only one attractor exists |
| FAIL-010 | `DISCOVERY-003` | no genuine bifurcation in five families |
| FAIL-011 | `DISCOVERY-003` | `T^0.625` was pre-asymptotic |
| FAIL-012 | `DISCOVERY-004` | no geometric phase under paired CW/CCW |
| FAIL-013 | `DISCOVERY-005` | curriculum holonomy failed all six discriminators |
| FAIL-014 | `SYNTHESIS-001` | global contraction to a single attractor refuted |
| FAIL-015 | `DISCOVERY-007` | curvature-based navigation fails; subspace exactly flat |
| FAIL-016 | `DISCOVERY-008` | navigator novelty dead — it is proximal EWC |
| FAIL-017 | `DISCOVERY-009` | no training-accessible signal finds the OOD direction |
| FAIL-018 | `DISCOVERY-010` | navigation closed; `d_free = 0` |
| FAIL-019 | `DFREE-LAW-P1` | freeze compromised — instrument changed after failure |
| FAIL-020 | `DFREE-LAW-P2` | `g` is activation-dependent; 118/360 |
| FAIL-021 | `DFREE-LAW-P3` | 2 of 48; ELU off by exactly one |
| FAIL-022 | `QNEURO3-CYCLE-001` | adaptive depth does not beat fixed depth at matched compute |
| FAIL-023 | `QNEURO3-Q3-P1` | frozen distribution law failed 3 of 4 cells |
| FAIL-024 | `QNEURO3-Q3-VARIANCE-001` | headline reproduces on 6/10 seeds; silent failure mode |
| FAIL-025 | `QNEURO3-Q4-P1` | reliability repair failed 0/10; kill condition fired |
| FAIL-026 | `QNEURO3-DECOUPLE-001` | the decoupled control leaked; every configuration passed it |
| FAIL-027 | `QNEURO3-PREDICT-001` | no observable predicts the training mode earlier than accuracy |
| FAIL-028 | `QNEURO3-DECOUPLE-002` | read ordering and a linear head made the predicate inexpressible |
| FAIL-029 | `QNEURO3-ATTRIB-P1` | carry-distance mechanism false; profile flat at 0.22 |
| FAIL-030 | `QNEURO3-TRANSFER-P1` | the separation does not generalise; 0.007 gap against 0.20 required |
| FAIL-031 | `QNEURO3-EXTRAP-P1` | no depth extrapolation; and normalisation's effect inverted |
| FAIL-032 | `QNEURO3-RUNTIME-P1` | cost model put the compaction crossover at 45; it is below 16 |
| FAIL-033 | `QNEURO3-RUNTIME-P2` | the ceiling removal does not transfer to a cheap core |
| FAIL-034 | `QNEURO3-HAR-P1` | on real data the mechanism is fourth of five; ACT and confidence exit win |
| FAIL-035 | `QNEURO3-GATE5-A` | complex fields identical to the real control at matched parameters |
| FAIL-036 | `QNEURO3-GATE5-B` | adaptive width beaten by a smaller static model at equal cost |
| FAIL-037 | `NOVA-H-DILUTION` | length-invariant attention: real operator property, no task effect |
| FAIL-038 | `NOVA-H-INTERFERENCE-P1` | branch dropout does not de-conflict routes, it removes one |
| FAIL-039 | `NOVA-H-COMPOSE-P1` | composing three routes moves the conflict rather than resolving it |

---

# Appendix B — Rules that earned their cost

Each of these was expensive at least once. Each is kept.

**Freeze before you look.** Serialize the prediction, hash it, and have the test read its thresholds
out of the frozen record so the code cannot drift from the prediction it is testing.

**A freeze broken by a corrected instrument stays broken.** `DFREE-LAW-P1`'s substance held 126/126
and its bug was diagnosed to the exact count (3 × 7 = 21). It is still recorded as compromised,
because the measurement changed after failure was observed. This cost us a defensible result.

**One attempt, then a new identifier.** A materially changed estimator is a new discovery, not a
rescue. `DISCOVERY-001` got two attempts and stopped.

**Never rename a failed idea and rerun it.** Twenty-five failures, twenty-five original identifiers.

**Attack the boring explanation first.** It was right about the Q3 headline, right about the
bifurcation count, right about the holonomy metric, and right about the exponent.

**Run prior art as gate 1, before any performance comparison.** The navigator's novelty died at
cosine 0.66 with OGD before a single benchmark was run, which saved the effort of benchmarking it.

**Match everything.** Parameters, FLOPs, data, search budget, wall-clock — *and seeds*. The last one
is not standard and it is the one that killed Q3.

**A gate you can talk past is not a gate.** `run_qe_000010.py` refuses to freeze an estimator when
Gate D fails, and exits non-zero.

**Preserve superseded wording.** `EQUIVALENCE_SCIENCE_AMENDMENT_001.md` §1 keeps the original text
of a claim it downgrades, so the correction is auditable rather than invisible.

**Say what you did not do.** The 3.0 branches never attempted are listed by name. An omission that
is recorded is a limitation; an omission that is not is a misrepresentation. (The Q3b control was
listed here as unrun after cycle 1. Cycle 2 ran it, twice, after the first version leaked.)

**Audit a control that everything passes.** `FAIL-026` was caught because a model halting at the
wrong step still scored 1.0000, which is impossible unless the task never required the halting. A
control nothing fails is measuring nothing.

**Measure the thing the user feels, not the proxy.** A 6.5× step-count saving measured 1.0× in
wall-clock. The step counter is the exact instrument cycle 1 had already warned could not be trusted
alone, and it was trusted alone anyway for one more round.

**Predict where your result stops working.** `QNEURO3-NICHE-P1` passed because its ceiling clause
(N4) was checkable and checked. A claim that only predicts its own success cannot be caught being
wrong.

**A frozen prediction whose hash cannot be re-verified FROM DISK is not frozen.** Integer keys sort
numerically in memory and lexicographically after a JSON reload; the first `RUNTIME-P1` hash did not
round-trip. Caught before evidence existed, and it changed the freeze procedure.

**Say which policy you measured, not which mechanism you assumed.** The ceiling was written up as a
fact about adaptive computation and was a fact about lockstep execution. The correction cost nothing
because no measurement changed — but the over-reading had already been published once.

---

*End of monograph. Companion documents: `docs/PAPER.md`, `docs/TECHNICAL_BREAKDOWN.md`.*
