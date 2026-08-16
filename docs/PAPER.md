# Equivalence Breaking Under Training: A Falsification-First Study

**Status:** research report. Synthetic, analytic, nonclinical computational work only.
**Date:** 2026-08-16.
**Companion documents:** `docs/TECHNICAL_BREAKDOWN.md` (reproduction), `docs/MONOGRAPH.md` (full narrative including everything that failed).

---

## Abstract

Two neural networks can compute the identical function and still train to different places. We set
out to find the law governing that divergence — a *transport-covariance conjecture* holding that the
discrepancy between equivalent models under training is predictable from a defect measured on the
map relating them. We froze the conjecture, built the machinery to test it, and it **failed**: on
held-out families the best candidate estimator scored a mean out-of-family R² of −31.7, worse than
predicting the mean. Every subsequent attempt to recover a weaker version also failed, across three
further frozen predictions.

What survives is not the law. It is (1) an **equivalence compiler** that forces every claim of
model equivalence to name a level, a domain, and a transport class, and that refuses to certify what
it cannot verify; (2) an exact account of **when equivalence classes contain free directions at
all**, which retroactively explains a string of empirical failures as structurally inevitable;
and (3) a set of **measurement failure modes** — each of which produced a plausible, publishable,
wrong answer before being caught.

We then applied the same discipline to architecture design. An adaptive-depth model achieved a
genuine 1.77× inference-compute reduction at matched accuracy and fewer parameters. It reproduces
on 6 of 10 seeds. The fixed-depth baseline reproduces on 10 of 10. The speedup is real and is not
reportable.

We claim no state-of-the-art result, no new capability, and no general superiority for any method
described here.

---

## 1. The problem

Reparameterize a trained network — permute its hidden units, rescale a layer and inverse-scale the
next, factorize a dense matrix — and the function is unchanged. Start training from the two
descriptions and they do not stay together. The optimizer sees coordinates, not functions.

This is not a curiosity. It is why an "architecture improvement" can be an artifact of coordinates,
why an ablation can measure the parameterization rather than the mechanism, and why two teams
implementing the same idea report different numbers. Anything you would want to say about
architecture rests on knowing which differences are real.

The natural conjecture is that the divergence is *predictable*: that if you can measure how badly a
map fails to commute with the optimizer, you can predict how far apart the two models end up. We
call this the **transport-covariance conjecture**. It is the thing this program was built to test.

## 2. The principle

The organizing idea is a negative one, and it is the most load-bearing thing here:

> **Equivalence is not a property of two models. It is a property of a map, at a stated level,
> on a stated domain, with a stated transport class.**

"These two models are equivalent" is not a well-formed claim. Four questions have to be answered
before it means anything:

- **Which map** carries one to the other?
- **At what level** does it hold — symbolically, in exact finite-precision arithmetic, under
  adversarial probing, distributionally, or only in aggregate metrics?
- **On what domain** — everywhere, or with a region excluded?
- **What transports** — the parameters only, or the gradients, the optimizer state, the learning
  rate, the weight decay?

Almost every informal equivalence claim in practice answers none of these, and the answers turn out
to matter enormously. Two maps that are both "exact" can behave completely differently under
training, because one transports optimizer state and the other cannot.

## 3. The theory

### 3.1 Two ladders

**Equivalence levels**, strongest first:

| Level | Meaning |
|---|---|
| **E0** | Symbolic identity of the computed function |
| **E1** | Bit-exact agreement in finite precision |
| **E2** | Survives an adversarial audit suite on a declared domain |
| **E3** | Distributional agreement |
| **E4** | Agreement in aggregate metrics only |

**Transport levels** T0–T5 grade what the map carries: T0 parameters only; higher levels add
gradients, optimizer moments, the learning-rate policy, and weight decay.

The two ladders are independent, and their product is where the interesting failures live. E0 with
T0 is common and nearly useless: the models are the same function and will still diverge on the
first step. E2 with T5 is rare and strong.

### 3.2 Certificates that can refuse

The compiler emits a certificate naming level, domain, and transport class. Three refusals are
enforced in code rather than in prose:

- Declaring **E0 or E1 together with a domain restriction** is a construction-time error. A map
  that needs a region excluded is not globally exact, and the type system says so.
- A certificate may be **downgraded, never upgraded**. Attempting to strengthen one raises.
- A map that cannot transport gradients **says so** (`supports_optimizer_transport = False`) and
  refuses rather than approximating. The dense/factorized map is the canonical case: factor descent
  preconditions the product, so no gradient transport exists, and the framework declines to invent
  one.

This is the part we would defend hardest. It is unglamorous and it caught real errors.

### 3.3 Transport-degeneracy

A pair is **transport-degenerate** when the parameter map is the identity on shared coordinates.
Then every transport level is vacuously satisfied and the pair can tell you nothing about transport.

This sounds pedantic. It invalidated our own earlier result. The complex-versus-real comparison that
motivated the program used an "exact real" control that shared coordinates with the complex model;
the entire transport story was vacuous by construction, and the measured 5.245e-06 forward
discrepancy was numerical implementation, not equivalence breaking (`docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md`).

### 3.4 The dimension law

Let `P` be the parameter count, `g_arch` the dimension of the architecture's exact
function-preserving symmetry group, `n` the number of training points, and `C` the class count.
The first-order subspace of directions preserving all training predictions while able to change
predictions elsewhere has dimension

```
d_free = max(0, P − g_arch − n(C−1))
```

`g_arch` has two identified components: softmax common-mode invariance contributes `h_last + 1`,
independent of the class count; and each positively homogeneous hidden layer contributes `h` more,
because `(W₁, W₂) → (cW₁, W₂/c)` is then an exact symmetry.

This is rank-nullity plus a correct symmetry count. **No novelty is claimed** — ReLU scaling
symmetry and softmax shift invariance are textbook. Its value is diagnostic, and it is stated here
because it is the explanation for §5.

## 4. What was frozen, and what happened when it was opened

Every prediction below was serialized and SHA-256 hashed **before** the evidence existed. Hashes are
verified at load time; the test code reads its thresholds from the frozen record so it cannot drift
from the prediction it is testing.

| Prediction | Claim | Attempts | Verdict |
|---|---|---|---|
| Gate D (`QE-000009`) | A defect estimator beats every baseline on ≥2 held-out families | 1 | **FAIL** — won 1 family |
| `DISCOVERY-001-P1` | `ρ = ηλ_max/2 > 1 ⟹ divergence`, nonlinear | 1 | **VACUOUS** — grid never reached ρ ≥ 1.1 |
| `DISCOVERY-001-P2` | same, with ρ placed by construction | 1 | **FAIL** — 96 of 96 cells at ρ ≥ 1.1 converged |
| `DFREE-LAW-P1` | `g = h_last+1` universally | 1 | **COMPROMISED** — substance held 126/126 but the instrument was changed after seeing failure |
| `DFREE-LAW-P2` | same, untouched grid, full protocol | 1 | **FAIL** — 118 of 360 |
| `DFREE-LAW-P3` | `g = (h_last+1) + h·[homogeneous]` | 1 | **FAIL** — 2 of 48 |
| `QNEURO3-Q3-P1` | adaptive-depth saving = `max_depth / E[distance]` | 1 | **FAIL** — 3 of 4 distributions |
| `QNEURO3-Q4-P1` | position grounding repairs seed reliability | 1 | **FAIL** — kill condition triggered |

Eight frozen predictions. **Zero passed as written.** One (`DFREE-LAW-P3`) contained a genuine
prospective success inside a failure: `leaky_relu` and `abs` were derived as positively homogeneous
before measurement and gave exactly `2h+1` in every cell — but two ELU cells at `h=15` measured 17
against a predicted 16, and the frozen criterion was exactness.

We report that record as the primary result of the program.

## 5. Mechanism: why the law failed

Two mechanisms account for essentially all of it, and both were identified *after* the failures and
then checked.

### 5.1 The families straddle a phase boundary

Gate D's failure mode is **calibration, not absence of signal**. Within a family, the
`cumulative_defect` estimator is the strongest feature available and beats every baseline (R² 0.962
on factorization, 0.812 on the scaling orbit). Across families it collapses to −31.7, because family
medians span about **6.5 orders of magnitude** — permutation sits near 1e-7 because its map is
conjugate and there is nothing left to predict, while the scaling orbit sits near 1e-0.6. The ranges
chain rather than clustering, so a single global slope and intercept is the wrong object.

Why do they span that range? Because reparameterization moves the effective step size. Under uniform
scale `s` with an untransported learning rate, the target's update operator is `I − (η/s²)H`, so its
effective step is `η/s²` and it is stable exactly when `ρ = ηλ_max(H)/(2s²) < 1`. For `s < 1` there
is an open window where a model converges and its **exact equivalent diverges**. We measured this:
across a 1.4% change in `s`, paired divergence moves ~14 orders of magnitude, with 0 false alarms in
1,476 SGD cells and prediction accuracy 0.9912. The differential prediction confirms the mechanism —
Adam's update is scale-free in the gradient, so the boundary should be absent, and it is: 1 divergent
cell out of 1,476 against 720 for SGD.

The discovery families sit on opposite sides of that boundary. They are not one population, so no
single calibration can span them.

### 5.2 Initialization curvature does not govern a nonlinear trajectory

The stability boundary is exact for quadratic objectives and **fails completely** for nonlinear ones.
`DISCOVERY-001-P2` put 96 cells at `ρ ≥ 1.1` and all 96 converged; the SGD divergence rate was 0.0000
even at `ρ = 3.0`, with growth ratios of 1.03–1.27 against a threshold of 2.0.

This was the failure mode written into P2's `anticipated_failure_modes` before the run. `ρ` is
computed from the Hessian at initialization, but a ReLU network under cross-entropy relocates to
flatter regions and the loss saturates, so an initially over-large step does not compound. The defect
is not calibration. It is that one initialization-time curvature does not govern a nonlinear
trajectory — and the surviving half, `ρ < 1 ⟹ stable`, has produced zero counterexamples across 269
scored cells.

### 5.3 Why the free directions were never there

Seven consecutive attempts to navigate the near-optimal set — find directions that preserve training
behavior while improving out-of-distribution behavior — failed. The dimension law explains all of
them at once: those searches ran at `n = 600` with `P − g = 193`, so `d_free = 0` **exactly**. There
was no subspace to find. The searches were structurally doomed, not unlucky.

Stated before measurement and then confirmed exactly in 9 of 9 cells, including the predicted
transition at `n = 193`:

| n | 50 | 100 | 150 | 180 | 190 | 193 | 200 | 250 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| predicted | 143 | 93 | 43 | 13 | 3 | 0 | 0 | 0 | 0 |
| measured | 143 | 93 | 43 | 13 | 3 | 0 | 0 | 0 | 0 |

Three reusable controls come out of this: check `d_free` before searching for free directions; check
integrability before trusting a first-order subspace; check whether the simplest use of the same
information already dominates. On the last point — joint training on the combined data beat every
navigation method on all four measured axes.

## 6. Q-Neuro 3.0: the same discipline applied to architecture

Cycle 1 asked whether a model that decides its own depth can beat a fixed-depth model when the
required depth genuinely varies per example and is discoverable only by working.

**Task.** `chase_to_goal`: a permutation defines a single cycle through 24 nodes; the model starts
somewhere on it and reports how many hops away node 0 is, up to 8. Nothing in the input announces the
answer. Guessing gives 0.136.

**Results**, matched on parameters, data, training budget and wall-clock:

| Model | Mechanism | Params | Accuracy | Steps | Reliable seeds |
|---|---|---:|---|---:|---|
| Q0 | fixed depth 8 | 28,360 | 1.0000 | 8.00 | **10 / 10** |
| Q1 | PonderNet-style mixture halting | 28,425 | 0.6241 | 3.27 | — |
| Q2 | hard commit, straight-through | 28,425 | 0.9999 | 8.00 | — |
| Q3 | halt on detected arrival | **27,970** | 0.9994–1.0000 | **4.54** | **6 / 10** |
| Q4 | Q3 + training-only position grounding | 27,970 | 0.6322–0.9500 | 4.50–5.33 | 0 / 10 |

Three findings, in decreasing order of how much we like them.

**Mixture halting collapses where commit halting does not.** Q1 caps at 0.6241; changing only the
halting rule to a hard straight-through commit lifts the same core to 0.9999. Single-variable
intervention, seed held fixed.

**The 1.77× is real and is not reportable.** Q3's saving is genuine on the seeds where it trains, and
reproduces to within 0.0006 accuracy and 0.00 steps on seeds 0, 2 and 3. But the outcome is
**bimodal**: every run lands either at 0.9994–1.0000 with 4.54 steps or at 0.42–0.57 with 5.2–6.1
steps, and nothing lands in between. Training volume changes nothing; seed decides. Q0 under identical
conditions is 10 of 10 with a minimum of 0.9919, so the unreliability belongs to the architecture.
Expected accuracy across seeds: **0.78 for Q3 against 1.00 for Q0.**

**The failure mode is silent and mimics success.** A collapsed Q3 run reports 5.2–6.1 average steps —
a plausible, non-degenerate, adaptive-looking allocation, comfortably below the fixed depth of 8. Read
the step counter alone and all thirteen failed runs look like working elastic models delivering a
1.3–1.5× saving. Only the accuracy column reveals the model is wrong more than half the time.

We consider this the most portable result in the section. **Any adaptive-compute system whose
halting signal is also its answer can fail this way**, and a compute-saving figure reported without a
matched accuracy figure and a matched seed-reliability rate cannot distinguish a working elastic model
from a broken one.

The attempted repair failed instructively. `QNEURO3-Q4-P1` was frozen before its code was written:
supply the missing positional representation at zero inference cost and reliability should reach
≥9/10. Result: 0 of 10, kill condition triggered as written. It removed the collapse mode — no run
below 0.6322 — and removed the good mode with it, best run 0.9500. Reducing seed variance destroyed
the solution worth having.

## 7. Compute accounting

All experiments ran on one fanless Apple M2 MacBook Air: 8.0 GiB unified memory (2.08 GiB available),
8 physical cores, 4 torch threads, MPS with `complex64` support, measured CPU↔MPS crossover at 65,536
elements, working budget 1.04 GiB at a deliberately conservative 50% of available memory — sustained
swapping on a fanless machine costs more than a smaller model does.

No experiment in this program required more. Q3 trains in 28.8 s. The largest sweeps are the Gate C
grid (405 cells, 360 scored) and the DISCOVERY-001 sweep (1,476 SGD cells). This is a deliberate
property of the design: microcosms small enough that the ground truth is computable are worth more
than scale, when the question is whether a claim is true.

## 8. What we claim, and what we do not

**We claim:**

1. The transport-covariance conjecture, as stated, is **false** at the level of a single global
   calibration across equivalence families, and we have identified the mechanism.
2. Equivalence claims are only meaningful with a level, a domain, and a transport class attached, and
   this can be enforced mechanically.
3. `d_free = max(0, P − g_arch − n(C−1))` predicted a rank transition exactly, and explains a string
   of prior failures.
4. On `chase_to_goal`, adaptive depth loses to fixed depth once reliability is counted alongside
   compute.

**We do not claim:** any state-of-the-art result; that adaptive depth cannot win on some other
workload; that the transport conjecture is false in every possible form; any clinical validity; any
connection to quantum cognition; or that any method here is generally superior to any other.

**Open, and stated as open:** whether Q3's result depends on its answer coinciding with its step
count (the decoupling control was specified and not run); why two task constructions inducing the
same distance distribution give 6/10 versus 1/10 success; and non-vacuity of the transport bound in
the nonlinear setting, where Lipschitz constants must be estimated rather than computed.

## 9. On the negative result

Eight frozen predictions, zero passes. Twenty-five preserved failures. It would have been easy to
produce a positive result from this material — widen a tolerance, average over seeds, report the
easy-distribution cell as partial confirmation, quote 1.77× without the 6-in-10. Each of those was
available and each was declined, in writing, in the record that declined it.

The measurement defects are the reason the discipline earned its cost. Each produced a
plausible-looking wrong answer:

- A norm can overflow before any of its entries do, giving `inf/inf = nan`; `nan > threshold` is
  `False`, so runaway runs were silently scored **convergent**. Caught only because an exact-ρ probe
  disagreed with the sweep.
- A transport bound applied `S⁻¹` once too many times, producing bound ratios *below* 1.0 — a
  violated bound. Caught only because the invariant was written as a test.
- Counting crossings of an arbitrary 0.9 accuracy line suggested 6 of 16 systems were bifurcating; a
  proper bimodality check gave 0.20–0.47, all unimodal.
- A 400-sample gauge probe cannot saturate rank when `P − g > 400(C−1)`, producing exactly 21
  mismatches. Diagnosed precisely — and the freeze is still recorded as compromised, because the
  measurement changed after the failure was seen.

A program that only reported its successes would have reported at least three of these as
discoveries.
