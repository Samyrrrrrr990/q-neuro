# ML2-PREREG-001 gate status

Living record of results measured against the frozen protocol in `docs/ML2_PREREGISTRATION_001.md`.

**This file records outcomes. It does not amend the protocol.** Section 16 of the preregistration
allows changes only through a numbered, dated amendment file; thresholds, split assignments, and
ladder order are unchanged by anything below.

Last updated: 2026-08-15

Scope: synthetic, analytic, and nonclinical computational research only.

---

## Gate summary

| Gate | Condition | Status | Evidence |
|---|---|---|---|
| **A — exactness** | Valid certificate at a stated level and domain | **Met, with a recorded downgrade** | QE-000001: E2 on a declared domain; E1 refused |
| **B — conjugacy** | First update discrepancy eliminated or explained | **Met** | QE-000002, QE-000003, QE-000006 |
| **C — non-vacuous bound** | Bound / observed ≤ 100 on analytic cases | **PASS** | QE-000008: worst 58.74, median 2.90, 0 violations |
| **D — cross-family signal** | Estimator beats every baseline on ≥ 2 discovery families | **FAIL** | QE-000009: best candidate won 1 family; QE-000010 refused to freeze |
| **E — frozen confirmation** | One attempt, no changes after opening | **Blocked** | No estimator frozen; QE-000012 may not run |
| **F — scale** | Justified by measurement cost and predictive value | **Not started** | — |
| **G — independent replication** | External implementation in a second framework | **Not started** | — |
| **H — medical** | Full governance gate | **Not applicable** | No patient data; unchanged |

Rungs 5–8 of the ladder (unitary/orthogonal, Fourier/time-domain, state-space, attention gauges)
remain **sealed**. They are confirmation families under §5.1 and may not be opened before the
estimator freeze at QE-000010.

---

## Gate C — detail

`OBSERVED`. 405 cells; 360 scored after excluding cells where the observed divergence sits at the
float64 rounding floor and a ratio would be meaningless.

| Quantity | Value |
|---|---:|
| Bound violations | **0** |
| Median bound ratio | 2.90 |
| Worst bound ratio | **58.74** |
| Gate C threshold | 100 |
| Worst ratio using a triangle-inequality Lipschitz constant | **1.472e+259** |

Grid: condition numbers 1 … 10⁴, step sizes 0.1–0.9 of the stability threshold, scales 1.5/2/4,
horizons 50/200/1000, three problem seeds. Least squares under diagonal reparameterization, float64,
with both update maps affine so their Lipschitz constants are exact spectral norms.

**The finding is not that the inequality is good.** It is that the inequality's usefulness is a
property of *how the Lipschitz constant is obtained*. Computed as the spectral norm of the whole
update operator, the bound is tight to within about 59×. Computed by the triangle inequality
`‖I − ηH‖ ≤ ‖I‖ + η‖H‖` — which discards exactly the cancellation that makes gradient descent
contractive — the same bound becomes vacuous by more than 250 orders of magnitude. This is §6.10 of
the research program, measured.

**Scope, and it is narrow.** Affine update maps with exactly computed constants are the most
favourable possible setting. Gate C passing here does **not** establish non-vacuity for nonlinear
models, where the constant must be estimated and over-estimation is the expected failure mode.
`docs/CLAUDE_CODE_SHAFIEE_RESEARCH_PROGRAM.md` §30 lists "the transport bound may be mathematically
correct but uselessly loose" as a live risk, and this result does not retire it.

**Counterexample search.** §23.1 asks for an unstable system where the Lipschitz product explodes
while predictive divergence stays small. It was searched for and **not found** in this microcosm
class: non-contractive targets diverge along with the bound (ratios 1.5–1.8), and long horizons at
κ = 10⁴ stayed within 8×. The vacuous regime found instead was the naive-constant one above. The
§23.1 obligation therefore remains open and moves to the nonlinear setting.

---

## Gate A — detail

The complex/exact-real map is certified at **E2 on a declared domain**, never E0 or E1. The
downgrade is recorded rather than hidden, as Gate A requires, and `MapSpec` refuses at construction
any attempt to declare a globally exact level alongside a domain restriction.

Excluded region: `min_k |δ − i(2k+1)π/2| ≤ ρ_c`, with a **measured** `ρ_c` of 1.55e-03 (float32) and
3.16e-08 (float64) — several times larger than a naive `sqrt(eps/2)` estimate.

Reachability probe (QE-000001): closest observed approach 1.425, a margin of **919×**; the excluded
region was not entered. One configuration only; reachability elsewhere is unmeasured.

---

## Gate B — detail

The first update discrepancy is explained in every family measured so far.

| Family | Discrepancy | Explanation |
|---|---|---|
| Permutation (QE-000002) | 1.192e-07 | One float32 ULP from second-layer reduction order |
| Scaling orbit (QE-000003) | exactly 0 for the SGD gradient step at full transport | Conjugate under η → η·s², derived then confirmed bitwise |
| Scaling orbit, weight decay on | 3.405e-03 (SGD), 1.312e-04 (AdamW) | Structural: no single learning-rate policy transports the gradient step and weight decay together |
| Native complex (QE-000006) | exactly 0 for AdamW; 1 ULP for SGD | PyTorch keeps per-component moments via `view_as_real`; SGD residual is complex kernel arithmetic |
| Complex / exact-real (QE-000001) | 5.245e-06 forward | Transport-degenerate pair; numerical implementation only |
| Dense / factorized (QE-000004) | no transport exists | Factor descent preconditions the product; the framework refuses rather than approximating |

---

## Gate D — detail (FAILED)

`FALSIFIED`. 216 transport traces over four discovery families, scored by leave-one-family-out
held-out R² on log₁₀ final predictive divergence.

| Feature | Kind | Mean held-out R² | Within-family R² (factorization / scaling / native-complex / permutation) |
|---|---|---:|---|
| `amplified_defect` | candidate | −20.47 | 0.402 / 0.819 / 0.401 / 0.028 |
| `cumulative_defect` | candidate | −31.71 | **0.962** / **0.812** / 0.382 / 0.000 |
| `one_step_predictive_divergence` | baseline | −92.50 | 0.459 / 0.361 / 0.460 / 0.016 |
| `loss_decrease` | baseline | −330.70 | 0.000 / 0.084 / 0.028 / 0.012 |
| `parameter_count` | baseline | −378.52 | 0.000 / 0.000 / 0.000 / 0.000 |
| `learning_rate` | baseline | −380.51 | 0.896 / 0.296 / 0.509 / 0.005 |
| `total_gradient_norm` | baseline | −628.97 | 0.373 / 0.013 / 0.001 / 0.013 |
| `mean_amplification` | baseline | −898.28 | 0.070 / 0.004 / 0.348 / 0.000 |

**Verdict: FAIL.** `cumulative_defect` beat every baseline on one family (factorization);
Gate D requires two. `amplified_defect` beat every baseline on none.

**The failure mode is calibration, not absence of signal.** Within family, `cumulative_defect` is
the strongest feature on both families that have real spread, and beats every baseline there. But
family medians span about **6.5 orders of magnitude** — permutation sits at ~1e-7 because its map is
conjugate and there is nothing left to predict, while the scaling orbit sits at ~1e-0.6. The ranges
chain rather than separating into clean clusters, so the problem is not one gap to bridge: it is
that a single global slope and intercept are imposed on families whose own intercepts differ by
orders of magnitude. Every out-of-family fit is consequently worse than predicting the mean.

**Kill conditions touched.** §8.1 (quotient-aware measurement adds no predictive power) is **not**
triggered — within-family it clearly does. §8.2 (learning rate predicts as well) is **partially
live**: `learning_rate` alone reaches R² 0.896 within factorization and 0.509 within native complex,
beating both candidates on those families. That must be confronted before any future attempt.

**A candidate was disqualified mid-analysis.** `one_step_defect` was originally listed as a
candidate. With a mapped initialization `e_0 = 0`, so the first re-coupled step's defect is
bit-for-bit the first step's predictive divergence — identical in all 216 rows. It was a baseline
mislabelled as a candidate and has been removed, with the reason recorded in the source. No features
were added after the outcome was observed.

**Consequence.** QE-000010 **refused** to freeze a primary estimator. Rungs 5–8 stay sealed and
QE-000012 may not run. Recorded as `FAIL-006` in `research/failures.json` and in
`docs/FAILED_IDEAS.md`.

---

## Lane B — DISCOVERY-001

`OBSERVED`. A sharp, analytically predicted stability boundary in equivalence breaking.

Two models representing **exactly the same predictor** at initialization, trained on identical data
with identical optimizer and hyperparameters, land on opposite sides of a stability boundary purely
because of the coordinates they are written in. Across a 1.4% change in the scale parameter the
paired divergence moves ~14 orders of magnitude, with the source stable throughout.

Control parameter, derived before any sweep: `ρ = η·λ_max(H) / (2s²)`, transition predicted at
`ρ = 1`, independent of conditioning, seed, and problem scale.

| | SGD | AdamW |
|---|---:|---:|
| Cells | 1,476 | 1,476 |
| Prediction accuracy | **0.9912** | 0.5041 |
| False alarms (ρ ≤ 1 yet diverged) | **0** | — |
| Misses away from ρ = 1 | **0** | — |
| Misses exactly at ρ = 1 | 13 | — |
| Diverged cells | 720 | **1** |

**Zero false alarms in 1,476 cells.** All 13 disagreements sit at `ρ = 1.0` itself, the
marginal-stability point where the spectral radius is exactly 1 and neither verdict is defined.

**Differential prediction confirmed.** Adam's update is scale-free in the gradient, so its effective
step does not acquire the `1/s²` factor and the boundary should be absent. It is: 1 divergent cell
out of 1,476, against 720 for SGD.

**Simplest boring explanation, stated plainly:** this *is* textbook gradient-descent stability.
Reparameterization changes the effective Hessian, hence the effective step, hence stability. The
mechanism is not new and no novelty is claimed for it. What it contributes here is that equivalence
breaking has an exactly predictable location with a dimensionless control parameter — and a
mechanistic account of the Gate D failure: **the discovery families straddle a phase boundary**, so
they are not one population and no single calibration can span them.

**Three measurement defects were found and fixed during this work**, each of which had produced a
plausible-looking wrong answer:
1. an absolute threshold on paired divergence conflated *slow* with *unstable*;
2. a growth ratio on the *paired* divergence still mixed the target's stability with the source's
   convergence;
3. a norm can overflow before any entry does, giving `inf/inf = nan`, and `nan > threshold` is
   `False` — so runaway runs were silently scored convergent. This one was caught only because an
   exact-ρ probe disagreed with the sweep.

**Promotion stage: 3 of 5, with a failed step-5 attempt.** Step 4 was completed twice — two
predictions frozen and hashed before any nonlinear evidence existed. Step 5 was attempted twice and
did not pass.

| Attempt | sha256 (first 16) | Verdict | Detail |
|---|---|---|---|
| `DISCOVERY-001-P1` | `50e438e536738 4e1` | **VACUOUS** | Frozen grid never reached ρ ≥ 1.1: 197 chances to false-alarm, **zero** chances to miss. Preserved, not re-tuned. |
| `DISCOVERY-001-P2` | `7d77e1593096ee24` | **FAILED** | **96 of 96** cells at ρ ≥ 1.1 converged. SGD divergence rate 0.0000 even at ρ = 3.0. |

**What survived:** the `ρ < 1 ⟹ stable` direction, with zero false alarms across 269 scored cells
over both attempts.

**What failed:** the `ρ > 1 ⟹ diverges` direction, completely. This was the failure mode frozen in
P2 in advance: ρ uses the Hessian at initialization, but a ReLU network under cross-entropy does not
sustain that curvature — it relocates to flatter regions and the loss saturates, so an initially
over-large step does not produce exponential growth.

**DISCOVERY-001 does not promote.** It stands as a statement about **quadratic objectives only**.
Recorded as `FAIL-007`. No third prediction was issued; two attempts are consumed and a third would
need a materially different estimator, which would be a new discovery rather than a rescue.

**Programme consequence:** Gate C passed on affine maps with exact Lipschitz constants and was
explicitly scoped as saying nothing about nonlinear models. P2 converts that caveat from a stated
risk into a measured failure.

Artifact: `research/discovery_lab/generated/DISCOVERY-001.json`.

---

## Open obligations

1. **Gate D is failed, not open.** Any further attempt requires a new experiment ID and a stated
   reason for the change, per §16. Two directions are defensible and both weaken the claim: a
   per-family random intercept (a law needing per-family calibration is a weaker object than the
   conjecture states), or restricting the population to non-conjugate families (two of the four sit
   at the rounding floor by construction).
2. **Confront §8.2 directly.** `learning_rate` alone beats both candidates on two families. Any
   revived estimator must be shown to add predictive power *after* controlling for it.
3. **§23.1 counterexample** in the nonlinear setting, where Lipschitz constants are estimated.
4. **Pole reachability** beyond the single probed configuration.
5. **Raw predictions and per-step trajectories** for every QE run, per preregistration §13.
