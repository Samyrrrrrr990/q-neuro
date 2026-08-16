# Technical Breakdown

Full reproduction detail for the work summarized in `docs/PAPER.md`. Derivations, algorithms,
hyperparameters, controls, negative evidence, compute accounting, and the commands to re-run
everything.

Scope: synthetic, analytic, nonclinical computational research only. Lane discipline per
`docs/LANE_POLICY.md` — Lane A is confirmatory and frozen, Lane B is exploratory and may not be
cited by any claim.

---

## 1. Environment

| | |
|---|---|
| Machine | Apple M2 MacBook Air, fanless |
| Memory | 8.0 GiB unified, 2.08 GiB available at measurement |
| Cores | 8 physical, 4 torch threads |
| Accelerator | MPS present, `complex64` supported |
| CPU↔MPS crossover | 65,536 elements (measured, not assumed) |
| Working budget | 1.04 GiB (50% of available, deliberately conservative) |
| Precision | `float64` for all analytic and law work; `float32`/`complex64` where noted |

```bash
python -c "from qneuro3 import hardware; print(hardware.detect())"
```

`Profile.device_for(elements)` dispatches below/above the measured crossover rather than defaulting
to the accelerator. `memory_budget_bytes(fraction=0.5)` is intentionally pessimistic: sustained
swapping on a fanless machine costs more than a smaller model does.

**Reproduction preamble.** Every command below assumes:

```bash
cd "<repo root>"
export PYTHONPATH="$PWD"
```

---

## 2. The equivalence compiler

### 2.1 Types

`qneuro/equivalence/spec.py`

- `EquivalenceLevel` — `E0` symbolic, `E1` bit-exact finite precision, `E2` adversarial audit on a
  declared domain, `E3` distributional, `E4` metric-only.
- `TransportLevel` — `T0`…`T5`, grading what the map carries: parameters, gradients, optimizer
  moments, learning-rate policy, weight decay.
- `DomainRestriction` — the excluded region, if any.
- `MapSpec` — `__post_init__` **raises** if `E0` or `E1` is declared alongside a domain restriction.

`qneuro/equivalence/certificate.py`

- `Certificate`, `SCHEMA_VERSION = "1.0.0"`.
- `downgrade()` raises if the requested level is stronger than the current one. There is no upgrade
  path.
- Non-`EquivalenceLevel` arguments raise `TypeError`.

`qneuro/equivalence/maps.py`

- `ParameterMap` ABC exposing `supports_optimizer_transport`; `map_gradients()` raises
  `NotImplementedError` by default, so transport is opt-in and a map that cannot do it says so.
- `IdentityMap`, `IndexShuffleMap`, `HiddenUnitPermutationMap`.

### 2.2 The map families

| Module | Map | Transport | Measured first-update discrepancy |
|---|---|---|---|
| `maps.py` | hidden-unit permutation | full | 1.192e-07 — one float32 ULP from second-layer reduction order |
| `scaling.py` | diagonal / homogeneous scaling | full under `η → η·s²` | **exactly 0** for the SGD gradient step, derived then confirmed bitwise |
| `scaling.py` | same, weight decay on | **impossible** | 3.405e-03 (SGD), 1.312e-04 (AdamW) — structural |
| `factorization.py` | dense ↔ factorized | **refused** | no transport exists |
| `native_complex.py` | complex ↔ realified | full | exactly 0 for AdamW, 1 ULP for SGD |
| `complex_real.py` | complex ↔ exact-real | E2 on a domain | 5.245e-06 forward |

**Scaling transport, derived.** Under uniform scale `s`, gradients scale as `s^{-1}` per scaled
layer. Optimizer state must scale by the gradient's power:

```python
_STATE_GRADIENT_POWER = {"exp_avg": 1, "momentum_buffer": 1, "exp_avg_sq": 2, "max_exp_avg_sq": 2}
_LEARNING_RATE_EXPONENT = {"sgd": 2.0, "sgd_momentum": 2.0, "adam": 1.0, "adamw": 1.0}
```

SGD's update is `−η∇`, which picks up `s^{-1}`; matching the parameter's own `s` scaling needs
`η → η s²`. Adam's update is scale-free in the gradient, so the exponent is 1 and the learning rate
transports unchanged. This asymmetry is the mechanism behind the stability boundary in §5.

**Weight decay breaks it structurally.** The gradient step needs `η s²` and the decay term needs
`η s⁰`. No single learning-rate policy satisfies both, which is why the discrepancy is reported
rather than eliminated. This is a fact about the pair `(scaling, decoupled decay)`, not an
implementation defect.

**Factorization refuses.** `FactorizedToDenseMap.supports_optimizer_transport = False`. Factor
descent preconditions the product — the gradient with respect to `(U, V)` induces a different
effective metric on `UV` than the gradient with respect to a dense `W` — so no transport exists and
the framework declines to approximate one.

**Complex/exact-real is certified E2 on a domain, never E0 or E1.** The excluded region is
`min_k |δ − i(2k+1)π/2| ≤ ρ_c` with **measured** `ρ_c = 1.55e-03` (float32), `3.16e-08` (float64) —
several times larger than a naive `sqrt(eps/2)` estimate. Reachability was probed once
(`QE-000001`): closest observed approach 1.425, a margin of 919×. One configuration only;
reachability elsewhere is unmeasured and stated as such.

### 2.3 Transport-degeneracy

A pair is transport-degenerate when the parameter map is the identity on shared coordinates; every
transport level is then vacuously satisfied. Detected and recorded in
`docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md` §2. This invalidated the program's own earlier
complex-versus-real result: the "exact real" control shared coordinates with the complex model, so
the transport story was vacuous by construction.

### 2.4 Tests

```bash
python -m pytest tests/test_equivalence_spec.py tests/test_equivalence_permutation.py \
  tests/test_equivalence_scaling.py tests/test_equivalence_factorization.py \
  tests/test_equivalence_native_complex.py tests/test_equivalence_analytic.py \
  tests/test_equivalence_complex_real.py -q
```

---

## 3. Gates A–H

Protocol frozen in `docs/ML2_PREREGISTRATION_001.md` (`ML2-PREREG-001`); outcomes in
`docs/ML2_GATE_STATUS.md`, which records results and never amends thresholds.

| Gate | Condition | Status |
|---|---|---|
| A — exactness | valid certificate at a stated level and domain | met, with a recorded downgrade |
| B — conjugacy | first-update discrepancy eliminated or explained | met, all six families |
| C — non-vacuous bound | bound/observed ≤ 100 on analytic cases | **pass** |
| D — cross-family signal | estimator beats every baseline on ≥ 2 families | **FAIL** |
| E — frozen confirmation | one attempt, no changes after opening | blocked by D |
| F — scale | justified by measurement cost | not started |
| G — independent replication | second framework | not started |
| H — medical | full governance | not applicable, no patient data |

### 3.1 Gate C — the bound

```bash
python experiments/run_qe_000008.py
```

405 cells; 360 scored. Grid: condition numbers 1…10⁴, step sizes 0.1–0.9 of the stability
threshold, scales 1.5/2/4, horizons 50/200/1000, three problem seeds. Least squares under diagonal
reparameterization, float64, both update maps affine so Lipschitz constants are exact spectral
norms.

| Quantity | Value |
|---|---:|
| Bound violations | 0 |
| Median bound ratio | 2.90 |
| Worst bound ratio | 58.74 |
| Threshold | 100 |
| Worst ratio with a triangle-inequality constant | **1.472e+259** |

**The finding is not that the inequality is good.** It is that the inequality's usefulness is a
property of *how the Lipschitz constant is obtained*. As the spectral norm of the whole update
operator, the bound is tight to ~59×. Via `‖I − ηH‖ ≤ ‖I‖ + η‖H‖`, which discards exactly the
cancellation that makes gradient descent contractive, the same bound is vacuous by 250+ orders of
magnitude.

**Two exclusions, both declared.** 45 cells were excluded and flagged
`observed_at_numerical_floor`: the observed divergence sat at the float64 rounding floor, where a
ratio is meaningless. And the `§23.1` counterexample — an unstable system where the Lipschitz
product explodes while predictive divergence stays small — was searched for and **not found** in
this microcosm class. The obligation remains open and moves to the nonlinear setting.

**Scope.** Affine maps with exactly computed constants are the most favorable possible setting.
Gate C passing here says nothing about nonlinear models, where the constant must be estimated.
§5.2 converts that caveat into a measured failure.

### 3.2 Gate D — the failure

```bash
python experiments/run_qe_000009.py   # scores candidates against baselines
python experiments/run_qe_000010.py   # refuses to freeze; exits non-zero on gate failure
```

216 transport traces over four families, scored by **leave-one-family-out** held-out R² on log₁₀
final predictive divergence.

| Feature | Kind | Mean held-out R² | Within-family R² (fact. / scal. / n-cplx / perm.) |
|---|---|---:|---|
| `amplified_defect` | candidate | −20.47 | 0.402 / 0.819 / 0.401 / 0.028 |
| `cumulative_defect` | candidate | −31.71 | **0.962** / **0.812** / 0.382 / 0.000 |
| `one_step_predictive_divergence` | baseline | −92.50 | 0.459 / 0.361 / 0.460 / 0.016 |
| `loss_decrease` | baseline | −330.70 | 0.000 / 0.084 / 0.028 / 0.012 |
| `parameter_count` | baseline | −378.52 | 0.000 / 0.000 / 0.000 / 0.000 |
| `learning_rate` | baseline | −380.51 | 0.896 / 0.296 / 0.509 / 0.005 |
| `total_gradient_norm` | baseline | −628.97 | 0.373 / 0.013 / 0.001 / 0.013 |
| `mean_amplification` | baseline | −898.28 | 0.070 / 0.004 / 0.348 / 0.000 |

`cumulative_defect` beat every baseline on **one** family; the gate requires two. The gate is
enforced in code: `run_qe_000010.py` refuses to freeze an estimator, which keeps ladder rungs 5–8
(unitary/orthogonal, Fourier/time-domain, state-space, attention) sealed and blocks `QE-000012`.

**A candidate was disqualified mid-analysis, before scoring.** `one_step_defect` was listed as a
candidate. With a mapped initialization `e₀ = 0`, the first re-coupled step's defect is bit-for-bit
the first step's predictive divergence — identical in all 216 rows. It was a baseline mislabeled as
a candidate. Removed, with the reason recorded in source. **No features were added after the outcome
was observed.**

**Kill condition §8.2 is partially live and must be confronted by any revival:** `learning_rate`
alone reaches R² 0.896 within factorization and 0.509 within native-complex, beating both candidates
on those families.

---

## 4. Frozen predictions: protocol and results

### 4.1 Protocol

1. Serialize the prediction with `json.dumps(pred, indent=2, sort_keys=True)`.
2. SHA-256 the serialization; write `{"prediction": …, "sha256": …}`.
3. The test **reads its thresholds, grid and criteria out of the frozen record** and verifies the
   hash at load time, so the code cannot drift from the prediction it is testing
   (`research/discovery_lab/nonlinear_confirmation.py::load_frozen`,
   `experiments/run_qneuro3_cycle_001.py::stage_q4`).
4. One attempt. The verdict is recorded whichever way it comes out.

### 4.2 The eight

| ID | sha256 (16) | Claim | Verdict |
|---|---|---|---|
| Gate D `QE-000009` | — | estimator beats baselines on ≥2 families | **FAIL** (1 family) |
| `DISCOVERY-001-P1` | `50e438e536738 4e1` | `ρ > 1 ⟹ divergence`, nonlinear | **VACUOUS** |
| `DISCOVERY-001-P2` | `7d77e1593096ee24` | same, ρ placed by construction | **FAIL** 96/96 converged |
| `DFREE-LAW-P1` | `c4c7e6d5ba7fa85f` | `g = h_last+1` universally | **COMPROMISED** |
| `DFREE-LAW-P2` | `da962a222296f9d5` | same, untouched grid | **FAIL** 118/360 |
| `DFREE-LAW-P3` | `5a5abd9b1c1a3f7f` | `g = (h_last+1) + h·[homog.]` | **FAIL** 2/48 |
| `QNEURO3-Q3-P1` | `a29900e8a47a4d4c` | saving `= max_depth / E[d]` | **FAIL** 3/4 cells |
| `QNEURO3-Q4-P1` | `f315d09bf51dca21` | grounding repairs reliability | **FAIL**, kill triggered |

### 4.3 DISCOVERY-001 — the stability boundary

```bash
python research/discovery_lab/run_discovery_001.py
python -c "from research.discovery_lab.nonlinear_confirmation import confirm; print(confirm()['primary_prediction_passes'])"
```

**Derivation.** Under uniform scale `s` with an untransported learning rate, the target's update
operator is `I − (η/s²)H`, so its effective step is `η/s²`, and it is stable exactly when

```
ρ = η·λ_max(H) / (2s²) < 1
```

The source is stable when `ρ s² < 1`, so for `s < 1` there is an open window where the source
converges and its exact equivalent does not. `H_target = S⁻¹ H_source S⁻¹` holds exactly for any
twice-differentiable loss, so the nonlinear confirmation tests the *same* quantity rather than a
refit.

**Linear result** — 1,476 cells per optimizer:

| | SGD | AdamW |
|---|---:|---:|
| Prediction accuracy | **0.9912** | 0.5041 |
| False alarms (ρ ≤ 1 yet diverged) | **0** | — |
| Misses away from ρ = 1 | **0** | — |
| Misses exactly at ρ = 1 | 13 | — |
| Diverged cells | 720 | **1** |

All 13 disagreements sit at `ρ = 1.0`, the marginal-stability point where the spectral radius is
exactly 1 and neither verdict is defined. The differential prediction — Adam's scale-free update
should show no boundary — is confirmed at 1 divergent cell versus 720.

**Nonlinear confirmation, both attempts failed.** P1 was **vacuous**: its frozen grid used learning
rates 0.05–0.2 against a measured curvature range 0.51–10.9, so `ρ` never reached 1.1 — 197 chances
to false-alarm, **zero** to miss. It printed `passes: True` and that reading is meaningless. P2
added a non-vacuity guard requiring cells on both sides of the band, placed `η` per cell so `ρ`
equalled its target by construction, and **failed outright**: 96 of 96 cells at `ρ ≥ 1.1` converged,
SGD divergence rate 0.0000 even at `ρ = 3.0`, growth ratios 1.03–1.27 against a threshold of 2.0.

The cause was frozen in P2's `anticipated_failure_modes` beforehand: `ρ` uses the Hessian at
initialization, and a ReLU network under cross-entropy relocates to flatter regions while the loss
saturates. **Surviving fragment:** `ρ < 1 ⟹ stable`, zero false alarms across 269 scored cells over
both attempts. No third prediction was issued — two attempts are consumed and a third would need a
materially different estimator, which is a new discovery, not a rescue.

**Implementation note.** `largest_hessian_eigenvalue` uses *shifted* power iteration on
Hessian-vector products: iterating on `H + cI` converges to the most positive eigenvalue rather than
the largest in magnitude, which is what governs blow-up.

### 4.4 The dimension law

**Derivation.** `rank(J_train) = min(n(C−1), P − g_arch)`, because the gauge kernel is structural
and shared by every input distribution. Then `dim ker(J_train) = P − rank`, and removing the
`g_arch` gauge directions — inert for every distribution — leaves

```
d_free = max(0, P − g_arch − n(C−1))
```

`g_arch` components, both textbook, neither claimed as new:

- **Softmax common mode.** Adding a common vector to all logit rows leaves the softmax unchanged:
  `h_last + 1` directions (weight common mode + bias common mode), **independent of `C`**.
- **Positive homogeneity.** If `φ(cx) = cφ(x)` for all `c > 0`, then `(W₁,W₂) → (cW₁, W₂/c)` is an
  exact symmetry — one scaling per hidden unit, contributing `h` more. So tanh gives `h+1` and ReLU
  gives `2h+1`, confirmed 8/8 in a follow-up diagnostic.

**Measurement protocol** (from `DFREE-LAW-P3`): `g = P − rank(J_diff)` at `n = 600` saturation;
singular-value tolerance `1e-9·σ_max`; float64; parameters at initialization.

**Confirmation, stated before measurement, exact in 9 of 9 cells** (`P = 218`, `h = 24`, `g = 25`):

| n | 50 | 100 | 150 | 180 | 190 | **193** | 200 | 250 | 400 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| predicted | 143 | 93 | 43 | 13 | 3 | **0** | 0 | 0 | 0 |
| measured | 143 | 93 | 43 | 13 | 3 | **0** | 0 | 0 | 0 |

**And yet all three frozen attempts failed.** P1's substance held 126/126 on direct rank
measurement, but a 400-sample gauge probe cannot saturate rank when `P − g > 400(C−1)`, giving
exactly 21 mismatches (3 configurations × 7 n-values). The bug was diagnosed precisely — and the
freeze is still recorded as compromised, because **the measurement changed after failure was
observed**. P2 failed 118/360 by assuming `g = h_last+1` universally. P3 failed 2 of 48: the
homogeneity component was a genuine prospective success (`leaky_relu` and `abs` derived as
homogeneous before measurement, exactly `2h+1` in every cell), but two ELU cells at `h = 15`
measured 17 against a predicted 16 — most plausibly a singular value near the `1e-9` tolerance. The
frozen criterion was exactness, so it fails.

**Honest status:** an elementary and known relation, imperfectly confirmed under our own exactness
standard. Useful as a diagnostic, not as a discovery.

---

## 5. The navigation program and its closure

Seven attempts (`DISCOVERY-005` … `DISCOVERY-011`) to find directions preserving training behavior
while improving out-of-distribution behavior. All failed. Three independent mechanisms were needed
to close it, and each is a reusable control.

**1. `d_free` was zero.** The searches ran at `n = 600` with `P − g = 193`, so `d_free = 0`
*exactly*. There was no subspace to find; the searches were structurally doomed, not unlucky.

**2. The softmax gauge is exact, and enormous.** Principal cosines between the measured free
subspace and the predicted gauge subspace: **1.0000000000 across all 25 directions**. Traversing
472% of `‖θ‖` along it changes the maximum logit by 4.26e-11.

**3. The simplest use of the same information dominates.** Joint training on the combined data beat
every navigation method on all four measured axes simultaneously: `dL` 1.86e-2, test 0.9845,
`dOOD` +0.0758, held-out +0.0423.

**Novelty died at Gate 1, before any performance comparison.** The navigator's update is the
proximal step of diagonal EWC with full curvature: `(JᵀJ + λI)⁻¹g`. Cosine similarity 0.66 with
OGD and 0.23 with diagonal proximal-EWC. Marked dead on substance and retained as a strong
baseline. **No existing method was rebranded.**

### 5.1 Closed sub-branches, with mechanisms

| Record | Idea | Why it closed |
|---|---|---|
| `DISCOVERY-002-S2/S34` | canalized quotient dynamics | basin coherence added nothing beyond early loss |
| `DISCOVERY-002-COMMITTOR` | committor as incremental predictor | untestable — requires two distinguishable attractors, which do not exist here |
| `DISCOVERY-003` | functional bifurcation | none found in five families; the apparent one was a threshold artifact (see §7) |
| `DISCOVERY-003` | nonequilibrium exponent | `rms(M) ~ T^0.625` killed: `M/√T` was not constant (0.52→1.02) and long runs were non-stationary at t = 64000 |
| `DISCOVERY-004` | geometric phase / holonomy | paired CW/CCW at identical noise showed no phase |
| `DISCOVERY-005` | curriculum holonomy | failed all six discriminators under a deterministic ordering |
| `DISCOVERY-SYNTHESIS-001` | global contraction to one attractor | refuted; marked `SUPERSEDED_BY DISCOVERY-006` |

---

## 6. Q-Neuro 3.0, cycle 1

### 6.1 Task

`qneuro3/tasks.py`. A permutation defines a **single cycle** through `n_nodes = 24`
(`perm[order] = order.roll(-1)`); the model starts somewhere on it and must report how many hops
away node 0 is, capped at `max_hops = 8`. Guessing gives 0.136.

Two constructions induce the same distance distribution under uniform weights:
`chase_to_goal` rejection-samples a start over all nodes and keeps `1 ≤ d ≤ max_hops`;
`chase_to_goal_weighted` samples the distance from a weight vector and places the start at
`(pos − want) mod n_nodes`, and additionally attaches a goal label independent of the distance.

The task's ground truth is checked against the walk itself in
`tests/test_qneuro3.py::test_chase_to_goal_distance_agrees_with_the_walk` — the declared target must
be the first hop count that lands on node 0, and the permutation must be verified as a single cycle.

**A prior task design was discarded as unsolvable.** The model was never given the information
needed to disambiguate the target, and all ten candidate models returned exactly 0.1441.

### 6.2 Architectures

All share `Core` (`qneuro3/elastic.py`), so any difference is attributable to the halting mechanism
rather than capacity. `Core` uses **separate key and value embeddings** — chain following is an
associative lookup: match the current node against its identity (key), read its successor (value).
Using one embedding for both makes the lookup impossible and silently produces chance accuracy;
this was a real bug that sat everything at or below 0.136 until fixed.

```python
attn = softmax(keys @ h / sqrt(d));  h ← h + step(concat[h, (attn * values).sum])
```

| Model | Halting rule | Params |
|---|---|---:|
| `Q0Fixed` | always `depth` steps | 28,360 |
| `Q1Elastic` | PonderNet-style mixture, `halt_bias = −5.0` | 28,425 |
| `Q2Commit` | hard commit, straight-through | 28,425 |
| `Q3Arrival` | halt on detected arrival; the halt step **is** the answer | 27,970 |
| `Q4Grounded` | Q3 + training-only per-step position readout | 27,970 |

`Q3Arrival` computes a first-arrival distribution in log space:

```python
log_not = log1p(-p)
cum     = cat([zeros, log_not[:, :-1].cumsum(1)], dim=1)
log_first = log(p) + cum          # P(first firing at step k)
```

trained by `arrival_loss = −log_first[distance − 1].mean()`. Two invariants are tested: the masses
sum to at most 1 (falling short exactly by the probability of never firing), and moving mass onto
the true step lowers the loss.

`Q4Grounded`'s auxiliary head is **never called at inference**. The test
`test_q4_inference_path_is_identical_to_q3` loads the same weights into both models and asserts
bit-identical halting plus that every extra parameter lives in the auxiliary head — otherwise the
reliability comparison would be measuring two different things.

### 6.3 Hyperparameters, held fixed across every run

AdamW, lr 2e-3, 8 epochs, 500 training batches of 128, gradient-norm clip 1.0, `n_nodes = 24`,
`d = 64`, `max_depth = 8`, validation 25 batches of 256 from seed 90000 (disjoint from the training
range 1000–1499).

### 6.4 Commands

```bash
python experiments/run_qneuro3_cycle_001.py variance   # QNEURO3-Q3-VARIANCE-001
python experiments/run_qneuro3_cycle_001.py baseline   # QNEURO3-Q0-RELIABILITY-001
python experiments/run_qneuro3_cycle_001.py q4         # opens frozen QNEURO3-Q4-P1
python experiments/run_qneuro3_cycle_001.py all --output research/qneuro3/cycle_001_rerun.json
```

~30 s per training run; `all` is about 25 minutes on the reference machine.

### 6.5 Results

**Q3 across seeds** (task × training-budget × seed, 20 runs):

| | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| original, 500 | 0.9994 / 4.54 | **0.4308 / 5.96** | 0.9998 / 4.54 | 1.0000 / 4.54 | **0.4913 / 5.75** |
| original, 400 | 0.9998 / 4.54 | **0.4181 / 6.00** | 0.9997 / 4.54 | 0.9998 / 4.54 | **0.5105 / 5.55** |
| weighted, 500 | 1.0000 / 4.55 | **0.4680 / 5.68** | **0.4283 / 6.12** | **0.5387 / 5.53** | **0.5636 / 5.18** |
| weighted, 400 | **0.4356 / 6.12** | **0.4558 / 5.96** | **0.4455 / 6.05** | **0.4766 / 5.95** | **0.5664 / 5.33** |

7 of 20 reach ≥ 0.99. The distribution is **bimodal** — nothing between 0.5664 and 0.9994. Training
volume is irrelevant; seed decides.

**Q0 matched control:** 10 of 10 at ≥ 0.99, minimum 0.9919, on both task constructions. The task,
the optimizer and the budget are all fine; the unreliability belongs to the adaptive architecture.

**Q4:** 0 of 10 on both constructions. Range 0.6322–0.9500 — the collapse mode is genuinely gone, and
the good mode with it.

### 6.6 The frozen tests, scored

`QNEURO3-Q3-P1` predicted `average steps = E[distance] ± 0.25` with accuracy ≥ 0.99.

**Disclosure made before scoring:** the frozen record quoted idealized `E[d]` of 3.00 / 6.00 / 7.00;
the distributions as sampled give 3.38 / 5.72 / 6.98. Each cell is scored against its **measured**
`E[d]`, the reading most favorable to the prediction. Scoring against the idealized values would
only enlarge the failure.

| Distribution | E[d] | measured steps | \|err\| | accuracy | passes |
|---|---:|---:|---:|---:|---|
| uniform | 4.55 | 6.12 | 1.577 | 0.4356 | no |
| hard-skewed | 5.72 | 7.28 | 1.560 | 0.2461 | no |
| easy-skewed | 3.38 | 3.39 | **0.007** | **0.9959** | yes |
| narrow | 6.98 | 7.71 | 0.730 | 0.3334 | no |

The Q3b answer-decoupling control was **not run**: it would have measured training reliability a
second time under a different name. The question it was meant to settle — whether Q3's result
depends on the answer coinciding with the step count — **remains open**.

`QNEURO3-Q4-P1` required ≥9/10 and ≥8/10. It got 0/10 and 0/10; the kill condition fired as written
and no second variant was issued.

### 6.7 Ablation table

| Change from | Change to | Effect | Reading |
|---|---|---|---|
| shared key/value embedding | separate key and value | ≤0.136 → 1.0000 at depth 8 | chain following requires an associative lookup |
| fixed depth | mixture halting (Q1) | 1.0000 → 0.6241 | ponder collapse |
| mixture halting | hard commit (Q2) | 0.6241 → 0.9999 | the mixture, not the halting, was the defect |
| hard commit at full depth | halt on arrival (Q3) | 8.00 → 4.54 steps at 0.9995 | the saving is real, on 6 of 10 seeds |
| Q3 | + position grounding (Q4) | bimodal → unimodal at 0.63–0.95 | variance reduced by destroying the good mode |
| `halt_bias` −5.0 | −2.0 | did not cure ponder collapse | collapse is not merely an initialization artifact |

---

## 6A. Q-Neuro 3.0, cycle 2

### 6A.1 Commands

```bash
python experiments/run_qneuro3_cycle_002.py ceiling       # instant; the analytic result
python experiments/run_qneuro3_cycle_002.py reliability   # the normalisation fix, 5 variants
python experiments/run_qneuro3_cycle_002.py attribution   # 6 readout policies, lookup family
python experiments/run_qneuro3_cycle_002.py transfer      # opens QNEURO3-TRANSFER-P1 -- FAILS
python experiments/run_qneuro3_cycle_002.py niche         # opens QNEURO3-NICHE-P1 -- PASSES
python experiments/run_qneuro3_cycle_002.py all --seeds 3
```

### 6A.2 The reliability fix

Diagnosis first: accuracy conditioned on true distance. Failing Q3 runs score `1.00 1.00 0.34 0.12
0.03 0.01 0.25 0.74` across distances 1–8 — perfect for two hops, then collapse. The state stops
carrying position.

`research/qneuro3/variants.py` runs four single-variable interventions over one core:

| Variant | Change | Seeds at ≥0.99 |
|---|---|---|
| `V0_baseline` | none | 3/6 |
| **`V1_normalise`** | **RMS-normalise the state after each hop** | **6/6**, all exactly 1.0000 at 4.54 |
| `V2_goal_match` | arrival head also sees `h * key(goal)` | 3/6 |
| `V3_dense_halting` | per-step BCE instead of first-arrival NLL | 3/6 |
| `V4_match_dense` | both | 2/6 |

Confirmed at 20/20 seeds. `Core(normalise=...)` in `qneuro3/elastic.py` defaults to `False` so every
cycle-1 record reproduces bit-for-bit.

**It is an interaction, not a main effect.** Normalisation *destroys* the fixed-depth model on the
same task (1.0000 → 0.1281–0.2483), because an unnormalised residual state carries magnitude
information the distance readout uses. Matched-ponder controls were required to see this: comparing
normalised Q1 at ponder 0.02 against unnormalised Q1 at ponder 0.0 initially suggested the opposite,
and that inference was withdrawn.

### 6A.3 The decoupled task, and two rebuilds

`research/qneuro3/decoupled.py`. Final form: a cycle through `n_nodes` with per-node labels; given a
start and a query label, walk to the **first** node whose label matches and report **which node**.

Shortcut audit, mandatory before any model is scored:

| Statistic | Value |
|---|---:|
| chance | 0.042 |
| best from crossing distance alone | 0.064 |
| best by guessing any node carrying the query label | **0.291** |

Anything above ~0.30 requires walking. Two rebuilds were needed (`FAIL-026`, `FAIL-028`): a fixed
goal at node 0 made its label directly addressable, and then a read-ordering defect plus a linear
head on a concatenation made the match predicate literally inexpressible.

### 6A.4 Readout policies

Six policies over one core with identical per-step inputs, matched per-step supervision, and
parameters within 0.5% (54,744–55,001):

```
fixed             final state                          8.00 steps   0.2213, 0.2230
fixed (24 epochs) final state                          8.00 steps   0.2731, 0.2562
fixed_supervised  final state + per-step match loss    8.00 steps   0.2320, 0.2336, 0.2427
gated             final state + learned latch          8.00 steps   0.2392, 0.2344, 0.2216
mean_pooled       mean of all states                   8.00 steps   0.2687, 0.8472, 0.2586
select            input-selected step (argmax)         8.00 steps   1.0000 x4
arrival           first step the predicate fires       4.45 steps   1.0000 x5
```

Depth sweep: at `max_depth` 4 the fixed model solves it (1.0000, 0.8759); at 8 and 12 it does not
(0.22, 0.24). The frozen carry-distance explanation for that threshold is **false** (§6A.5).

### 6A.5 The frozen predictions of cycle 2

| ID | sha256 (16) | Claim | Verdict |
|---|---|---|---|
| `QNEURO3-ATTRIB-P1` | `51d342b16dcd9e4e` | carry distance explains the separation | **FAIL** — profile flat, 0.02/0.07 vs 0.30 |
| `QNEURO3-TRANSFER-P1` | `3ef3b2e56d08ca66` | the separation generalises | **FAIL** — 0.007 gap vs 0.20 required |
| `QNEURO3-EXTRAP-P1` | `09aabd9169a7713b` | halting buys depth extrapolation | **FAIL** — all three; E2 inverted |
| `QNEURO3-PARETO-P1` | `a855493fd713518e` | the saving scales to depth 32 | **FAIL** — R2 passed, R1/R3 did not |
| `QNEURO3-NICHE-P1` | `7fbcceb87f9a2193` | the win **and its ceiling** transfer | **PASS** — all four |

### 6A.6 The ceiling, derived

A batch cannot exit until its slowest member does, so batched cost is `E[max]` over the batch:

```
E[max] = Σ_k k · (F(k)^n − F(k−1)^n)
```

`qneuro3/adaptive.py::expected_max_halt`. For `P(k) ∝ 0.8^k` on 1..32:

| batch | 1 | 8 | 32 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| E[max halt] | 4.97 | 12.53 | 18.22 | 20.96 | 25.86 | 29.42 |
| realisable saving | 6.43× | 2.55× | 1.76× | 1.53× | 1.24× | 1.09× |

This is a property of per-example adaptive computation, not of this architecture, and applies to
ACT, PonderNet, early-exit transformers and depth-routed mixtures of experts.

### 6A.7 Wall-clock, measured

Matched-accuracy comparison (`arrival` vs `select`, both at 1.0000), M2 CPU, streaming family at
depth 32 and lookup family at depth 24:

| batch | lookup: select µs/ex | arrival µs/ex | speedup | streaming speedup |
|---|---:|---:|---:|---:|
| 1 | 1502.6 | 502.2 | **2.99×** | **4.89×** |
| 4 | 670.9 | 424.7 | 1.58× | 2.41× |
| 16 | 258.7 | 219.6 | 1.18× | 1.10× |
| 64 | 118.9 | 119.8 | 0.99× | 0.96× |
| 256 | 71.8 | 73.7 | 0.97× | 0.99× |

Batch-1 mean over 25 independent examples: 2.78×. Peak traced memory and parameter counts are
identical between the two.

**Three measurement errors were made and corrected while producing this table**, each of which
produced a wrong number first: benchmarking an *untrained* model whose halting head never fires (so
"early exit" ran every step plus the halting head, measuring 0.69×); comparing against a baseline
that skipped the halting head entirely; and measuring only at batch 256, where the answer is 1.0×
and the effect is invisible.

### 6A.8 The final architecture

`qneuro3/adaptive.py`: `first_arrival`, `halting_loss`, `expected_max_halt`, `plan`,
`PredicateHalting` with separate `forward` (training, all steps — the likelihood needs them) and
`infer` (genuine early termination). `plan` selects M2 Eco / Balanced / Throughput from the
difficulty distribution and batch size, and switches early exit **off** above the measured crossover
because there it is a penalty.

## 7. Measurement defects found and fixed

Each produced a plausible, reportable, **wrong** answer.

| Defect | Symptom | Cause | How it was caught |
|---|---|---|---|
| NaN misclassification | runaway runs scored **convergent** | a norm overflows to `inf` before its entries do → `inf/inf = nan`, and `nan > threshold` is `False` | an exact-ρ probe disagreed with the sweep |
| Transport-bound sign error | bound ratios **below 1.0**, a violated bound | `S⁻¹` applied once too many times to the target Hessian | the invariant was written as a test |
| Threshold artifact | "6 of 16 systems bifurcate" | counting crossings of an arbitrary 0.9 accuracy line | a proper max-gap/range bimodality check gave 0.20–0.47, all unimodal |
| Pre-asymptotic exponent | `rms(M) ~ T^0.625` vs equilibrium 0.5 | fitting inside a transient | `M/√T` was not constant (0.52→1.02); non-stationary at t = 64000 |
| Holonomy endpoint metric | every loop "showed" holonomy | `D_F(end,start)` dominated by convergence drift — stay-control 4.19 exceeded every loop | replaced by paired CW/CCW at identical noise |
| Tautological candidate | perfect agreement in all 216 rows | `one_step_defect ≡ one_step_predictive_divergence` because `e₀ = 0` | identical to 16 significant figures |
| Gauge-probe saturation | 21 spurious mismatches | a 400-sample probe cannot saturate rank when `P − g > 400(C−1)` | exact arithmetic on the mismatch count: 3 configs × 7 n-values |
| Unsolvable task | all ten models returned exactly 0.1441 | the target was ambiguous given the input | identical constant across architectures |
| Shared key/value embedding | everything at chance | associative lookup impossible with one embedding | accuracy pinned at the 0.136 guessing baseline |

Fixes are commented at their sites. The corrected transport-bound expression is
`inverse_scale * (hessian @ (inverse_scale * target) − linear_term)`.

---

## 8. Repository state and verification

```bash
python -m pytest -q                      # 239 tests
ruff check .                             # clean
python scripts/verify_release.py         # 22/22 semantic checks
python -c "import json; d=json.load(open('research/failures.json')); print(len(d['failures']))"
```

`verify_release.py --write-manifest` refuses to rewrite the manifest unless the semantic checks
pass first, so a stale manifest cannot be papered over.

**Frozen and never modified:** `experiments/results/**`,
`research/laws/FROZEN_CANDIDATE_001.json`, `docs/PREREGISTRATION_NEXT_PHASE.md`,
`docs/PROVISIONAL_LAW_FREEZE.md`, released manuscript binaries.

**31 preserved failures** in `research/failures.json`, narrated in `docs/FAILED_IDEAS.md`. No failed
idea has been renamed and rerun.

### 8.1 Record index

| Path | Contents |
|---|---|
| `research/discovery_lab/frozen/` | 8 hashed predictions and their results |
| `research/discovery_lab/generated/` | `DISCOVERY-001`…`011`, `SYNTHESIS-001`, DISCOVERY-002 sub-records |
| `research/qneuro3/` | cycle 1: `CYCLE-001`, `Q3-P1`(+result), `Q3-VARIANCE-001`, `Q0-RELIABILITY-001`, `Q4-P1`(+result), `CYCLE-001-CLOSE` |
| `research/qneuro3/` | cycle 2: `ATTRIB-P1`(+result), `ATTRIBUTION-001`, `TRANSFER-P1`(+result), `EXTRAP-P1`(+result), `PARETO-P1`(+result), `NICHE-P1`(+result) |
| `experiments/results/QE-*` | Gate A–D evidence |
| `docs/ML2_GATE_STATUS.md` | living gate record; records outcomes, never amends thresholds |
| `docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md` | superseded wording preserved; downgrade recorded |

---

## 9. Reproduction in order

```bash
export PYTHONPATH="$PWD"

# 0. environment and invariants
python -c "from qneuro3 import hardware; print(hardware.detect())"
python -m pytest -q && ruff check . && python scripts/verify_release.py

# 1. equivalence compiler, in ladder order
python experiments/run_qe_000002.py    # permutation
python experiments/run_qe_000003.py    # scaling orbit
python experiments/run_qe_000004.py    # dense/factorized — refuses transport
python experiments/run_qe_000001.py    # complex/exact-real — E2 on a domain
python experiments/run_qe_000006.py    # native complex

# 2. gates
python experiments/run_qe_000008.py    # Gate C: bound non-vacuity
python experiments/run_qe_000009.py    # Gate D: cross-family — FAILS
python experiments/run_qe_000010.py    # refuses to freeze; exits non-zero

# 3. discovery lane
python research/discovery_lab/run_discovery_001.py
python -c "from research.discovery_lab.nonlinear_confirmation import confirm; import json; print(json.dumps({k:v for k,v in confirm().items() if k!='records'}, indent=2))"

# 4. Q-Neuro 3.0
python experiments/run_qneuro3_cycle_001.py all
python experiments/run_qneuro3_cycle_002.py all
```

Expected: Gate D fails, `run_qe_000010.py` refuses, the nonlinear confirmation fails, cycle 1 closes
on a kill condition, `transfer` fails and `niche` passes. **Those are the results, not errors in
reproduction.**
