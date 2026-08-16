# Equivalence science amendment 001

Amendment ID: `EQ-AMEND-001`

Date: 2026-08-14

Parent commit: `a13061677035b5649109188d2125bb0d956c5fce`

Source audit: `docs/QE_AUDIT_MEMO_001.md`

Status: active amendment to interpretation. **No result artifact, frozen record, preregistration,
law record, or manuscript history was rewritten or deleted.**

Scope: synthetic and nonclinical computational research only.

---

## 0. What this amendment does and does not do

**Does:**

1. Preserves the superseded wording verbatim (§1) so the change is auditable in text, not only in
   version control.
2. Formalizes the exact status of the `ComplexOperatorState` / `ExactRealBlockOperatorState`
   relationship, which is not what the repository previously assumed (§2).
3. Downgrades the equivalence level of the complex↔real map and declares an explicit input-domain
   restriction, with a measured boundary (§3).
4. Corrects how the 1,920-cell held-out headline is composed (§4).

**Does not:**

- change any recorded number;
- modify `experiments/results/**`, `research/laws/FROZEN_CANDIDATE_001.json`,
  `docs/PREREGISTRATION_NEXT_PHASE.md`, or `docs/PROVISIONAL_LAW_FREEZE.md`;
- weaken, retract, or rescue the falsification of the intrinsic-complex-arithmetic claim;
- re-open QN-GRAND-001, which remains blocked and sealed;
- authorize any new positive claim.

**Net scientific effect: the negative result is strengthened and better identified.**

---

## 1. Superseded wording, preserved

These statements were and remain **numerically correct**. What is amended is the interpretation
attached to them.

> `README.md`, superseded interpretation column:
> "Untouched-family confirmation — 0/1,920 positive effects; mean −0.009158 — *Non-positive result
> transfers to four new families*"
> "Exact-real control — Top-1 matches in all 1,920 held-out cells — *Implemented complex
> computation is reproducible in real arithmetic*"

> `research/claims.json`, CLAIM-001 assumptions, superseded:
> `["Mapped initialization", "same data order", "same optimizer policy", "implemented architecture only"]`
>
> These four assumptions were listed as *conditions the experiment arranged*. §2 shows the first
> three were not arranged at all — they hold automatically and could not have failed.

> `docs/QE_AUDIT_MEMO_001.md` records the audit that produced this amendment.

The v1.0.0 manuscript is **not** superseded on this point. `paper/source/results.md:11` already
states the 1,478 / 442 split and already warns that "because the exact block is constrained to the
same mapped structure, these results do not show that an unrestricted real model is inherently
better." The manuscript is the most careful surface in the repository here. What it lacks is the
§2 finding — that the two models share parameter *coordinates*, not merely structure. That is
queued for the next manuscript revision; the released v1.0.0 binaries are deliberately left frozen.

---

## 2. Formal status of the complex / exact-real relationship

### 2.1 The finding

`EXACT`. The pair is **not two parameterizations of one function class.** It is **one
parameterization evaluated by two programs.**

Let `n` be the number of trainable real scalars.

- `ComplexOperatorState` declares twelve real-valued `nn.Parameter`s and holds **no complex leaf
  tensor** (`qneuro/models/operators.py:127-138`). Complex tensors are constructed inside
  `evolve` and `measure` by `torch.complex(...)` (`:158-160`) and are intermediate values only.
- `ExactRealBlockOperatorState` declares parameters with **identical names, identical shapes,
  identical dtypes, and identical initializer standard deviations**
  (`qneuro/models/equivalent.py:49-78`).
- `copy_from_complex` is a name-matched `parameter.copy_()` loop (`:80-92`).

Therefore both models have parameter space `Θ = R^n` with **the same coordinate labels**, and the
semantics-preserving map of §6.2 of the research program is

```
T = id_{R^n}
```

Verified: under a shared seed, `max |θ_complex − θ_real| = 0.0` across 564/564 parameters, with
zero `torch.is_complex` parameters on either side.

The two objects differ only in the *program* used to evaluate `F`:

```
F_complex , F_real : R^n × X → R^C
```

are distinct floating-point realizations of the same mathematical function `f`, differing in
operation order and in exactly one branch condition (§3).

### 2.2 Consequence: the pair is transport-degenerate

Define, for this program's use:

> **Transport-degenerate pair.** A pair of learning systems whose parameter map is the identity on a
> shared coordinate system. Every transport level T0–T5 is satisfied vacuously, the covariance
> defect `δ_k` is identically zero up to floating-point rounding, and no transport intervention is
> available.

The Q-Neuro complex/exact-real pair is transport-degenerate. Concretely, in every historical run:

| Transport component | Status | Why |
|---|---|---|
| Initialization | Exact, automatic | `set_seed()` before `build_model` (`run_generator_shift.py:59-61`); identical `reset_parameters` order and scales |
| Data order | Exact, automatic | `make_loader(..., shuffle=True, seed)` with `torch.Generator().manual_seed(seed)` (`run_experiment_zero.py:50`) |
| Optimizer update | Exact, automatic | AdamW over identical real coordinates; `exp_avg`, `exp_avg_sq` live in the same space |
| Regularization | Exact, automatic | Decoupled weight decay acts on identical coordinates |
| Stopping | Exact rule, tie-sensitive | Same `1e-5` NLL-improvement rule; selected step may still differ (§3.4) |
| Numerical kernel | **Not transported** | The only genuine difference |

### 2.3 The central lesson

This is the lesson the equivalence program exists to teach, and the repository learned it the
expensive way:

> An "exact real equivalent" control can be *too tightly coupled to be informative*. If the control
> shares the candidate's parameter coordinates, agreement between them is close to a
> self-consistency check, and disagreement measures only the numerics of the implementation. Neither
> outcome bears on inductive bias.

Two corollaries the program must carry forward:

1. **A control is only an architecture comparator if it is not transport-degenerate.** Before
   registering any equivalence family, compute the parameter map and check it is not the identity.
2. **Hypothesis attribution must follow the map.** The complex/exact-real pair can test only
   H4 (numerical implementation). It cannot test H1, H2, H3, H5, or H6, because the interventions
   those hypotheses require do not exist between transport-degenerate objects.

### 2.4 Reclassification

| | Previous implicit status | Amended status |
|---|---|---|
| Relationship | Two parameterizations of one function class | One parameterization, two implementations |
| Parameter map | Non-trivial realification map | `T = id` |
| Transport level | To be established | Degenerate; T0–T5 vacuous |
| Hypotheses addressable | H1, H2, H4, H5 | **H4 only** |
| Role in the QE program | First transport instrument | Historical motivating case; numerical-implementation study |

The complex/exact-real pair is **demoted from first instrument to fourth rung** of the experimental
ladder (see `docs/ML2_PREREGISTRATION_001.md` §4).

---

## 3. Equivalence-level downgrade: complex `tanh`

### 3.1 The discrepancy

`EXACT`. The two implementations of complex `tanh` are not the same function on the full input
domain, by construction.

`_complex_tanh_real_pair` (`qneuro/models/equivalent.py:13-20`) evaluates

```
tanh(r + i·m) = [ sinh 2r + i·sin 2m ] / [ cosh 2r + cos 2m ]
```

and then **clamps the denominator** to `torch.finfo(dtype).eps`. Native `torch.tanh` applies no
such clamp.

`tanh` has poles at `m = (2k+1)·π/2`, `k ∈ Z`, where `cosh 2r + cos 2m → 0`. Writing
`ρ` for the distance to the nearest pole, the denominator behaves as `≈ 2ρ²`. In float32,
`cosh(2r)` rounds to exactly `1.0` and `cos(2m)` to exactly `−1.0` well before `ρ` reaches zero, so
the **native** denominator becomes exactly `0.0` and the native result becomes `inf` or `nan`,
while the **realified** path clamps and returns a finite but arbitrary value.

Measured at the pole `m = π/2`:

| Re(δ) offset | native re | native im | realified re | realified im |
|---:|---:|---:|---:|---:|
| 1e-1 | 1.0033e+01 | −4.36e-06 | 1.0033e+01 | −4.36e-06 |
| 1e-3 | 9.8690e+02 | −4.31e-02 | 9.8690e+02 | −4.31e-02 |
| 1e-4 | **inf** | **−inf** | 1.6777e+03 | −7.33e-01 |
| 0 | **nan** | **−inf** | 0.0000e+00 | −7.33e-01 |

### 3.2 Measured critical radius

Bisection over 16 approach angles, agreement tolerance 1e-3 relative:

| dtype | `finfo.eps` | measured `ρ_c` | `sqrt(eps/2)` | ratio |
|---|---:|---:|---:|---:|
| float32 | 1.192e-07 | **1.55e-03** | 2.441e-04 | 6.35 |
| float64 | 2.220e-16 | **3.16e-08** | 1.054e-08 | 3.00 |

The measured radius is several times larger than a naive `sqrt(eps)` estimate. **Do not substitute
the analytic estimate for the measured one** in any certificate.

### 3.3 Declared domain

The complex↔real map for this operator family is certified only on

```
D(dtype) = { δ = r + i·m  :  min_k | δ − i·(2k+1)π/2 |  >  ρ_c(dtype) }
```

with `ρ_c(float32) = 1.55e-3` and `ρ_c(float64) = 3.16e-8`, applied to the pre-activation `δ` at
**every** evidence step and every batch element.

### 3.4 Equivalence-level downgrade

| Level | Previous implicit claim | Amended |
|---|---|---|
| E0 (symbolic) | — | Holds. The realification identity is exact in exact arithmetic. |
| E1 (exact finite-precision, all representable inputs) | Implied by "exact real" naming | **FAILS.** Refuted by §3.1. |
| E2 (deterministic adversarial audit suite) | — | **Holds on `D`**, pending QE-000001. |
| E3 (distributional predictive equivalence within tolerance) | — | Holds empirically; observed max NLL difference 3.58e-7 in QN-000042. |

**The name `ExactRealBlockOperatorState` overstates the guarantee.** The module is not renamed in
this amendment — renaming would break historical reproducibility, which §1.3 of the research
program forbids. Instead the certificate, not the class name, is authoritative, and every future
reference must state the level and domain.

### 3.5 Open measurement — reachability

`UNSUPPORTED`. It is **not** established that training ever enters the excluded region.

It cannot be checked retrospectively: the falsification phase (QN-000027 … QN-000042) saved no
checkpoints and no raw activations, so `δ` was never recorded. Checkpoints exist only for
QN-000007 … QN-000021.

**Obligation on QE-000001:** instrument `δ` at every step and report
`min_k min_{steps,batch} |δ − i(2k+1)π/2|` against `ρ_c`. Until that measurement exists, the correct
statement is:

> A known equivalence-degradation region exists. Its reachability under training is unmeasured.

#### First measurement (QE-000001, 2026-08-14)

`OBSERVED`. `torch.tanh` was instrumented in place — patching the operator rather than
reimplementing the model's algebra, so the probe observes exactly the values the real model
computes. Three model seeds, 60 AdamW steps each, `state_dim=8`, `rank=2`:

| Quantity | Value |
|---|---:|
| Minimum observed pole distance | **1.425** |
| Declared critical radius `ρ_c` (float32) | 1.55e-03 |
| Margin | **919×** |
| Entered excluded region | **No** |

**Interpretation, bounded.** Training in this configuration stayed roughly three orders of
magnitude clear of the excluded region. That is a substantial margin, and it is evidence that the
E1 failure in §3.1 was *not* exercised by anything resembling the historical runs.

It is **not** a proof of unreachability. The probe covers one architecture size, one optimizer, one
step budget, and random token streams rather than the NeuroWorld or independent generators — the
historical runs saved no activations, so their actual `δ` remains permanently unknown. The
certificate's domain restriction therefore stays in force, and the correct statement is now:

> A known equivalence-degradation region exists. One probed configuration stayed 919× outside it.
> Reachability under other configurations remains unmeasured.

Artifacts: `experiments/results/QE-000001/metrics.json`, `certificate.json`.

Two secondary obligations:

- **Early stopping interacts with this.** `train_one` selects checkpoints on a `1e-5` NLL
  improvement threshold, which is the same order as the observed float32 divergence (§2.2 of the
  audit memo). A discrepancy far below task significance can still select a different epoch.
  QE-000006 must measure the rate at which paired runs select different checkpoints.
- **The clamp is asymmetric.** The realified path is *more* numerically robust than native
  `torch.tanh` here. Any future native-complex model (§5) will be the fragile side, not the
  reference side.

---

## 4. Composition of the 1,920-cell held-out headline

`OBSERVED`. Decomposition of QN-000042's 1,920 paired effects by the cellwise best-real winner:

| best_real winner | cells | mean effect | exactly zero |
|---|---:|---:|---:|
| `exact_real_block_operator` | 1,478 (77%) | +0.00000 | 1,478 (100%) |
| `real_polar_operator` | 442 (23%) | −0.03978 | 60 (13.6%) |
| **all** | **1,920** | **−0.00916** | **1,538** |

Given §2, the 1,478 zeros are **equivalence-induced**: the selected best-real model is a
transport-degenerate implementation of the candidate itself. They must not be presented as 1,478
independent wins by a distinct real architecture.

The entire −0.00916 mean is carried by the 442 `real_polar_operator` cells. `real_polar_operator`
is a genuinely different model — polar coordinates, different function class, **not** an
equivalence map — so this remains a real architecture-level reversal.

QN-000040 is the healthier heterogeneous-control result and should carry more of the argument:

| best_real winner | cells | mean effect | exactly zero |
|---|---:|---:|---:|
| `exact_real_block_operator` | 1,391 (48%) | +0.00000 | 1,391 (100%) |
| `state_space` | 637 | −0.09063 | 19 |
| `real_polar_operator` | 485 | −0.04687 | 47 |
| `gru` | 367 | −0.07073 | 19 |

Genuinely distinct models win 52% of discovery cells with substantial negative means.

### 4.1 Amended headline

> The intrinsic-complex-arithmetic claim fails on two separable grounds. First, exact realification
> reproduces the implemented computation on a declared numerical domain, so complex arithmetic is
> not *required*. Second, genuinely distinct real controls — `real_polar_operator`, `state_space`,
> and `gru` — outperform the complex operator, so complex arithmetic is not *advantageous* in the
> tested pipeline. The 1,920-cell count must distinguish equivalence-induced zeros from independent
> architecture wins; only the latter bear on the second ground.

### 4.2 Surfaces amended

| Surface | Action |
|---|---|
| `README.md` | "Amendment 001 — how the 1,920 held-out cells are composed" section added; interpretation column narrowed |
| `RESULTS.md` | "Amendment 001 (2026-08-14)" paragraph added after the confirmation summary |
| `research/claims.json` | `AMENDMENT-001` counterevidence entries and `amendments` blocks added to CLAIM-001 and CLAIM-002 |
| `dashboard/data.js` | Regenerated from the amended ledger via `scripts/build_dashboard_data.py` |
| `release/manifest.json`, `release/verification_report.json` | Re-frozen; 22/22 semantic checks pass |
| `paper/source/**` | **Not modified.** Already discloses the split; §2 finding queued for the next revision |
| `experiments/results/**`, frozen law, preregistrations | **Not modified** |

---

## 5. Forward obligations created by this amendment

1. **`QE-000001`** must emit a certificate stating equivalence level **E2/E3 on `D(dtype)`**, never
   E0/E1 globally, and must measure pole reachability (§3.5).
2. **Every registered equivalence family** must publish its parameter map and an explicit
   `transport_degenerate: true/false` flag. Degenerate pairs may not be used to test H1, H2, H3, H5,
   or H6.
3. **The first instruments of the QE program** must be non-degenerate: hidden-unit permutation,
   scaling orbit, and dense-vs-factorized. See `docs/ML2_PREREGISTRATION_001.md`.
4. **A genuinely `torch.complex`-parameterized model** is required before any claim about PyTorch
   optimizer behaviour on complex tensors. It is a new prospective family, not a modification of
   historical Q-Neuro evidence, and it may not be used to revive the complex-superiority claim.
   **Discharged by QE-000006 — see §5.1.**
5. **Raw predictions and per-step trajectories must be preserved** for all QE runs. The absence of
   these artifacts is what made §3.5 unanswerable retrospectively.

---

### 5.1 Obligation 4 discharged (QE-000006, 2026-08-14)

`OBSERVED`. The audit's Q3 — how the installed PyTorch optimizer represents complex parameters —
could not be answered from the historical code, because no complex leaf parameter existed. A
genuinely `complex64`-parameterized model was built with a **non-degenerate** realification map
(`C^n → R^2n`; one complex tensor becomes two real tensors, and the dtype changes).

PyTorch keeps **per-component** moments for complex parameters via `view_as_real`, not
modulus-based ones: `exp_avg_sq.real` accumulates `grad.real²`, not `|grad|²`. Measured with the
forward and backward passes factored out entirely, by supplying identical gradients to both sides:

| Optimizer | weight decay | eps | Max parameter divergence over 25 steps |
|---|---:|---:|---:|
| AdamW | 0.00 | 1e-8 | **0.000e+00** |
| AdamW | 0.00 | 1e-1 | **0.000e+00** |
| AdamW | 0.01 | 1e-8 | **0.000e+00** |
| AdamW | 0.01 | 1e-1 | **0.000e+00** |
| SGD (momentum 0.9) | 0.00 | — | 1.192e-07 |
| SGD (momentum 0.9) | 0.01 | — | 1.192e-07 |

**Complex AdamW is bitwise identical to AdamW on the realified real pair**, including under weight
decay and a large epsilon — an epsilon that would immediately expose a modulus-based second moment
if one existed. The SGD residual first appears at step 6–11, never exceeds one unit in the last
place, and is complex kernel arithmetic (H4), not optimizer geometry (H2).

**Consequence, and it is stronger than the §2 finding.** The degeneracy of the historical pair was
not merely an artifact of how Q-Neuro wrote its model. Even a correctly complex-parameterized model
is **exactly conjugate** to its realification under the framework's own optimizers. Complex
parameterization supplies **no optimizer geometry** in PyTorch.

Therefore any complex-versus-real performance gap observed in PyTorch must originate in function
class, initialization, regularization, or numerics — it cannot be attributed to optimizer
coordinates. The whole complex/real family is optimizer-inert, and H2 cannot be tested through it
at all.

Scope: AdamW and SGD-with-momentum, float32, CPU, `torch 2.13.0`. Other optimizers, dtypes,
devices, and versions are unmeasured. Artifacts: `experiments/results/QE-000006/metrics.json`.

---

## 6. What this amendment does not change

- QN-GRAND-001 remains **blocked**, sealed, and unexecuted. Six preflight gates remain unmet.
- QN-LAW-001 remains **falsified** on held-out magnitude (R² −30.94, MAE 0.0313).
- No clinical, biological-quantum, or universal-superiority claim is supported.
- The complex-advantage hypothesis remains **falsified within the tested scope**, on firmer footing
  than before.
- Q-Neuro's first hypothesis dying remains the origin of this framework and is not to be undone.
