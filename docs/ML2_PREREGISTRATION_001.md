# Equivalence-aware learning science: prospective preregistration

Preregistration ID: `ML2-PREREG-001`

Version: `1.0.0`

Date frozen: 2026-08-14

Parent commit: `a13061677035b5649109188d2125bb0d956c5fce`

Parent audit: `docs/QE_AUDIT_MEMO_001.md`

Parent amendment: `docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md`

Status: **frozen before any QE experiment is implemented or any QE outcome is observed.**

Experiment namespace: `QE-000001` onward. The `QN-` series is closed to this program.

Evidence scope: synthetic, analytic, and nonclinical computational research only.

---

## 0. Naming discipline

The scientific name of this program is **equivalence-aware learning science**.

`ML 2.0` is an **internal working label only.** It may appear in planning documents and internal
discussion. It must not appear in a preprint, a repository README headline, a figure, an abstract,
or any external communication as the name of an established field. It earns that status only under
Outcome G (§9), and only by community adoption rather than repository assertion.

The central empirical conjecture is named `transport_covariance_conjecture` in code, configuration,
and preregistration. **No eponym is permitted** in code, tests, artifacts, or a first technical
preprint. The gates that would be required before any eponym is even discussable are listed in
§7.5 of `docs/CLAUDE_CODE_SHAFIEE_RESEARCH_PROGRAM.md` and are not restated here as achievable.

Prohibited framings, enforced by `research/adversarial_reviewer.py`: any claim of clinical validity,
quantum cognition, universal superiority, or a discovered law prior to the gates in §8.

---

## 1. Research question

> When two learning systems are connected by a semantics-preserving map, which components of a
> practical training pipeline respect that equivalence, which break it, and can the measured failure
> of covariance predict the later divergence of their predictors?

This inverts the question the repository previously asked. It is no longer "does complex arithmetic
win"; it is "what part of a training result is functional, and what part is an artifact of the
coordinates the network happened to be written in."

### 1.1 Why this program exists

Q-Neuro's preferred hypothesis was falsified. The falsification is the origin of this program and
is not to be undone, softened, or relabelled. `docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md` §2
records the specific methodological failure that motivates the design below: the repository's
"exact real equivalent" control turned out to be **transport-degenerate** — it shared the
candidate's parameter coordinates, so agreement with it was close to a self-consistency check.

The program's first duty is to make that class of error detectable before an experiment runs, not
after.

---

## 2. Definitions frozen by this preregistration

**Learning system.** `S = (Θ, F, P₀, D, B, U, R, C, Q)` — parameter space, prediction map,
initialization distribution, data/minibatch stream, optimizer state space, discrete update map,
explicit regularization, stopping/selection rule, resource and numerical policy. An architecture
label identifies only part of `S`.

**Semantics-preserving map.** `T : Θ → Θ̃` with `F̃(T(θ)) = F(θ)` for all declared `θ` and all
inputs in a declared domain `D_in`.

**Transport-degenerate pair.** A pair whose parameter map is the identity on a shared coordinate
system. Every transport level is satisfied vacuously and `δ_k ≡ 0` up to rounding. **A
transport-degenerate pair may not be used as evidence for or against H1, H2, H3, H5, or H6.**

**Equivalence levels** E0 (symbolic) > E1 (exact finite-precision, all representable inputs) >
E2 (deterministic adversarial audit suite) > E3 (distributional, within tolerance) > E4 (task-metric
only). **Every claim must state its level and its declared domain.** A level may never be asserted
globally when a domain restriction applies.

**Transport levels** T0 (nominal hyperparameters only) → T5 (matched stopping, selection, search,
precision, and resource contracts).

**One-step covariance defect.** `δ_k = d(T̄(U_k(s_k)), Ũ_k(T̄(s_k)))` over the joint
parameter-optimizer state, for a declared metric `d`. Both a parameter-space metric and a
predictive-space metric must be reported; the predictive one is primary because the parameter one is
coordinate dependent.

**Accumulated geometric residue.** `G_K = Σ_i δ_i · Π_{j>i} λ_j(v_j)` where `λ` is a measured local
directional amplification, not a global Lipschitz constant.

---

## 3. Hypotheses

- **H0 — no useful law.** Measured covariance defects do not predict practical divergence better
  than simple baselines. Architecture gaps remain idiosyncratic.
- **H1 — exact transport collapse.** Under full transport, semantics-equivalent systems produce
  indistinguishable predictive trajectories.
- **H2 — optimizer geometry.** Native practical optimizers create systematic divergence across
  equivalent parameterizations, predictable from update covariance defect.
- **H3 — prior and regularizer.** Initialization and regularization transport explain more gap than
  optimizer update geometry.
- **H4 — numerical implementation.** Kernel choice, dtype, and normalization dominate in long
  sequential systems.
- **H5 — local stability.** Equal one-step defects yield different final gaps because trajectory
  amplification differs; Lyapunov-weighted defect predicts the difference.
- **H6 — quotient metric.** Function-space or Fisher-horizontal metrics generalize better than
  Euclidean parameter metrics.
- **H7 — architecture residual.** A residual difference survives all known transport, because the
  systems were not function-class equivalent or an unmodeled component differs.
- **H8 — broad generality.** One frozen estimator predicts across all sealed families.
- **H9 — artifact.** The relationship holds only for complex-to-real maps, or only at Q-Neuro scale.
- **H10 — audit utility.** Even with no quantitative law, the compiler materially improves the
  validity of architecture comparisons.

H0, H9, and H10 are **acceptable outcomes**, not failures. See §9.

---

## 4. The equivalence ladder

Families are ordered by increasing risk and decreasing analytic tractability. **The order is frozen.
A later rung may not be attempted before the preceding rung's gate passes.** Rationale: the
measurement system must be shown able to distinguish true zero defect, purely numerical defect,
optimizer-induced defect, regularizer-induced defect, initialization-prior defect,
stopping/search-induced defect, and genuine function-class difference — before it is pointed at
anything complicated.

| Rung | Family | Map | Degenerate? | Primary role |
|---|---|---|---|---|
| 1 | Hidden-unit permutation | Exact discrete symmetry `P` | No | **Zero-defect positive control** |
| 2 | Scaling orbit `W₂W₁ = (W₂/c)(cW₁)` | Exact continuous symmetry | No | Coordinate dependence, cleanly isolated |
| 3 | Dense vs factorized `W` ↔ `UV` | Non-injective | No | Implicit-bias-rich benchmark |
| 4 | Complex ↔ exact real block | `T = id` | **Yes** | Historical motivating case; H4 only |
| 5 | Unitary ↔ orthogonal real block | Structured | No | Long-memory, norm preservation |
| 6 | Fourier ↔ time domain | Exact, conjugate-symmetric | No | Regularizer transport under a basis change |
| 7 | State-space realization change | Similarity transform | No | Known transfer function, analytic ground truth |
| 8 | Attention / transformer gauges | Basis transformations | No | **Sealed until rungs 1–7 pass** |

Rung 1 is a **stop condition**: if the framework cannot make the permutation control approximately
exact under correctly permuted optimizer state, the measurement system is wrong and the program
halts for repair rather than proceeding.

Large transformers and language models are **sealed** and may not be touched under this
preregistration. A separate versioned amendment is required.

---

## 5. Experiment schedule

Registered prospectively. IDs are reserved now and may not be reused.

| ID | Purpose | Outcome-eligible |
|---|---|---|
| `QE-000001` | Equivalence spec + certificate interface; complex↔real certificate at E2/E3 on declared domain; pole-reachability measurement | No — instrument |
| `QE-000002` | Permutation zero-defect control, with and without optimizer-state permutation | No — control |
| `QE-000003` | Scaling orbit across SGD, momentum, Adam, AdamW, weight decay, clipping, and transported variants | Discovery |
| `QE-000004` | Dense vs factorized `W` ↔ `UV` | Discovery |
| `QE-000005` | Complex ↔ exact real block: numerical-implementation study (H4 only) | Discovery, H4 only |
| `QE-000006` | Native `torch.complex` family and its derived real map | Discovery |
| `QE-000007` | Optimizer-state transport implementation and audit | Discovery |
| `QE-000008` | Analytic microcosms: exact Jacobians, known Lipschitz constants, bound tightness | Discovery |
| `QE-000009` | Candidate estimator comparison on discovery families only | Discovery |
| `QE-000010` | **Estimator freeze record** (`research/laws/FROZEN_TRANSPORT_001.json`) | Freeze |
| `QE-000011` | Invalid-map negative control; certificate must reject | No — control |
| `QE-000012` | Sealed confirmation over held-out families | **Confirmatory** |

### 5.1 Split assignment, frozen

- **Fixtures** (debug only, cannot contribute to any estimate): tiny deterministic pairs in `tests/`.
- **Discovery families** (estimator development): permutation, scaling orbit, dense/factorized,
  complex↔exact-real, native-complex.
- **Sealed confirmation families** (opened once, at QE-000012): unitary↔orthogonal block,
  Fourier↔time-domain, state-space realization, quaternion↔four-channel real, and one
  normalization-reparameterization family that is **expected to fail** certification (Family L of the
  parent program) as a certificate-integrity check.
- **Independent replication**: attention gauge family plus at least one reimplementation in a second
  framework.

No family may move between splits after any outcome in it is observed.

### 5.2 Top-level unit

The top-level unit is an **independently specified equivalence problem within a transformation
family.** Training seeds are repeated measurements. Batches and examples are **not** independent
evidence and may not be used to inflate significance.

---

## 6. Endpoints

### 6.1 Primary integrity endpoint (co-primary)

Rate of **false equivalence certificates**: the compiler must reject every deliberately invalid map
in QE-000011 and the expected-to-fail confirmation family. Target: **zero false certificates.** A
single false certificate is a program-level failure requiring repair before any other endpoint is
reported.

### 6.2 Primary empirical endpoint

Out-of-family predictive accuracy of the **frozen** geometric-residue estimator for paired
predictive divergence, on sealed families. Frozen thresholds:

| Metric | Threshold |
|---|---|
| Out-of-family `R²` on log₁₀ predictive divergence | `≥ 0.50` |
| Sign accuracy of the predicted gap | `≥ 0.80` |
| **Magnitude accuracy**: median \|log₁₀(predicted / observed)\| | `≤ 0.5` (within a factor of ~3) |
| Bound coverage | `≥ 0.90` of observed divergences below the predicted bound |
| Improvement over every baseline in §6.4 | Required on **all** baselines |

The magnitude criterion is mandatory. **Sign-only prediction does not satisfy this
preregistration.** Failure of any single threshold rejects the general-law claim.

### 6.3 Zero-defect control thresholds (QE-000002)

- With optimizer state permuted correctly: paired predictive divergence must be statistically
  indistinguishable from a bitwise-identical rerun, and `max |Δ logit| ≤ 100 · eps_dtype · max|logit|`.
- With optimizer state deliberately **not** permuted: divergence must be detectable above that
  threshold. A control that cannot detect this deliberate defect is not a working instrument.

### 6.4 Baselines the estimator must beat

Frozen now: zero gap; mean discovery gap; architecture-family mean; optimizer-family mean; learning
rate; total gradient norm; training loss decrease; parameter count; condition number; raw one-step
prediction divergence; cumulative unweighted defect; cumulative stability-weighted defect.

Beating "raw one-step prediction divergence" is the hard case. If the elaborate estimator reduces to
it, that is a **negative result and must be reported as one** (§8).

### 6.5 Secondary endpoints

Stepwise logit / probability / loss / gradient / optimizer-state divergence; Fisher path length;
horizontal vs vertical update energy; directional amplification; final metric gap; calibration gap;
compute, memory, and energy gaps; certificate generation time; cross-framework reproducibility.

Secondary endpoints are corrected for multiplicity within named families and are **exploratory**
unless listed in §6.2.

---

## 7. Gates

| Gate | Condition to advance |
|---|---|
| **A — exactness** | Source→target map has a valid certificate at a stated level and domain. Downgrades are recorded, never hidden. |
| **B — conjugacy** | The first update discrepancy is eliminated or mathematically explained. |
| **C — non-vacuous bound** | Bound / observed divergence ratio `≤ 100` on analytic microcosms. Cannot be evaluated on a transport-degenerate pair. |
| **D — cross-family signal** | Frozen-candidate estimator beats every §6.4 baseline on `≥ 2` discovery families. |
| **E — frozen confirmation** | One attempt. No coefficient, form, family, seed, or threshold changes after opening. |
| **F — scale** | Advance beyond MacBook scale only if measurement cost and predictive value justify it. |
| **G — independent replication** | No law language without external implementation in a second framework. |
| **H — medical** | No patient-data work without the full governance gate. Unchanged from the parent program. |

---

## 8. Kill conditions

The `transport_covariance_conjecture` is **rejected**, and the rejection published, if any of the
following occurs:

1. Quotient-aware measurements add no predictive power over §6.4 baselines.
2. Learning rate or loss decrease predicts divergence as well as the estimator.
3. Transport residuals are numerically unstable across repeated measurement.
4. Bounds remain vacuous — ratio `> 100` — on analytic cases where every term is computable.
5. The relationship holds only for complex-to-real conversion.
6. Symmetry-orbit averaging (§10) provides no benefit over ordinary optimization.
7. Equivalent models diverge despite near-zero measured defect, showing the variable set is
   incomplete.
8. Non-equivalent models exhibit the same relationship, making equivalence irrelevant.
9. Prior art already contains the central theorem.
10. External groups cannot reproduce the measurements.

**Failed ideas are preserved under their original identifiers in `docs/FAILED_IDEAS.md` and
`research/failures.json`. A failed idea may not be renamed and rerun until it looks successful.**
Any re-attempt requires a new experiment ID, a versioned amendment, and an explicit statement of
what changed and why.

---

## 9. Acceptable outcomes

All of the following are scientific successes. The program does not require a law.

| Outcome | Description |
|---|---|
| **A** | The transport law fails; the Equivalence Compiler becomes a useful architecture-auditing tool. |
| **B** | Most supposed parameterization advantages reduce to a few known optimizer/regularization defects. |
| **C** | Genuine residual inductive biases are isolated that survive complete transport. |
| **D** | A frozen estimator predicts unseen function-space divergence across unrelated families and optimizers. **Major.** |
| **E** | A non-vacuous theorem connects measurable finite-step covariance defects to later predictive divergence, validated broadly. **Very major.** |
| **F** | Practical quotient-aware optimization removes arbitrary coordinate effects while preserving or improving learning at scale. |
| **G** | Architecture benchmarking changes: papers report equivalence certificates, transported controls, and residual advantage. |

Outcome A alone justifies the program. Outcomes D–G are not planned for and may not be assumed.

---

## 10. Deferred branch: symmetry-orbit learning

**Status: deferred. Not authorized by this preregistration.** Requires a separate amendment, and
may not begin until finite equivalence transport (rungs 1–3) is working.

Recorded now so the design is prospective rather than retrofitted:

For a symmetry group `G`, the orbit `{ T_g(θ) : g ∈ G }` may be infinite. The question is whether a
learner can integrate over an equivalence class rather than arbitrarily choosing one coordinate
realization — deriving an update on `[θ]` rather than on `θ`.

Constraints frozen now:

- **Do not literally instantiate millions or infinitely many models.** The claim of interest is
  analytic or sampled projection, not brute-force ensembling.
- **For compact groups with an invariant (Haar) measure**, group averaging or analytic projection is
  admissible.
- **For noncompact groups such as unrestricted scaling, do not write fake uniform integrals.** Use
  gauge fixing, quotient geometry, a normalized sampling scheme, or another mathematically valid
  construction, and state which.
- Convergence ladder to test: 1, 4, 16, 64, 256 sampled representations → analytic orbit projection
  where available → direct quotient-aware optimizer. The claim is that finite samples converge
  toward the analytic quotient-aware update.
- **"Quantum superposition" is not an available explanation.** Amplitude, Hilbert-space,
  interference, density-matrix, gauge, unitary, Hamiltonian, and symplectic mathematics may be used
  only where they produce an operationally distinct algorithm or a measurable invariant.

---

## 11. Quantum-inspired branch, purpose inverted

Quantum mathematics is retained **as an adversarial source of structured equivalences**, never as
branding and never as a claim about physics.

For each exotic representation: build it → compile it into an equivalent classical representation
where mathematically possible → transport training → measure what survives. **The surviving effect
is the object of study.**

Registered pairs: complex ↔ structured real; unitary ↔ orthogonal block; Hermitian ↔ structured
symmetric real; density matrix ↔ structured PSD real; quaternion ↔ four-channel structured real;
phase-sensitive readout ↔ paired-real bilinear readout.

Prohibited, unchanged from the parent program: calling magnitude-squared output a physical Born
measurement; treating a hidden vector as a wavefunction without operational reason; inferring
quantum cognition from a density-shaped tensor; claiming conservation from the word "Hamiltonian"
without a flow and integrator that preserve it.

---

## 12. The Equivalence Compiler

The compiler is the program's **primary deliverable**, and must be useful even if every theory here
fails.

Target interface:

```
qe audit <model_a> <model_b> [--transform SPEC]
```

Certificate contents, frozen as the required schema: semantic equivalence level; declared input and
parameter domains; dtype and device; parameter map and inverse with round-trip residuals; state map;
gradient transport residual; optimizer-state transport residual; one-step and multi-step conjugacy
residuals; numerical failure regions; regularizer mismatch; initialization mismatch; stopping
mismatch; search mismatch; compute mismatch; `transport_degenerate` flag; predicted functional
divergence; residual unexplained advantage; code and test hashes; environment; known failure modes.

Package: `qneuro.equivalence`. Design intent: a reviewer should be able to run this against a
published architecture claim. If a "new architecture" is largely a different coordinate
representation plus a friendlier optimizer interaction, the tool exposes that. If the architecture
still wins after the audit, **the claim is strengthened**, and the tool must say so as clearly as it
reports a reduction.

---

## 13. Artifact and reproducibility requirements

Binding on every QE run:

1. `schema_version` on every emitted artifact.
2. Per-step trajectories, probe logits, and defect series in compressed arrays (`.npz`), not JSON.
3. **Raw predictions preserved.** The absence of these is what made pole reachability
   retrospectively unanswerable for QN-000027 … QN-000042 and it may not recur.
4. Checkpoints for every paired run, both endpoints of the map.
5. Compute accounting: theoretical FLOPs, measured kernel time, optimizer-state bytes, activation
   bytes, peak resident memory, energy where reliable, total search cost.
6. Clean-worktree provenance and artifact hashing, as already enforced for the QN series.
7. Intention-to-run reporting: failed and excluded runs are reported, and failure rate is itself an
   adverse outcome. Mathematical-domain violations are separated from infrastructure failure.

---

## 14. Statistical design

- Power the study on **top-level equivalence problems**, not prediction rows.
- Hierarchical model with partial pooling over transformation family, task family, and framework;
  optimizer interactions permitted in discovery, fixed simple form in confirmation.
- Uncertainty: bootstrap over top-level problems; leave-one-family-out sensitivity; sign-flip tests
  where pairing permits; measurement-error sensitivity on amplification estimates.
- Multiplicity: **one** primary estimator and **one** primary held-out score. Everything else is
  secondary or exploratory.
- No estimator may be invented and confirmed on the same cells. Cross-fitting required for mediators.

---

## 15. Immediate execution order

Frozen. Steps 1–4 are complete at the time of freezing.

1. ~~Amend the interpretation of QN-000042 without rewriting history.~~ Complete.
2. ~~Formalize the exact status of the complex/exact-real relationship.~~ Complete —
   `EQUIVALENCE_SCIENCE_AMENDMENT_001.md` §2.
3. ~~Downgrade equivalence claims around native complex `tanh`.~~ Complete — §3, domain declared.
4. ~~Write this preregistration.~~ Complete.
5. Build the permutation zero-defect control (`QE-000002`). **Stop condition.**
6. Build the scaling-orbit experiment (`QE-000003`).
7. Build the dense/factorized experiment (`QE-000004`).
8. Promote the complex-real map into the first certificate (`QE-000001`).
9. Add the native `torch.complex` family (`QE-000006`).
10. Implement optimizer-state transport (`QE-000007`).
11. Measure the first true nonzero covariance defects.
12. Freeze candidate defect estimators on analytic microcosms (`QE-000009`, `QE-000010`).
13. Only then scale to broader model families.
14. Transformers and LLMs remain sealed.
15. Symmetry-orbit experiments only after finite equivalence transport works.

---

## 16. Amendment policy

This document may be amended only by appending a numbered, dated amendment file that states what
changed, why, and what evidence had been observed at the time. **Frozen thresholds, split
assignments, and the ladder order may not be changed after any outcome in the affected family is
observed.** An implementation correction requires a versioned amendment and a new experiment ID.

Optimize for a result that survives people trying to destroy it.
