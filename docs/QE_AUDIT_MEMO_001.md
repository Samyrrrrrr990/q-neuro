# QE audit memo 001

Requested by: `docs/CLAUDE_CODE_SHAFIEE_RESEARCH_PROGRAM.md` §27.2 ("produce an audit memo, not code")

Date: 2026-08-14

Audited commit: `a13061677035b5649109188d2125bb0d956c5fce`

Branch: `codex/q-neuro-falsification`

Status: audit only. No repository code was changed. No historical record was modified. No experiment
was registered.

Scope note: this memo evaluates the *readiness of the equivalence-transport program*, not any
scientific claim about Q-Neuro's architecture. Nothing here rescues, weakens, or re-labels the
existing falsification result.

---

## 0. §27.1 preconditions

| Precondition | Result |
|---|---|
| Parent commit | `a130616` — identical to the commit the handoff manifest was generated from |
| Worktree | Clean; single untracked file, the handoff document itself |
| Remote | `https://github.com/Samyrrrrrr990/q-neuro.git`, branch tracked |
| Full test suite | 146 passed, exit 0 |
| Environment | Python 3.12.13, torch 2.13.0, numpy 2.5.2, MPS available |
| Required documents | `CLAIMS.md`, `FAILED_IDEAS.md`, `NEXT_PHASE_AUDIT.md`, `PREREGISTRATION_NEXT_PHASE.md`, `PROVISIONAL_LAW_FREEZE.md`, `MATHEMATICAL_FRAMEWORK_V2.md`, `PRIOR_ART_V2.md`, `RESEARCH_LOG.md`, `ROADMAP.md` — read |
| Required artifacts | `research/claims.json`, QN-000033, QN-000040, QN-000042, QN-GRAND-001 preflight — inspected |

The manifest in Appendix A of the handoff is valid against the current tree.

---

## 1. Headline finding: the first instrument does not have the property the program needs

**`EXACT`. The Q-Neuro complex/exact-real pair is not two parameterizations of one function class.
It is one parameterization with two implementations. The parameter map `T` is the identity.**

Evidence:

- `ComplexOperatorState` declares twelve **real-valued** `nn.Parameter`s — `initial_real`,
  `initial_imag`, `injection_real`, … (`qneuro/models/operators.py:127-138`). Complex tensors are
  constructed *inside* `evolve`/`measure` by `torch.complex(...)`
  (`qneuro/models/operators.py:158-160`). No complex tensor is ever a leaf parameter.
- `ExactRealBlockOperatorState` declares parameters with **identical names, identical shapes, and
  identical initializer standard deviations** (`qneuro/models/equivalent.py:49-78`).
- `copy_from_complex` is a name-matched `parameter.copy_()` loop
  (`qneuro/models/equivalent.py:80-92`) — a memcpy, not an algebraic transform.

Measured confirmation (audit script, scratchpad only):

```
complex model parameter dtypes : ['torch.float32']
any torch.is_complex parameter : False
name sets equal                : True
shapes equal                   : True
max |init difference| same seed : 0.0        (564 / 564 parameters)
```

The training harness then closes the remaining gaps by itself.
`train_with_validation_tuning` calls `set_seed(seed)` immediately before `build_model`
(`experiments/run_generator_shift.py:59-61`), and builds the train loader with
`make_loader(train_cases, batch_size, True, seed)` (`:71`), whose `DataLoader` generator is
`torch.Generator().manual_seed(seed)` (`experiments/run_experiment_zero.py:50`).

Therefore, in every historical run, the complex and exact-real models shared:

- bitwise-identical initial parameters (not merely a matched distribution);
- identical minibatch order;
- the same optimizer (`AdamW`), learning rate, weight decay, and `clip_grad_norm_(5.0)`
  (`experiments/run_experiment_zero.py:152-181`);
- the same early-stopping rule and the same coordinates for weight decay to act in.

### 1.1 Consequence for the transport ladder

The handoff's transport levels T0–T5 (§5.4) **collapse to a single point** on this pair. The
historical QN-000040/42 runs were already at approximately T4–T5 *under the identity map*, without
anyone implementing transport. Specifically:

- **Optimizer-state transport is already exact.** AdamW's `exp_avg` and `exp_avg_sq` live in the
  same real coordinates on both sides, so the conjugacy condition `Ũ∘T̄ = T̄∘U` is satisfied by
  `T̄ = id` whenever the gradients agree.
- **Regularizer transport is already exact.** Decoupled weight decay acts on identical coordinates,
  so §6.14's coordinate-dependence problem does not arise here.
- **Initialization transport is already exact**, and is not distinguishable from the "native
  initialization" arm of §6.15, because both models' `reset_parameters` draw the same values in the
  same order.

**`INFERRED`: QE-000002 through QE-000006 as specified in §21 are near-vacuous on this pair.** They
would measure float32 rounding, not optimizer geometry, because there is no non-identity map whose
covariance defect could be non-zero. They cannot discriminate H2 (optimizer geometry), H3
(prior/regularizer), H5 (local stability), or H6 (quotient metric), because the interventions those
hypotheses require do not exist between these two objects.

What the pair *does* exercise is **H4 (numerical implementation)** — and only H4. That is a real
hypothesis in the handoff's own list, and it is worth studying, but it must be labelled correctly.

---

## 2. Measured conjugacy: the smallest deterministic experiment, already run

Two models, `state_dim=8`, `rank=2`, 24-case fixed batch, full-batch AdamW(1e-3, wd 1e-4),
grad-clip 5.0, no data shuffling, 60 steps. Maximum absolute divergence across all parameters:

| dtype | step | Δlogit | Δloss | Δgrad | Δexp_avg | Δexp_avg_sq | Δparam |
|---|---:|---:|---:|---:|---:|---:|---:|
| float32 | 1 | 1.9e-06 | 0.0 | 2.4e-06 | 1.2e-07 | 2.2e-09 | 7.5e-09 |
| float32 | 10 | 3.8e-06 | 0.0 | 1.5e-06 | 1.8e-07 | 7.5e-09 | 6.0e-08 |
| float32 | 60 | 9.5e-06 | 6.0e-08 | 3.9e-06 | 5.0e-07 | 4.1e-08 | 4.4e-07 |
| float64 | 1 | 7.1e-15 | 0.0 | 1.1e-14 | 5.6e-17 | 2.6e-18 | 1.4e-16 |
| float64 | 60 | 9.8e-15 | 1.1e-16 | 3.1e-15 | 9.4e-16 | 2.8e-17 | 2.2e-16 |

`OBSERVED`: divergence appears at step 1 at the float32 rounding scale, and grows roughly linearly —
about 5× over 60 steps — with no exponential amplification at this horizon. Moving to float64 drops
every quantity by ~9 orders of magnitude, which is what pure rounding predicts and what a genuine
mathematical discrepancy would not.

This is the answer to §27.2's "smallest deterministic conjugacy experiment", and it also pre-answers
Phase 1's "identify the first nonzero discrepancy": **the first discrepancy is at step 1, in the
forward pass, at float32 epsilon, attributable to einsum/activation evaluation order.**

**Implication for Gate C.** The handoff requires a bound within a prespecified multiplicative range
of *observed* predictive divergence. On this pair the observed divergence is ~1e-5 in float32 and
~1e-14 in float64. Any Lipschitz-product bound will be many orders above that, so **Gate C cannot be
passed using Family A.** A pair with a genuinely non-zero defect is required.

---

## 3. Where equivalence actually breaks (Gate A downgrade)

`EXACT`. The two implementations of complex `tanh` are **not** the same function on the full input
domain, by construction.

`_complex_tanh_real_pair` (`qneuro/models/equivalent.py:13-20`) computes
`tanh(r+ii) = [sinh 2r + i sin 2i] / [cosh 2r + cos 2i]` and then **clamps the denominator** to
`torch.finfo(dtype).eps`. Native `torch.tanh` on a complex tensor applies no such clamp. Near the
pole of `tanh` at `i·π/2`, `cosh(2r) + cos(2i) → 0`, and the two paths separate completely:

| Re(δ) offset from pole | native re | native im | realified re | realified im |
|---:|---:|---:|---:|---:|
| 1e-1 | 1.0033e+01 | -4.36e-06 | 1.0033e+01 | -4.36e-06 |
| 1e-3 | 9.8690e+02 | -4.31e-02 | 9.8690e+02 | -4.31e-02 |
| **1e-4** | **inf** | **-inf** | **1.6777e+03** | **-7.33e-01** |
| **0** | **nan** | **-inf** | **0.0000e+00** | **-7.33e-01** |

Mechanism: at offset 1e-4 in float32, `cosh(2e-4)` rounds to exactly `1.0` and `cos(π)` to exactly
`-1.0`, so the native denominator is exactly `0.0`. The realified path clamps to `1.19e-7` and
returns a finite (but arbitrary) value.

**Therefore the pair is not at equivalence level E1** ("exact finite-precision forward equivalence
for all representable inputs"). It is at **E2/E3 on the reached domain**. §29 Gate A requires this
downgrade to be recorded, not hidden — this memo is that record. The eventual certificate must carry
an explicit **input-domain restriction** excluding a neighbourhood of `Re(δ)≈0, Im(δ)≈π/2`.

**Important caveat, stated plainly:** I have **not** shown that training ever reaches this region.
It cannot be checked retrospectively, because no checkpoints and no raw activations exist for the
falsification-phase runs (see §5, Q5). Measuring the closest approach of `δ` to the pole is a
mandatory QE-000001 measurement, not an established fact. Until measured, the correct statement is
that a *known* equivalence-degradation region exists whose *reachability is unknown*.

---

## 4. Reinterpretation of the QN-000042 headline statistic

This is the one place where the audit touches a published number. The number is correct; its
*interpretation* overcounts independent evidence.

Decomposing the 1,920 held-out paired effects by which model was actually `best_real`:

| best_real model | cells | mean difference | exactly zero |
|---|---:|---:|---:|
| `exact_real_block_operator` | 1,478 (77%) | +0.00000 | 1,478 (100%) |
| `real_polar_operator` | 442 (23%) | −0.03978 | 60 (13.6%) |
| **all** | **1,920** | **−0.00916** | **1,538** |

In 77% of cells the "best real model" *is* the bitwise-coupled exact-real implementation, and the
difference is exactly zero in every one of them. Given §1, a tie there is closer to a
self-consistency check than to an architecture comparison — it is very nearly definitional.

The entire −0.00916 effect is carried by `real_polar_operator` in the remaining 442 cells. That is a
legitimately different model (polar coordinates, different function class, **not** an equivalence
map), so **the negative result itself stands.** What does not stand is the implication that
"0/1,920 positive effects" represents 1,920 independent falsification events.

QN-000040 is materially healthier and should carry more of the argument:

| best_real model | cells | mean difference | exactly zero |
|---|---:|---:|---:|
| `exact_real_block_operator` | 1,391 (48%) | +0.00000 | 1,391 (100%) |
| `state_space` | 637 | −0.09063 | 19 |
| `real_polar_operator` | 485 | −0.04687 | 47 |
| `gru` | 367 | −0.07073 | 19 |

Here genuinely distinct models win 52% of cells with substantial negative means.

**Recommended action — amendment, never rewrite** (§1.3). `RESULTS.md:24` already discloses that
1,538 effects were exactly zero, which is good practice. Two surfaces do not:

- `README.md:23-24` — presents "0/1,920 positive effects" and "top-1 matches in all 1,920 cells"
  adjacently without noting that the second largely *causes* the first;
- `research/claims.json:33-34` — same framing in the machine-readable ledger.

The fix is an added amendment note recording the best-real decomposition, not an edit to any frozen
result artifact.

---

## 5. The eight required questions

**Q1 — Which exact maps already exist?**
Exactly one, and it is the identity: `ExactRealBlockOperatorState.copy_from_complex`
(`qneuro/models/equivalent.py:80-92`), source→target only. There is no inverse
(`copy_to_complex` does not exist), no composition, no map object, no certificate. The realification
algebra is *implemented* correctly in `evolve_pair`/`forward` and I verified it term-by-term against
`ComplexOperatorState` (projection, delta, and the conjugate readout `w*z` all match), but it is
hand-written inline, not expressed as a reusable `ParameterMap`. `RealRotationBlockOperator`,
`RealPolarOperatorState`, and `TwoChannelRealOperatorState` are **controls, not maps** — they are
not semantics-preserving and must never be registered as equivalences.

**Q2 — Which maps lack optimizer transport?**
All of them, in the sense that no `OptimizerStateMap` exists anywhere in the codebase. But for the
only map that does exist, optimizer transport is *already satisfied by the identity* (§1.1). The
correct statement is: **the repository has no optimizer transport code and does not yet need any**,
and will need it the moment a non-identity map is introduced.

**Q3 — How are complex parameters represented by the installed PyTorch optimizer?**
They are not. The optimizer never sees a complex tensor — every leaf parameter is float32 (§1). The
handoff's §11.3 and §28.6 assume PyTorch's complex-parameter handling (the internal
`view_as_real` path in Adam, and the question of whether `exp_avg_sq` is computed per real component
or per complex modulus) is on the critical path. **On this codebase it is never exercised.** That
question is still worth answering — it is a real source of optimizer non-covariance — but it
requires *first building a model with genuine `torch.complex` parameters*, which does not exist here.
This is a design decision the program should make deliberately rather than inherit.

**Q4 — Which historical paired seeds can be reconstructed?**
All of them, at config level. `experiments/results/QN-000042/config.yaml` records 32 world seeds, 5
training seeds, train/validation/test/counterfactual seeds, learning rates, weight decay, epochs,
patience, batch size, and `device: cpu`. Combined with `set_seed` and the seeded loader generator,
runs are deterministically **re-runnable**.
They are not **recoverable**: `train_one` keeps `best_state` in memory and loads it back into the
model (`experiments/run_experiment_zero.py:199-209`) but never writes it to disk. Checkpoints exist
for QN-000007 … QN-000021 only. **The entire falsification phase (QN-000027 … QN-000042) has no
saved weights.** Every trajectory the QE program needs must be regenerated, not mined.

**Q5 — Which raw predictions are missing?**
All of them. QN-GRAND-001's own preflight gate `raw_predictions_preserved` records
`"Prediction artifacts found: []"` and is one of the six blocking failures. Result files contain
only aggregated per-cell `metrics` dicts (`top1`, `nll`, `ece`, `order_accuracy`) keyed by
(family, model, world_seed, severity, train_size, training_seed). There are no logits, no
probabilities, no per-example predictions anywhere in the tree. `NEXT_PHASE_AUDIT.md:149-151`
already flagged that this blocks retrospective hierarchical bootstrap; it equally blocks every
per-step defect measurement in §18.4.
Per-*epoch* history (`train_loss`, `validation_nll`, `validation_top1`) *is* retained in
`training_runs[].selected_trial.resources.history` — coarse, but the only trajectory data that exists.

**Q6 — Which artifact schemas should be migrated?**
`experiments/results/*/metrics.json` are monolithic JSON, up to 10 MB (QN-000040), 65 MB total, and
carry **no `schema_version` field** — unlike `research/claims.json`, `research/failures.json`,
`research/laws/FROZEN_CANDIDATE_001.json`, and `release/manifest.json`, which all do. Three changes
are needed before QE runs begin, per §20.6:
1. add `schema_version` to the results schema (new QE runs only; historical files stay frozen);
2. move per-step trajectories, probe logits, and defect series to compressed arrays (`.npz`), not JSON;
3. add a separate raw-prediction store so the `raw_predictions_preserved` gate can actually pass.

**Q7 — What is the smallest deterministic conjugacy experiment?**
Already run — §2. Two models at `state_dim=8, rank=2` (564 parameters), one fixed 24-case batch,
full-batch AdamW, no shuffling, 60 steps, float32 and float64, logging max-abs divergence of logits,
loss, gradients, `exp_avg`, `exp_avg_sq`, and parameters. It runs in seconds on the M2 and needs no
task generator. This should become QE-000001's training-free core plus a 60-step extension; the
script is in the session scratchpad and should be rewritten as a test, not copied.

**Q8 — What prior work most threatens novelty?**
In descending order of threat:
1. **Tan et al., NeurIPS 2022** — real-valued backpropagation reduces infinite-width complex training
   dynamics to ordinary real dynamics for many activations. This is close to S0/H1 *for exactly this
   complex-real pair* and is already logged in `PRIOR_ART_V2.md`.
2. **Zhao, Walters & Yu, TMLR 2026** (symmetry in NN parameter spaces) and **Wang & Wang, NeurReps
   2025** (gauge fiber bundle geometry of transformers) — the handoff's own §8.5 concedes these put
   the novelty bar high for "quotient-space learning" and "neural gauge analysis". Family H
   (attention gauges) should be treated as validation, never as a novelty claim.
3. **Kristiadi, Dangel & Hennig, NeurIPS 2023** — geometry under reparametrization; directly covers
   the coordinate-dependence-of-sharpness argument in §6.14/§15.6.
4. **Amari 1998** + **Song et al., ICML 2018** (higher-order invariance) — natural-gradient
   reparameterization invariance and its finite-step failure are established; S1 must be positioned
   as an *estimator* contribution, not an invariance discovery.
5. **Arjovsky, Shah & Bengio, ICML 2016** — the realification identity itself, used for the same
   implementation purpose. `PRIOR_ART_V2.md:12-15` already concedes this correctly.

The defensible novelty window remains §8.7's *integration* claim (compile → transport → measure →
bound → predict → benchmark). Nothing in this audit strengthens it; §1 slightly weakens the "Q-Neuro
as first case study" framing, because the case study turns out to exercise only the numerical rung
of the ladder.

---

## 6. Recommended revision to the experiment sequence

The handoff's §21 ordering assumes Family A has a non-trivial map. It does not. Proposed correction,
preserving the handoff's gates and intent:

**Keep as specified**
- **QE-000001** (forward/state/loss/gradient certificate) — well-posed, and now has a concrete
  adversarial target (§3) rather than a generic one. Must additionally measure how close trained
  `δ` gets to the `tanh` pole.
- **QE-000007** (numerical stress across dtype/length/device) — this is no longer a side-study. Given
  §1, it is the *only* QE experiment on Family A that measures a real effect, and it is the correct
  home for H4.
- **QE-000010** (permutation zero control) and **QE-000011** (invalid-map negative control) — both
  test the compiler rather than the pair, so both are unaffected.

**Re-target**
- **QE-000002 … QE-000006** (SGD conjugacy, AdamW state audit, transported AdamW, regularization
  ladder, stopping ladder) — on Family A these measure float32 noise. Either re-target them at a
  pair with non-trivial `T`, or keep them on Family A and relabel them honestly as an H4
  numerical-implementation study with no optimizer-geometry interpretation. QE-000006 is the partial
  exception: early stopping *can* amplify 1e-5 differences into discrete checkpoint differences
  (§23.3, Theorem T9), so it retains meaning even here — and the `1e-5` improvement threshold in
  `train_one` is the same order as the observed divergence, which is worth measuring directly.

**Promote**
The program needs a first instrument whose defect is non-zero. The cheapest ones are already in the
handoff's own §28.11 toy list and need no task generator:
- **hidden-unit permutation** (Family G) — exact discrete symmetry, expected zero defect *only if*
  optimizer state is permuted too; the true positive control;
- **homogeneous scaling orbit** (Family F) — `W₂W₁ = (W₂/c)(cW₁)`; large Euclidean defect, zero
  predictive defect. This is the cleanest available demonstration that parameter-space metrics
  mislead, and it directly instantiates counterexample §23.2;
- **matrix factorization** `W = UV` (Family E) — non-trivial, non-injective, and the first case where
  optimizer transport is genuinely undefined rather than trivial.

These three should sit between QE-000001 and the optimizer audit. Only after one of them shows a
measurable, non-vacuous defect does Gate C become evaluable at all.

---

## 7. What was not done

- No code was written or modified. §27.3 explicitly limits the first code change to the equivalence
  specification and certificate interface; that has not been started.
- No historical result, preregistration, claim, or figure was altered.
- The conjugacy and pole probes ran from the session scratchpad against the installed package. They
  produced no repository artifact and are not registered experiments.
- No QE experiment ID has been allocated. `QE-000001` remains unclaimed.
- The §4 amendment is *recommended*, not applied — amending the claim ledger is a scientific
  decision for the author, not an audit action.

---

## 8. Immediate next actions, in order

1. Decide §4: accept or reject the recommended best-real decomposition amendment to `README.md` and
   `research/claims.json`.
2. Decide §6: accept, reject, or modify the re-sequencing before any QE ID is allocated.
3. Decide Q3: whether the program should introduce a genuinely `torch.complex`-parameterized model.
   This determines whether the PyTorch optimizer questions in §28.6 are in scope at all.
4. Only then begin §27.3 — the equivalence specification and certificate interface, failing tests
   first.

Gate status at the close of this audit: **Gate A downgraded to E2/E3 with a declared domain
restriction (§3). Gate B satisfied on Family A but trivially, by the identity map (§1). Gate C not
evaluable on Family A (§2).**
