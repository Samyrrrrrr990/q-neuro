# Q-Neuro next-phase scientific audit

Date frozen: 2026-08-13  
Repository state audited: `v0.1.0` / `df10ed6989ade31effc853118050f82fdc292ba8`  
Audit status: pre-experimental; no next-phase outcome has been observed  
Scope: computational research on synthetic data; no clinical inference

## Audit decision

Q-Neuro has a credible **within-simulator phenomenon** and a substantially better negative-result
record than most exploratory repositories. It does not yet establish an intrinsic complex-valued
advantage, a general computational law, or external validity.

The strongest current result is QN-000008: across five unseen NeuroWorld seeds, the complex
operator exceeds the tested two-channel real control by 0.0538–0.0627 absolute top-1 accuracy over
four jointly varied shift severities. The analysis first averages training seeds within world and
uses world seed as the top-level unit. QN-000016 then finds directional losses after retraining a
commutative accumulator, a magnitude-only readout, and a no-negative-evidence variant.

Those results justify a falsification phase, not a superiority claim. The two-channel control is
not the exact real block-matrix representation of the complex recurrence; compute is not matched;
optimizer-search budgets are unequal; only five closely related worlds support the leading
interval; shift dimensions are entangled; and no independent task generator or external dataset
has been evaluated. H0, H1, H2, H5, H6, and H7 therefore remain open. H3 and H4 have preliminary
mechanism evidence but have not survived the interventions required to identify them.

The project must not be described as Outcome E. If forced into the final decision taxonomy using
only current evidence, it is provisionally closest to **Outcome B inside NeuroWorld**, with the
important qualification that stronger controls could still move it to Outcome A or C.

## Material reviewed

This audit covered:

- all model implementations in `qneuro/models/` and construction logic in
  `qneuro/model_factory.py`;
- NeuroWorld generation, counterfactuals, ambiguity, composition, omitted-label, and hidden-
  syndrome tasks;
- every experiment runner and YAML configuration;
- the SQLite schema and all QN-000001–QN-000026 result directories;
- all generated paired analyses and discovery artifacts;
- the research log, results narrative, mathematical specification, model card, claim ledger,
  failure ledger, prior-art ledger, architecture notes, and roadmap;
- all 36 current tests and the CI/build targets;
- the complete canonical manuscript source, generated tables/figures, and manuscript metadata.

All tracked JSON artifacts parsed successfully. At the audited commit, 36 tests pass, Ruff lint
and formatting checks pass, and `git diff --check` is clean. These checks establish repository
integrity; they do not cure the scientific-design limitations below.

## 1. Strongest current evidence

### 1.1 Chronology impossibility control

QN-000003 is the cleanest conceptual result. Chronology twins have identical aggregate evidence
and demographic context but opposite labels determined by marker order. The unordered MLP obtains
zero counterfactual-pair accuracy, while the tested ordered models solve the pairs. This validates
the benchmark's basic order intervention and demonstrates that an order-free representation
cannot solve the constructed problem without leakage.

This result supports ordered computation, not complex computation. Real operators, the
Transformer, GRU, and diagonal state-space controls also learn chronology.

### 1.2 Multi-world shifted robustness signal

QN-000008 is the strongest comparative evidence. At 1,000 training cases, complex-minus-two-
channel top-1 differences are:

| Shift | Mean paired world difference | 95% Student-t interval | Top-level n |
|---|---:|---:|---:|
| Nuisance | +0.0627 | [+0.0552, +0.0702] | 5 worlds |
| Mild | +0.0609 | [+0.0502, +0.0715] | 5 worlds |
| Moderate | +0.0602 | [+0.0529, +0.0674] | 5 worlds |
| Severe | +0.0538 | [+0.0471, +0.0606] | 5 worlds |

The direction is stable across the declared seeds and severities. The result is not a case-level
pseudoreplication: training seeds are averaged within world before the across-world interval.
This is a meaningful improvement over the earlier three-training-seed summaries.

### 1.3 Retrained mechanism interventions

QN-000016 is stronger than post-training phase scrambling because each ablation is retrained.
Across three moderate-shift worlds, the full complex model exceeds:

- the commutative complex accumulator by +0.232 top-1;
- the magnitude-only constructive readout by +0.104 top-1;
- the no-negative-evidence variant by +0.072 top-1.

All three directions repeat by world, but the exact two-sided sign-flip p-value is necessarily at
least 0.25 with three worlds. The accumulator removes order and state-conditioned multiplication
together, the readout ablation changes both function class and optimization, and the evidence
ablation deletes information. These are useful interventions but not yet one-factor causal
identification.

### 1.4 Evidence against the preferred architecture

Several negative results are unusually valuable and should remain primary outcomes:

- the tuned GRU dominates the lowest-data source regime;
- the real operator is substantially better on irreducible ambiguity;
- source-fitted temperature scaling worsens shifted calibration;
- omitted-class and hidden-syndrome separation are not uniquely complex;
- expected-information acquisition is not universally beneficial;
- density rank, generic dissipation, phase-gradient optimization, local pretraining, and the
  fixed factor graph do not improve the relevant frontier;
- hard halting degenerates to uniform two-step truncation;
- conventional states are at least as linearly probe-accessible for several simulator factors.

These failures narrow the claim and make the next phase falsifiable.

### 1.5 Engineering and auditability strengths

The repository has deterministic seeds, explicit missingness, causal counterfactual pairs,
never-reused numeric run directories, stored configurations, per-run summaries, environment
metadata, parameter accounting, tests for mathematical invariants, generated figures/tables,
negative-result documentation, a model card, and a synthetic-only safety boundary. QN-000024 is
retained as a failed run. The paper distinguishes phase use from comparative advantage and avoids
physical quantum claims.

## 2. Weakest current evidence

### 2.1 No external validity

All performance evidence comes from one simulator implementation. New `world_seed` values alter
templates inside the same code path; they do not constitute independently designed environments.
There are no non-medical independent generators, real-world nonclinical datasets, external
implementations, or independent replications. H6 and H7 are completely unresolved.

### 2.2 No discovered computational law

The code can report learned operator commutator norms, but no experiment varies a task-level
order-dependence quantity independently of other difficulty factors and predicts held-out complex
advantage. QN-000026 ranks existing candidates; it does not discover or confirm a law. There is no
discovery/confirmation data split for candidate equations, no held-out task-family prediction,
and no frozen threshold or functional form.

### 2.3 Mechanistic exclusivity is not shown

The current evidence says that the implemented complex parameterization uses ordered composition
and phase-sensitive measurement. It does not say that complex arithmetic uniquely supplies the
useful mechanism. An exact real block implementation, an unrestricted paired-real implementation,
real rotation blocks, polar state, orthogonal recurrence, and commutator-controlled real models
are absent.

### 2.4 Small top-level samples

The leading comparison has five worlds; most mechanism comparisons have three. Student-t
intervals at those sample sizes are fragile, and the many exploratory contrasts are not adjusted
for multiplicity. The result artifacts contain aggregated metrics rather than per-case predictions,
which prevents retrospective hierarchical bootstrap and some diagnostic reanalysis without
retraining.

### 2.5 Exploratory paper framing

The current manuscript is candid and internally coherent, but its central robustness result was
developed through several rounds of simulator and architecture exploration. A configuration
committed before a run is not equivalent to a prospective, immutable preregistration. The paper
cannot present the existing sweep as the required untouched grand confirmation.

## 3. Most dangerous confounds

### 3.1 Inexact real equivalence

At the QN-000008 budget, the complex model has 24 complex state coordinates (48 real degrees of
freedom) and 20,304 trainable real scalars. The two-channel control has a 45-dimensional flat real
state and 19,975 scalars. Parameter counts are close, but the transition families are not exact
reparameterizations. Complex multiplication and conjugation correspond to structured real block
matrices; the existing control instead learns a different low-rank flat-real update. A positive
gap may be caused by block structure, effective rank, tying, conditioning, or initialization.

### 3.2 Compute and tuning mismatch

QN-000008 gives the GRU three learning-rate trials and the Transformer two, but gives each
operator one. This favors two conventional controls but does not establish a common search budget.
The complex model takes roughly 6.6–7.1 CPU seconds per selected trial versus about 4.2 for the
two-channel control, so the primary result is parameter-matched but not compute-matched. FLOPs,
optimizer steps to threshold, isolated peak memory, GPU latency, and energy proxies are absent.

### 3.3 Entangled shift severity

The mild-to-severe path simultaneously changes observation probability, probability mixing,
temporal jitter, and chronology-marker visibility. It does not identify which shift causes the
gap, whether effects interact, or whether robustness is monotone in any one structural property.
The next phase must factorially separate shift families and include commutative negative controls.

### 3.4 Simulator-specific temporal signatures

NeuroWorld assigns label-specific evidence stages. For chronology twins this is intentional, but
other orderings can also encode label through template stages. The GRU's source success and seed-
shift collapse show that exploitable simulator timing signatures exist. The complex model may be
using a more stable version of the same shortcut rather than a general relational principle.

### 3.5 Correlated evaluation construction

Every shifted world is sampled with the same case RNG seed. This intentionally pairs class draws
and reduces Monte Carlo noise, but it also correlates evaluation sets across worlds. World
templates are different; observations are not fully independent top-level replications. A future
analysis must state the pairing explicitly and either use independent case seeds or model the
cross-world pairing.

### 3.6 Development-set reuse at the research-program level

The same simulator family, task motifs, training size, and shift direction informed successive
model and ablation choices. Test labels are not used inside an individual optimizer, but research
decisions have repeatedly observed similar held-out environments. Only a sealed confirmatory task
bank can address this adaptive overfitting.

### 3.7 Normalization and measurement bundle

Every operator step projects the state to a fixed norm. The complex model also couples a complex
analytic nonlinearity, conjugate low-rank update, and magnitude-squared readout. Current ablations
do not independently vary normalization, transition algebra, rank, state degrees of freedom,
readout, and initialization. The apparent phase effect may be a bundled parameterization effect.

## 4. Strongest alternative explanations

| Hypothesis | Current assessment | Decisive next test |
|---|---|---|
| H0: no intrinsic advantage | Open | Equal-search exact and unrestricted real controls eliminate or retain the aggregate gap. |
| H1: effective capacity | Open | Match real degrees of freedom, effective matrix rank, parameters, and readout exactly; run state/rank sweeps. |
| H2: optimization geometry | Open | Initialization grid, optimizer grid, convergence/gradient diagnostics, and exact-real forward equivalence. |
| H3: noncommutativity | Preliminary | Independently vary task commutator/order information and force/penalize learned commutators. |
| H4: phase memory | Preliminary | Destroy phase during training and inference; compare real polar and SO(2)-block models. |
| H5: implicit regularization | Untested | Match functions at initialization and measure norms, margins, sharpness, stability, and data scaling. |
| H6: NeuroWorld artifact | Leading threat | Separately authored generators and simulator shortcut audit. |
| H7: general phenomenon | Untested | Freeze the law, then predict outcomes on untouched non-medical task families and datasets. |

The simplest boring explanation is currently: **the constrained complex block parameterization is
a useful regularizer for one family of synthetic temporal shifts, while the tested real control
does not encode the same structure as efficiently.** This explanation is compatible with the
data and must be tested before any deeper claim.

## 5. Missing baselines

The current suite includes MLP, GRU, one-layer bidirectional-style Transformer encoder with pooled
readout, diagonal state-space, real operator, paired-real readout, and several exploratory laws.
It still lacks the following mandatory controls:

1. vanilla RNN;
2. LSTM;
3. compact causal Transformer with last-token/causal readout;
4. unrestricted dense real matrix recurrence;
5. exact real block-matrix implementation of the complex model;
6. untied/unrestricted paired-real version of that block model;
7. real SO(2) rotation-block recurrence;
8. real polar magnitude/angle state;
9. orthogonal or Cayley-parameterized real recurrence;
10. residual gated recurrent model;
11. a stronger modern state-space sequence baseline suitable for longer horizons;
12. fixed two-state attractor, as already implied by QN-000023;
13. ambiguity-aware real and complex heads with identical objectives;
14. label-prior, length-only, order-only, metadata-only, single-feature, tree, and nearest-neighbor
    exploitative controls.

Each complex candidate needs three comparisons: exact functional real equivalent, parameter-
matched unrestricted real control, and compute-matched strongest sequential control.

## 6. Statistical weaknesses

- No prospective next-phase primary endpoint or minimum practically meaningful effect is frozen.
- The current five-world result is underpowered for heterogeneous shift-family effects.
- Most ablations use three worlds and cannot yield a two-sided exact sign-flip p-value below 0.25.
- Repeated Student-t summaries are duplicated across analysis scripts rather than centralized in a
  tested statistical module.
- No power simulation, hierarchical bootstrap, mixed-effects model, probability of superiority,
  rank-stability analysis, worst-case aggregate, or robustness-curve area is implemented.
- No familywise or false-discovery correction is declared for the large model/metric matrix.
- ECE uses ten bins without uncertainty and is insufficient as the sole calibration error.
- World-level effects are based on summaries; raw predictions and identifiers are not retained.
- Training seeds are not independent task replications and must never be promoted to the main
  statistical unit for simulator generalization.
- Hyperparameter selection variance is not propagated into comparative uncertainty.
- Missing/failing runs do not yet have preregistered exclusion and intention-to-treat rules.

The next phase must implement `research/statistics.py`, test it on distributions with known
answers, and freeze the inferential hierarchy before QN-GRAND-001.

## 7. Simulator weaknesses

- One codebase defines every current world, so all worlds share causal factorization, token
  semantics, class count, and observation mechanics.
- Factorial diagnoses use hand-coded hallmark blocks and weak secondary features; shortcuts may
  remain even when chronology is controlled.
- Demographics are label-dependent and can provide a stable shortcut unless explicitly ablated.
- Sequence length, positive/negative ratio, evidence stage, and missingness may be label-predictive.
- The same RNG seed is reused across evaluation worlds, creating paired but correlated cases.
- Existing tests validate determinism, signed missingness, counterfactual marker swaps, and task
  filters; they do not test metadata leakage, train/test generator overlap, seed recoverability,
  trivial feature classifiers, or impossible combinations.
- Current shifts do not include prevalence shift, spurious inversion, MCAR/MAR/MNAR separation,
  duplication, distractors, adversarial order, temporal-dependency change, or true class expansion.
- Ambiguity is limited to deleted chronology markers for twin labels; broader posterior
  multimodality is absent.

The red-team gate must run before any grand comparison. If a shortcut control crosses a frozen
threshold, QN-GRAND-001 must remain blocked until the generator is repaired and re-preregistered.

## 8. Theoretical gaps

- The implementation reports learned linear-residual commutators, while the actual update also
  includes injection, complex `tanh`, and norm projection. The measured commutator is not the full
  nonlinear state-conditioned order effect.
- No task-level normalized order-dependence functional is defined and validated.
- There is no analytic toy problem comparing generalization or stability at fixed resources.
- No proposition connects task structure, environmental shift, and expected excess risk.
- No invariant has been shown to predict robustness.
- Phase retention, trajectory curvature, and noncommutativity have not been separated from one
  another.
- Exact real representability is acknowledged in prose but not implemented as a numerical
  equivalence test.
- The discovery engine has no held-out confirmation split and cannot estimate selection bias.
- No meta-model predicts the winning architecture on unseen generators.

A valid next-phase theoretical result may identify a real-valued mechanism and remove complex
numbers from the final contribution. The program must treat that as success, not dilution.

## 9. External-validity gaps

- No independent synthetic task outside NeuroWorld.
- No real-world nonclinical sequential benchmark.
- No prospectively frozen dataset selection or license/ethics ledger.
- No separately authored replication.
- No container image or minimal clean-room replication workflow.
- No clinical data, clinician review, hospital validation, subgroup analysis, or prospective
  protocol. Q-Neuro remains a research model, not a decision-support prototype or medical device.
- The prior-art ledger is careful but not systematic and predates the next-phase architecture and
  law claims.

The repository therefore provides no evidence for medical performance and must not use clinical
accuracy, clinical robustness, diagnostic superiority, or quantum-cognition language.

## 10. Reproducibility and provenance gaps

The current registry is useful locally but insufficient for the mandated grand study:

- `experiments/registry.sqlite3` is ignored and is not present in the Git release;
- 14 run environments report a dirty Git worktree, including several full follow-up studies;
- 639 local checkpoints exist, but none are tracked; observable, halting, and trajectory runners
  depend on ignored source checkpoints;
- result JSON stores metrics but not an artifact hash, command, complete package inventory,
  dataset fingerprint, or per-example prediction file;
- registry artifact paths are absolute machine-local paths;
- `register_hypothesis` and `register_architecture` use `INSERT OR REPLACE`, which is not immutable;
- the schema lacks explicit hypothesis, preregistration version, world, shift, severity, seed,
  parameter, compute, and result-hash fields on the experiment row;
- `make reproduce-paper` rebuilds the paper from cached artifacts but does not reproduce a central
  training run;
- `REPLICATION.md` and the required next-phase make targets do not exist.

Before confirmatory execution, every run must reject dirty source by default, hash inputs and
outputs, store relative paths, preserve raw predictions, and expose a portable registry snapshot.

## Exact recommended execution sequence

The following order is binding unless a change is documented as a preregistration amendment made
before any affected outcome is observed.

### Gate 0 — freeze this audit

1. Commit this audit alone.
2. Do not reinterpret v0.1 experiments as prospective confirmation.
3. Assign the current evidence the provisional status “internal simulator signal; mechanism and
   generality unresolved.”

### Gate 1 — preregistration and provenance

4. Write `docs/PREREGISTRATION_NEXT_PHASE.md` and
   `experiments/configs/grand_falsification.yaml` with H0–H7, endpoints, minimum effect,
   multiplicity, exclusions, stopping rules, seed hierarchy, and prohibited post-hoc changes.
5. Add immutable hypothesis/claim/failure ledgers in JSON and extend the registry without altering
   QN-000001–QN-000026.
6. Add result/config/environment hashing, clean-tree enforcement, relative artifact paths, raw
   prediction storage, resume semantics, and portable registry export.
7. Implement and test `research/statistics.py`; simulate power to choose world count rather than
   adopting 30–50 by fiat.

### Gate 2 — strongest controls

8. Implement exact real-block equivalence and a forward/gradient equivalence test.
9. Add LSTM, vanilla RNN, causal Transformer, dense real recurrence, orthogonal recurrence,
   residual gated recurrence, real rotation blocks, polar state, and the fixed two-state attractor.
10. Freeze parameter-, state-degree-, optimizer-search-, initialization-, update-, and compute-
    matching policies. Controls receive at least the complex model's tuning budget.

### Gate 3 — simulator red team

11. Build the formal shortcut suite and exploitative baselines.
12. Test label leakage, metadata, length, order markers, missingness, seed overlap, deterministic
    artifacts, class imbalance, and impossible combinations.
13. Repair and re-preregister only if a frozen shortcut threshold fails; retain every failed audit.

### Gate 4 — ShiftGauntlet and pilot

14. Implement each shift family as an independent axis with unit tests and commutative controls.
15. Add reduced smoke, pilot, and full profiles with deterministic caching and resume.
16. Run pilot variance estimation only. Use it to freeze world count, training-seed count, severity
    grid, minimum effect, and compute budget. Pilot worlds cannot enter confirmation.

### Gate 5 — mechanisms and law discovery

17. Implement phase destruction during training/inference, commuting penalties/constraints,
    commutator encouragement, readout/dynamics freezing, state/rank/depth sweeps, and real models
    that steal each suspected mechanism.
18. Define normalized task order dependence, counterfactual-order divergence, trajectory geometry,
    robustness curves, and information-retention measures.
19. Add at least four unrelated non-medical task generators, including explicit commutative and
    noncommutative controls.
20. Run only the designated discovery split and fit competing linear, threshold, saturating,
    interaction, and nonlinear laws with nested validation.
21. Freeze one candidate law, its coefficients/selection procedure, and its falsifier before
    opening confirmation tasks.

### Gate 6 — untouched confirmation and grand test

22. Run the frozen law on untouched task families and worlds. Do not tune on their outcomes.
23. Preflight QN-GRAND-001: verify artifact hashes, no shortcut gate failures, equal search
    budgets, complete baselines, power target, and clean commit.
24. Execute QN-GRAND-001 once. The primary comparison is complex Q-Neuro versus the strongest
    preregistered parameter- and compute-matched real sequential control.
25. Apply the frozen decision rule. If any required survival condition fails, classify the
    hypothesis as failed or ambiguous without changing thresholds.

### Gate 7 — external tasks, synthesis, and release

26. Only after dataset choices and licenses are preregistered, evaluate selected real-world
    nonclinical datasets. Keep these separate from QN-GRAND-001 unless preregistered otherwise.
27. Generate all tables, figures, robustness curves, phase diagrams, and Pareto fronts directly
    from immutable artifacts.
28. Upgrade the dashboard so counterevidence, failed hypotheses, replication status, and open
    falsifiers are as prominent as positive results.
29. Rewrite the paper around the central question and final decision category. State explicitly
    what is not established.
30. Produce `REPLICATION.md`, a clean-room smoke path, expected hashes, and an independent rerun
    command.
31. Run tests, lint, registry validation, artifact hash verification, manuscript render checks,
    secret scanning, and a fresh-clone replication rehearsal before release.

## Advancement and stop rules

The project advances to QN-GRAND-001 only if all of the following are true:

- the exact real-equivalent implementation is tested and included;
- parameter and compute matching pass automated tolerances;
- simulator red-team gates pass;
- the top-level replication count meets the frozen power target;
- a non-medical untouched task family is included;
- the discovery law is frozen before confirmation;
- no confirmatory run originates from a dirty worktree;
- all exclusions follow the preregistration;
- raw predictions and artifact hashes are preserved.

The main robustness hypothesis is falsified if the strongest well-tuned matched real control
eliminates the complex model's preregistered aggregate OOD advantage. It is also considered failed
for QN-GRAND-001 if the remaining effect is smaller than the frozen practical threshold, is driven
by one shift family, catastrophically degrades calibration, depends on a simulator shortcut, or
cannot be reproduced from the release artifact.

## Current claim allowed after this audit

> Within the existing NeuroWorld simulator family, the implemented complex operator shows a
> repeatable shifted-classification advantage over the tested finite controls. Current experiments
> do not distinguish complex arithmetic from structured real-equivalent parameterization,
> optimization, regularization, or simulator-specific explanations, and they provide no clinical
> or external evidence.

No broader wording is supported before the next-phase gates are completed.
