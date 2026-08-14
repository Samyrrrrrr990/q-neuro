# Q-Neuro grand falsification preregistration

Preregistration ID: `QNF-PREREG-002`  
Version: `2.0.0`  
Status: frozen before next-phase model implementation or outcome access  
Parent audit: `docs/NEXT_PHASE_AUDIT.md`  
Machine-readable protocol: `experiments/configs/grand_falsification.yaml`  
Evidence scope: synthetic and nonclinical computational research only

## 1. Research question

Are there computational regimes in which a constrained complex-valued, noncommutative
hypothesis-state update provides a reproducible and theoretically explainable robustness advantage
over equivalently expressive real-valued sequential models at matched parameter and compute
budgets?

The purpose of this protocol is to falsify that proposition. A finding that an exact or structured
real model reproduces the effect is a successful mechanism result and a failure of any claim that
complex arithmetic is uniquely responsible.

## 2. Primary and competing hypotheses

### Primary hypothesis

On tasks where target-relevant ordered relational structure interacts with environmental shift,
the complex operator will exceed the strongest preregistered matched real-control envelope on the
aggregate out-of-distribution area under the robustness curve (OOD-AURC).

### H0 — no intrinsic complex advantage

After matching real degrees of freedom, trainable scalars, update count, optimization search, and
compute, the complex-minus-best-real OOD-AURC effect is smaller than the minimum practically
meaningful effect or is statistically compatible with no advantage.

### H1 — capacity explanation

The effect disappears when state degrees of freedom, effective transition rank, readout capacity,
and parameter count are jointly matched.

### H2 — optimization explanation

The effect changes materially across initialization/optimizer conditions or disappears under an
exact real-block implementation of the same function class.

### H3 — noncommutativity explanation

Advantage increases with target-relevant task order dependence, disappears on commutative
controls, and is reduced by interventions that force or encourage commuting transitions.

### H4 — phase-memory explanation

Phase destruction degrades retained order information and robustness, while a matched real polar
or rotation-block model recovers some or all of the effect.

### H5 — regularization explanation

The complex constraints change norm, margin, stability, or loss geometry in a way that predicts
robustness even after representational equivalence is established.

### H6 — simulator artifact

The effect does not survive shortcut controls, independently implemented synthetic generators, or
task semantics outside NeuroWorld.

### H7 — general computational phenomenon

A frozen task-level structural law predicts the direction and magnitude of the architecture effect
on untouched task families.

## 3. Study families and separation of roles

Data are partitioned by **generator family and world**, never by individual case alone.

1. **Development tests:** tiny deterministic fixtures used only to debug invariants. They cannot
   contribute to an effect estimate.
2. **Pilot split:** estimates runtime and top-level variance. Pilot worlds cannot enter discovery
   or confirmation.
3. **Discovery split:** varies task structure and fits candidate laws. It cannot be used to report
   confirmatory coverage or choose QN-GRAND-001 thresholds.
4. **Confirmation split:** sealed until one candidate law and all analysis code are frozen.
5. **QN-GRAND-001:** the primary confirmatory run over the sealed split.
6. **External nonclinical studies:** separately preregistered by dataset before final-test access;
   they cannot retroactively rescue QN-GRAND-001.

At least one confirmation generator family must be absent from mechanism discovery.

## 4. Task families

### NeuroWorld

NeuroWorld remains one task family. It includes chronology twins, factorial archetypes, explicit
missingness, signed evidence, ambiguity controls, and latent interventions. Its contribution is
weighted equally to each other top-level generator family; it cannot dominate through more cases.

### Independent synthetic families

The full protocol requires:

- hidden causal machine;
- sequential detective;
- machine fault diagnosis;
- hidden-rule relational classification;
- noncommutative analytic toy tasks;
- matched commutative controls.

Generator code, token semantics, latent structure, and observation processes must not import
NeuroWorld templates. Discovery and confirmation assignments are specified in the YAML protocol.

### Real-world nonclinical datasets

Dataset selection, license, environment split, target, metric, and exclusion criteria must be
recorded in `docs/EXTERNAL_DATASETS.md` before final-test evaluation. Results are secondary because
their task structure may not permit analytic order-dependence measurement.

## 5. ShiftGauntlet

The preregistered shift families are:

1. nuisance-variable shift;
2. prevalence shift;
3. conditional feature shift;
4. spurious-correlation inversion;
5. missingness shift with MCAR, MAR, and MNAR-like mechanisms;
6. observation noise;
7. evidence-order perturbation: canonical, random, partial, adversarial, and reversed;
8. distractor evidence;
9. contradictory evidence;
10. delayed decisive evidence;
11. unseen factor combinations;
12. unseen world mechanisms;
13. evidence deletion;
14. evidence duplication;
15. temporal-dependency change;
16. class expansion;
17. irreducible ambiguity.

Each applicable shift is evaluated at normalized severity `s in {0.00, 0.25, 0.50, 0.75, 1.00}`.
Only one shift family varies at a time for main effects. A small, separately labeled interaction
grid varies order dependence and shift strength jointly. Severity mappings are generator-specific
but frozen before pilot outcomes.

## 6. Baseline families

The required primary comparison set is:

- complex low-rank operator;
- exact tied real block-matrix equivalent;
- unrestricted paired-real operator;
- real low-rank operator;
- real polar-state operator;
- real SO(2) rotation-block operator;
- unrestricted dense real matrix recurrence;
- orthogonal/unitary-style real recurrence;
- residual gated recurrent model;
- GRU;
- LSTM;
- vanilla RNN;
- Transformer encoder;
- compact causal Transformer;
- state-space sequence model;
- unordered MLP negative control.

Exploitative simulator controls—class prior, metadata-only, length-only, order-only, single-feature,
shallow tree, permutation-invariant, and nearest neighbor—are red-team checks rather than members
of the primary architecture envelope.

The **best-real envelope** is the maximum real-control OOD-AURC within each paired top-level unit.
Using the test-time maximum deliberately favors H0. Secondary comparisons report every fixed
comparator separately so the envelope cannot hide model-specific failures.

## 7. Matching policy

Comparisons are reported in three panels:

1. **parameter matched:** trainable real-scalar count within 2% of the complex candidate;
2. **state matched:** real hidden degrees of freedom within one scalar coordinate and readout depth
   identical;
3. **compute matched:** cumulative training FLOPs within 5%, optimizer steps identical, and
   inference update FLOPs reported. If FLOP matching is impossible, the control receives the next
   larger budget and the difference is recorded.

The exact real-block model must pass forward, probability, loss, and gradient equivalence tests at
fixed mapped parameters before it enters an experiment.

All architectures receive the same base optimizer families, schedules, batch sizes, clipping,
epoch ceilings, and early-stopping rule. The complex model receives eight hyperparameter trials;
each real control receives ten. The extra real trials intentionally favor falsification. Trial
selection uses source validation NLL followed by source validation top-1 as a deterministic
tie-breaker. Shifted outcomes are unavailable to selection.

Each trial has two initialization attempts in discovery and one frozen selected initialization
policy in confirmation. Wall-clock measurements are secondary; FLOPs and optimizer steps define
compute matching.

## 8. Sample sizes and seeds

### Pilot

- 8 independently generated worlds per pilot task family;
- 3 training seeds per architecture;
- training sizes 250 and 1,000;
- severities 0.00, 0.50, and 1.00;
- reduced baseline set sufficient to estimate paired variance.

The pilot may change only the number of top-level worlds, using the frozen power-simulation rule.
No endpoint, effect threshold, model, or shift may be removed because of pilot performance.

### Full discovery

- 24 worlds per discovery generator family;
- 5 training seeds;
- training sizes 250, 1,000, 5,000, and 25,000 when feasible;
- sequence lengths 4, 8, 16, 32, 64, 128, and 256 when supported.

### QN-GRAND-001 confirmation

- minimum 32 worlds per confirmation generator family;
- 5 training seeds per architecture;
- primary training size 5,000;
- all applicable shift families and five severities;
- at least one untouched non-NeuroWorld generator family.

Power is simulated from pilot world-level paired differences. Let `delta_min = 0.02` absolute
OOD-AURC. The number of worlds is the smallest of `{32, 40, 48, 60}` yielding at least 90% power
for a two-sided 5% paired test under the conservative pilot variance, capped at 60. If 60 worlds
does not reach 90%, the study proceeds with 60 and the shortfall is declared before confirmation.

World seeds and training seeds in the YAML file are immutable. Generator-family seed namespaces
must be hashed with the family name to avoid accidental sample reuse.

## 9. Endpoints

### Primary endpoint

For model `m`, shift family `k`, world `w`, and normalized severity `s`, let `A_mkws` be top-1
accuracy. The per-world robustness area is the trapezoidal integral

`R_mkw = integral_0^1 A_mkw(s) ds`.

Generator families and shift families receive equal weight. The primary paired effect is

`Delta_w = mean_(generator, shift) [R_complex - max_(m in matched real controls) R_m]`.

The estimand is the mean of `Delta_w` across confirmation worlds under the prespecified hierarchy.

### Co-primary safeguard

Shifted negative log-likelihood must be non-inferior to the best-real envelope with margin `+0.10`
nats, and ECE must be non-inferior with margin `+0.03` absolute. A ranking gain accompanied by
catastrophic confidence degradation does not survive.

### Secondary endpoints

- median and worst-decile OOD-AURC;
- robustness slope and catastrophic failure severity;
- counterfactual accuracy and false order sensitivity;
- Brier score, NLL, adaptive ECE, calibration slope/intercept;
- ambiguity valid-set mass and set NLL;
- area under the active-evidence accuracy curve;
- parameter count, FLOPs, optimizer steps, CPU/GPU latency, peak memory, and model bytes;
- probability of superiority and architecture rank stability;
- task order-dependence, commutator, trajectory, and information-retention measures;
- source and shifted sample-complexity crossover with uncertainty.

### Exploratory endpoints

Geometry plots, UMAP, nonlinear probes, Hessian approximations, mutual-information estimators, and
meta-model feature importance are exploratory unless elevated by a preregistration amendment
before confirmation data are opened.

## 10. Statistical units and hierarchy

The top-level unit is an independently generated **world within generator family**. Cases estimate
performance inside a world. Training seeds estimate optimization variability and are averaged
within world before the main contrast. Shift severity is repeated within world. Generator and
shift families are crossed fixed design factors; world is a random factor.

The primary analysis reports:

- hierarchical bootstrap 95% interval with 20,000 resamples, resampling generator family, then
  world, then training seed where applicable;
- paired world-level mean, median, standard deviation, and 10% trimmed mean;
- probability of superiority `P(Delta_w > 0)`;
- a prespecified mixed-effects sensitivity model with architecture, shift, severity, and their
  interactions as fixed effects and world/family as nested random intercepts;
- an exact or Monte Carlo paired sign-flip test over top-level world contrasts.

The bootstrap interval is primary. The mixed model and sign-flip result are sensitivity analyses.
No per-example significance test will be used for architecture claims.

## 11. Multiplicity

There is one primary aggregate endpoint and two co-primary calibration safeguards. Holm correction
at familywise alpha 0.05 applies to the three confirmatory tests. Prespecified fixed-comparator and
mechanism contrasts use Holm correction within their named families. Exploratory scans use
Benjamini-Hochberg FDR at 0.10 and are labeled hypothesis-generating. A discovered law receives no
confirmatory status until evaluated on the sealed split.

## 12. Falsification, survival, and ambiguity rules

### The primary robustness claim is falsified if any holds

1. mean complex-minus-best-real OOD-AURC is non-positive;
2. the point estimate is smaller than `delta_min = 0.02`;
3. the Holm-adjusted 95% lower confidence bound does not exceed zero and the point estimate does
   not exceed `delta_min`;
4. an exact real or structured real model eliminates the aggregate advantage;
5. one shift family contributes more than 50% of the positive aggregate effect;
6. fewer than 60% of applicable shift families have positive mean effects;
7. NLL or ECE violates the frozen non-inferiority margin;
8. a simulator leakage gate fails;
9. the result cannot be reproduced from frozen release artifacts.

### Strong survival requires all of the following

1. mean OOD-AURC effect is positive;
2. the Holm-adjusted 95% lower bound exceeds `delta_min = 0.02`;
3. at least 75% of applicable shift families have positive mean effects;
4. no shift family contributes more than 50% of the total positive effect;
5. both calibration safeguards pass;
6. parameter, state, and compute matching pass;
7. no leakage gate fails;
8. direction replicates in at least one untouched non-NeuroWorld family;
9. a fresh-clone rerun reproduces the registered primary estimate within `0.005` absolute.

### Ambiguous result

The result is ambiguous if the effect is positive but the lower bound does not exceed
`delta_min`, if heterogeneity reverses the effect in more than 40% of shift families, if power is
below the declared target, or if calibration is inconclusive without crossing its failure margin.
Ambiguous results may motivate a new preregistration but cannot be called confirmation.

### Mechanism-only result

If a real model that explicitly steals the suspected mechanism matches the complex model within
`0.005` OOD-AURC and its paired 95% interval lies within `[-0.01, +0.01]`, Outcome C is selected:
the mechanism is not uniquely complex. This overrides branding but does not erase a valid
structural discovery.

## 13. Exclusions and failures

A run is excluded only for:

- non-finite loss or metrics;
- verified implementation/invariant failure;
- interrupted hardware/process failure before the frozen epoch budget;
- corrupted artifact hash;
- duplicate seed/world caused by a registry error.

Poor performance, long runtime, calibration failure, unfavorable direction, or failure to converge
to a preferred solution are not exclusion grounds. Every attempted run receives an immutable ID
and status. The primary analysis is intention-to-run: failed runs are counted and reported. If
failure rates differ by architecture by more than 5 percentage points, reliability becomes a
co-primary adverse result and no complete-case superiority claim is allowed.

## 14. Stopping rules

Training uses the frozen source-validation early-stopping rule. The study does not stop early for
positive or negative comparative performance. It stops for safety/integrity only if:

- shortcut/leakage gates fail;
- artifact hashing or registry immutability fails;
- more than 10% of runs fail for infrastructure reasons;
- the compute projection exceeds the declared ceiling by more than 25%.

After a stop, completed results remain sealed from design changes until the failure and proposed
amendment are recorded. A resumed study uses new IDs and a new preregistration version.

## 15. Prohibited post-hoc changes

After pilot outcomes are read, it is prohibited to:

- change the primary endpoint, `delta_min`, calibration margins, or hierarchy;
- drop a baseline, task, shift, severity, seed, world, or metric because it is unfavorable;
- select hyperparameters using shifted or confirmation outcomes;
- move a task from discovery to confirmation;
- change the candidate law after opening confirmation data;
- redefine order dependence to improve a correlation;
- replace the best-real envelope with a weaker comparator;
- omit failed runs or inconvenient negative results;
- present exploratory FDR results as confirmatory;
- use QN-GRAND-001 more than once under this preregistration.

Corrections for implementation errors require a versioned amendment, preserved old artifacts, new
experiment IDs, and a statement of whether any affected outcomes had been observed.

## 16. Decision mapping

- **Outcome A — falsified:** a matched real envelope removes the practical advantage or any
  mandatory survival condition fails decisively.
- **Outcome B — narrow conditional effect:** a robust effect survives only in a prespecified
  structural region but no general held-out law is established.
- **Outcome C — mechanism identified, not uniquely complex:** a real model stealing the mechanism
  matches complex performance.
- **Outcome D — robust unexplained phenomenon:** the effect survives strong controls and external
  task families, but no frozen law predicts it.
- **Outcome E — general computational law:** the frozen structural law predicts effect direction
  and magnitude on untouched task families with preregistered accuracy and calibration criteria.

Outcome E additionally requires out-of-family predictive `R^2 >= 0.50`, correct sign prediction on
at least 80% of confirmation task cells, and calibration of predicted versus observed advantage
within mean absolute error `0.015`. These thresholds cannot be weakened after discovery.

## 17. Clinical and communication boundary

This protocol contains no patient data and cannot establish clinical validity. “Diagnosis” refers
to synthetic or nonclinical latent-state identification. No outcome permits claims that the brain
is quantum mechanical, that Q-Neuro is a quantum computer, that the architecture is clinically
safe, or that it is universally superior. Novelty remains subject to the prior-art audit.

## 18. Freeze statement

The human author and any automated agent must preserve this document and its machine-readable
counterpart. The next permissible implementation work is provenance/statistics infrastructure and
the preregistered baseline suite. QN-GRAND-001 remains locked until all preflight gates pass.
