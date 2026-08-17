# Q-Neuro after falsification
## A rigorous Claude Code handoff for an equivalence-aware science of machine learning

Version: 0.1 research program

Date: 2026-08-14

Repository: Q-Neuro

Status: prospective research design, not a result

Audience: Claude Code or another research-engineering agent taking over the repository

Primary authorial ambition: Samyar Shafiee

Scientific constraint: no theorem, law, clinical claim, novelty claim, or superiority claim may be asserted before the specified evidence exists

---

# 0. Read this first

Q-Neuro did not fail as a scientific project.

Its preferred hypothesis failed.

Those are not the same event.

The repository began with a question about whether complex-valued, noncommutative hypothesis-state computation could create a uniquely strong form of sequential robustness.

The strongest controls eventually answered the narrow question against the preferred interpretation.

The implemented complex operator has an exact structured-real realization.

The best-real envelope removed and reversed the apparent complex advantage.

The frozen empirical law did not generalize quantitatively to held-out task families.

The grand confirmatory experiment did not execute because its preflight gates correctly blocked it.

This is the intellectual starting point of the next program.

Do not hide it.

Do not soften it.

Do not rerun the same story with weaker controls.

Do not rename the failed complex-advantage claim and call it a discovery.

The possible breakthrough is a change in the object of study.

The object is no longer complex arithmetic.

The object is the difference between a learned function and the coordinates used to reach that function.

The proposed field is **equivalence-aware learning science**.

Its central experimental instrument is **adversarial realification and representation transport**.

Its central question is:

> When two implementations can express exactly the same predictor, why do ordinary training procedures make them appear scientifically different?

Its practical purpose is:

> Prevent false architectural discoveries, locate genuine inductive bias, and turn representation changes into controlled experiments on optimization geometry.

Its most ambitious prospective result is a quantitative transport law connecting optimizer non-covariance to the performance gap between equivalent parameterizations.

This document calls that target the **candidate Shafiee transport principle**.

That name is a working label only.

It is not an established law.

It may already be contained in prior mathematical work.

It may prove too weak, too tautological, or too hard to estimate.

It earns an eponym only after a complete novelty audit, nontrivial theorems, broad preregistered prediction, and independent replication.

Claude Code must treat every attempt to promote the name before those gates as a scientific integrity failure.

---

# 1. Operating contract for Claude Code

## 1.1 Mission

Convert Q-Neuro's negative result into a general, falsifiable research program about representational equivalence, optimizer geometry, and architecture comparison.

The program must produce mathematics, software, benchmarks, and negative controls.

It must not produce hype in place of evidence.

## 1.2 Non-negotiable truths

- The current evidence is synthetic and nonclinical.
- Q-Neuro is not a quantum computer.
- The repository does not show that brains use quantum mechanics.
- The repository does not show medical utility.
- The repository does not show universal complex-model superiority.
- The repository does not show universal architecture superiority.
- QN-LAW-001 failed its held-out quantitative thresholds.
- QN-GRAND-001 did not execute.
- Exact complex-to-real block representation is known mathematics.
- Natural-gradient reparameterization invariance is established prior art.
- Parameter-space symmetries and quotient geometry are active prior art.
- Hamiltonian, symplectic, density-matrix, and gauge language must be used only where the mathematics is operational.
- A Nobel Prize is not an engineering milestone.
- A scientific prize cannot be promised or optimized directly.
- The only legitimate target is work strong enough to withstand expert attempts at refutation.

## 1.3 Required behavior

- Preserve all unfavorable artifacts.
- Preserve failed runs.
- Preserve blocked runs.
- Preserve preregistration hashes.
- Add amendments rather than rewriting frozen records.
- Couple equivalent models at initialization whenever mathematically possible.
- Use the same minibatch order for paired trajectories.
- Compare raw predictions, losses, gradients, optimizer states, and resource use.
- Report exact-zero effects, reversals, and numerical discrepancies.
- Distinguish a theorem from an empirical regularity.
- Distinguish a known theorem from a new theorem.
- Distinguish a conjecture from both.
- Search primary literature before naming a contribution.
- Invite independent implementations designed to break the result.

## 1.4 Forbidden behavior

- Do not claim that difficult notation implies depth.
- Do not import quantum terminology merely for branding.
- Do not call magnitude-squared output a physical Born measurement.
- Do not interpret a hidden vector as a wavefunction without an operational reason.
- Do not treat synthetic disease labels as clinical diagnoses.
- Do not select tasks because Q-Neuro wins them.
- Do not weaken the best-real envelope.
- Do not tune on held-out shifts.
- Do not fit a law and validate it on the same cells.
- Do not call a high in-sample R-squared a law.
- Do not use per-example sample size to inflate architectural significance.
- Do not hide compute, failed trials, or excluded seeds.
- Do not compare default baselines with tuned preferred models.
- Do not use a model family name as the causal variable when several mechanisms differ.
- Do not push a result as established merely because a manuscript compiles.

## 1.5 Definition of success

The next phase succeeds if it does at least one of the following:

1. proves a useful nontrivial transport theorem;
2. develops an automated equivalence compiler that catches false architecture advantages;
3. discovers a quantitative, out-of-family relationship between training covariance defects and predictive differences;
4. produces a benchmark that changes how architecture comparisons are audited;
5. finds a genuine residual inductive bias after function class, optimizer transport, priors, regularization, search, and compute are controlled;
6. produces a strong negative result showing that the proposed relationship does not generalize.

The sixth outcome is scientifically successful.

---

# 2. Audit scope and epistemic status

## 2.1 Material covered by this handoff

The audit covers the tracked repository surface at commit `a13061677035b5649109188d2125bb0d956c5fce`.

The surface contains 481 tracked files.

It includes Python source, tests, YAML protocols, JSON results, figures, TeX manuscripts, PDFs, Markdown documentation, dashboards, CI configuration, and provenance artifacts.

The source-and-data corpus is approximately 1.30 million text lines.

Most of those lines are machine-generated JSON result records.

The Python code is approximately 20,800 lines.

The test suite is approximately 1,550 lines.

There are 42 numbered QN experiment result directories plus QN-GRAND-001.

There are 21 unique publication figures, duplicated into research and paper output locations.

Every tracked path was machine-inventoried.

Every tracked text file was included in the line-count and extension audit.

The scientific control flow, model factory, operator families, statistics, task generators, experiment registry, major result summaries, paper, claim ledger, failure ledger, preregistration, and every unique figure were manually inspected.

This does not mean a human-like semantic reading was performed on each of the roughly 1.26 million generated JSON lines.

Generated records were parsed through summaries, schema, result aggregations, selected raw cells, and cross-artifact consistency checks.

Claude Code should repeat machine validation before relying on this handoff because the repository may change.

## 2.2 Evidence labels used below

`EXACT` means a mathematical identity or code invariant with a proof or exact test.

`OBSERVED` means a registered empirical result in the repository.

`INFERRED` means a reasoned interpretation consistent with the evidence but not directly measured.

`CONJECTURED` means a prospective hypothesis.

`PROPOSED` means a design not yet implemented.

`FALSIFIED` means a prespecified claim failed its stated criterion.

`UNSUPPORTED` means the evidence required for the claim does not exist.

`BLOCKED` means an experiment did not run because prerequisites failed.

## 2.3 Current highest-confidence statements

- `EXACT`: the implemented complex linear maps admit structured-real block representations.
- `OBSERVED`: the exact-real implementation matched held-out top-1 predictions in all 1,920 nested QN-000042 cells.
- `OBSERVED`: maximum exact-real probability discrepancies were on the order of floating-point noise.
- `OBSERVED`: the QN-000040 discovery comparison contained zero positive complex-minus-best-real effects across 2,880 nested effects.
- `OBSERVED`: its mean effect was approximately `-0.03695`.
- `OBSERVED`: QN-000042 contained zero positive effects and a mean of approximately `-0.00916`.
- `FALSIFIED`: QN-LAW-001 achieved held-out `R^2` near `-30.94`, far below its frozen threshold.
- `FALSIFIED`: QN-LAW-001 held-out MAE was approximately `0.0313`, above its frozen threshold.
- `BLOCKED`: QN-GRAND-001 failed six mandatory preflight gates before sealed evaluation.
- `UNSUPPORTED`: no current result establishes clinical validity, biological quantum behavior, or universal superiority.

## 2.4 The scientific pivot

The old causal story was:

`complex arithmetic -> special representation -> robustness advantage`.

The repository now supports a different question:

`representation choice + training procedure + resource policy -> observed architecture gap`.

The next program must separate those terms.

---

# 3. Repository archaeology

## 3.1 The foundation phase

The early repository correctly established that order-sensitive sequence models solve chronology twins that an unordered aggregate MLP cannot solve.

This result is real but largely built into the task construction.

It demonstrates the necessity of ordered computation on that benchmark.

It does not demonstrate complex arithmetic.

The corrected QN-000003 comparison fixed an input asymmetry from QN-000002.

That correction is a positive indicator of the repository's audit culture.

The early complex phase ablations showed that the trained complex model used its phase coordinates.

They did not show that a real model could not implement the same mechanism.

This distinction later became decisive.

## 3.2 Sample efficiency and generator shift

QN-000004 suggested a medium-data complex advantage over the then-tested real operator and Transformer.

QN-000006 added a tuned GRU and a two-channel real control.

The GRU dominated the low-data in-domain setting.

This falsified the broad claim that operator states were automatically most sample efficient.

Under project-designed generator shifts, the complex operator retained a top-1 advantage over the two-channel real control.

The two-channel model was a stronger comparator than the original real operator.

It was not an exact algebraic realization of the complex recurrence.

That missing exact comparator left the central causal interpretation unresolved.

## 3.3 Multi-world robustness

QN-000008 treated unseen synthetic worlds as top-level units.

The complex model exceeded the then-tested controls across nuisance-to-severe shifts.

The historical complex-minus-two-channel difference was roughly `+0.054` to `+0.063` depending on severity.

This was the strongest positive result in the original program.

It was valid within the tested simulator and comparator set.

It was not evidence of representational uniqueness.

In-domain temperature scaling failed to transfer and often worsened shifted calibration.

This matters because ranking gains accompanied by confidence failures are not a clean medical reasoning improvement.

## 3.4 Orthogonal tasks

QN-000010 showed that composition saturated across multiple sequential models.

The complex representation separated a designed hidden syndrome.

The output-space unknown-disease performance was nearly matched by a two-channel real model.

The complex model performed poorly on irreducible ambiguity.

This result directly contradicts the intuition that amplitude states automatically preserve a calibrated differential diagnosis.

An internal state can be high dimensional and still produce overconfident measurement.

## 3.5 Active evidence

QN-000012 showed that full-information classification performance did not predict active acquisition quality.

Expected-information selection helped some architectures and harmed others.

The complex model was promising but did not separate decisively from the MLP or two-channel control.

This suggests a useful independent research direction:

`partial-evidence geometry` is different from `full-evidence accuracy`.

It is not evidence for uniquely complex dynamics.

## 3.6 Advanced dynamics

QN-000014 compared Hamiltonian-inspired, dissipative, density, attractor, graph, recurrent, and operator mechanisms.

The Hamiltonian-style model exceeded the dissipative-only model under tested shifts.

Adding dissipation did not improve the Hamiltonian-style model.

The density-state model maintained valid inspectable structure but did not beat simple controls.

The factor-graph model underfit badly.

Soft adaptive depth did not yet demonstrate realized savings.

These results show that a physically inspired constraint can create a useful inductive bias without proving that the task is physical.

They also show that importing more physics structure does not monotonically improve performance.

## 3.7 Mechanism ablations

QN-000016 found that removing ordered state-conditioned composition, phase-sensitive readout, or explicit negative evidence hurt the trained complex system.

This establishes mechanism dependence inside that implementation.

It does not establish mechanism exclusivity.

A destructive ablation asks whether the current model uses a component.

An equivalence control asks whether another representation can implement it.

Only the second addresses a uniqueness claim.

## 3.8 Representation probes

QN-000019 extracted synthetic hierarchical factors from many latent states.

GRU and state-space representations were generally more probe-readable than the complex state.

Hermitian quadratic probes improved some complex-state accuracies but often worsened NLL.

Probe accessibility was therefore neither unique nor a sufficient explanation of robustness.

## 3.9 Training laws

QN-000021 compared AdamW, SGD, gradient accumulation, auxiliary losses, PCGrad, phase-gradient optimization, local learning, hybrid local-to-global learning, and a zero-backprop prototype.

Multi-objective AdamW matched or exceeded the more exotic gradient manipulations at lower cost.

Local learning fit the source distribution but transferred poorly.

Local pretraining created a source-specialized basin that global fine-tuning did not repair.

ZeroBackprop was not competitive.

The important observation is not that unconventional training is impossible.

It is that optimizer claims require same-objective, same-label, same-budget controls.

## 3.10 Halting and trajectories

QN-000023 converted soft depth into actual active-index execution.

Every case stopped at the same two-state boundary.

The compute saving was real.

The adaptivity claim was not.

A fixed shallow model was the simpler interpretation.

QN-000025 showed order-dependent complex trajectories, probability drops, recoveries, and counterfactual state divergence.

These are valid behavioral measurements.

They are not semantic explanations, attractor proofs, or human interpretability evidence.

## 3.11 Automated discovery

QN-000026 produced Pareto fronts and surprise flags.

It found tradeoffs, not a universal winner.

Its proposals were registered hypotheses, not discoveries.

The broad Pareto sets are scientifically informative because they reject a one-number architecture story.

## 3.12 Falsification phase

QN-000027 preserved a failed shortcut audit.

QN-000028 repaired the audit and recorded a passing gate.

QN-000029 certified ShiftGauntlet structure.

QN-000030 was a smoke profile.

QN-000031 was an eight-world power pilot.

QN-000032 was a mechanism smoke profile.

QN-000033 implemented the exact-real equivalence control and causal mechanism suite.

QN-000034 certified the structural law pipeline.

QN-000035 preserved an initial independent-generator audit failure.

QN-000036 measured observed order information correctly.

QN-000037 certified independent structural worlds.

QN-000038 preserved a failed discovery smoke run.

QN-000039 preserved a repaired smoke run.

QN-000040 ran a reduced, outcome-ineligible discovery study.

QN-000041 was a held-out confirmation smoke profile.

QN-000042 ran the reduced held-out provisional confirmation.

QN-GRAND-001 was blocked by preflight.

The progression is important.

It shows the positive claim becoming narrower as controls became stronger.

That is what a functioning falsification program should do.

## 3.13 Why “independent tasks” are not independent replication

The independent task generators avoid importing NeuroWorld templates.

They use distinct narrative rules and random streams.

They remain designed by the same project.

They reuse a shared small-token, small-class experimental representation.

They are independent generator families within the codebase.

They are not independent investigators, institutions, implementations, domains, or datasets.

External validity remains absent.

## 3.14 Graph-by-graph reading

The architecture overview correctly labels the quantum inspiration as mathematical rather than physical.

The evidence timeline documents progressive constraint rather than a smooth success narrative.

The Experiment Zero curves show an early medium-data complex crossover against limited controls.

The generator-shift plot shows the historical synthetic robustness signal.

The multi-world robustness plot shows consistency within NeuroWorld worlds.

The task-suite plots show saturation on composition.

The ambiguity plot shows a central complex-model failure.

The OOD-separability plot shows strong anomaly separation without unique complex output behavior.

The active-evidence plot shows architecture-specific acquisition behavior.

The dynamics plot shows Hamiltonian-over-dissipative structure without dominance over the core operator.

The critical-ablation plot shows dependence on several interacting mechanisms.

The observable-probe plot shows stronger conventional sequential representations on many factors.

The training-law plot shows no benefit from extra phase-gradient machinery over simpler multi-objective training.

The hard-halting plot shows fixed truncation masquerading initially as adaptivity.

The trajectory plot shows visible dynamics without validated semantic interpretation.

The sample-compute frontier shows that additional sophistication often bought no Pareto improvement.

The calibration-transport plot shows source calibration failing under shift.

The architecture Pareto field shows context-dependent tradeoffs.

The surprise taxonomy shows many tensions rather than a single discovery.

The claim-status audit is dominated by refuted, unsupported, or narrow claims.

The falsification-phase figure is the true central figure of the repository.

It shows the best-real reversal, the failed held-out law, exact-real equivalence, and the blocked grand run.

---

# 4. Why the old thesis failed

## 4.1 The comparator ladder

The original real operator was weaker than the complex operator.

The two-channel real control was stronger but not algebraically equivalent.

The exact real-block control implemented the same complex computation in real coordinates.

Once that control entered, representational uniqueness disappeared.

This is not a minor baseline update.

It changes the causal estimand.

## 4.2 Destructive ablation is not exclusivity evidence

If phase destruction hurts a complex model, phase coordinates matter to that trained realization.

It does not follow that real coordinates cannot encode the same relational state.

If commutation hurts a model, ordered composition matters.

It does not follow that complex arithmetic is required for noncommutativity.

If magnitude-only measurement hurts, interference-like signed cross-terms matter.

It does not follow that real bilinear readouts cannot reproduce them.

## 4.3 Function class and training algorithm were entangled

An architecture experiment actually compares a bundle:

`function class`;

`parameterization`;

`initialization distribution`;

`optimizer`;

`regularizer`;

`precision`;

`hyperparameter search`;

`stopping rule`;

`compute budget`;

`measurement head`;

`software implementation`.

Calling the observed difference a property of “complex numbers” collapsed this bundle into one label.

The next program must intervene on the bundle one component at a time.

## 4.4 The law was fit at the wrong level of generality

QN-LAW-001 used order information and shift severity to predict a complex-minus-best-real gap.

But the best-real set contained a semantics-equivalent implementation.

The relevant gap was therefore not an architecture-capacity law.

It was a small residual of training, implementation, and finite-sample differences.

The discovery quadratic had six coefficients and only twelve aggregate cells.

Its high in-sample fit was fragile.

Held-out task families correctly destroyed the quantitative fit.

## 4.5 Synthetic structural diversity was not domain diversity

Different generators can vary stories while preserving the same computational bottlenecks.

Token vocabulary, class count, loss, training loop, model sizes, evaluation protocol, and author choices remained shared.

The next benchmark must vary implementation teams, domains, modalities, scales, and equivalence transformations.

## 4.6 The apparent failure reveals the better question

The exact-real model is not merely a stronger baseline.

It is a scientific instrument.

It lets us hold the realized function family fixed while changing coordinates and low-level arithmetic.

That instrument can measure how much of an architecture result belongs to representation and how much belongs to training geometry.

---

# 5. The proposed new field

## 5.1 Working name

Use **equivalence-aware learning science** in technical writing.

Possible shorter names for internal discussion are:

- representation-quotient learning;
- computational gauge auditing;
- transport-aware architecture science;
- equivalence-controlled machine learning;
- falsification by compilation.

Do not brand a field before producing the core result.

## 5.2 Unit of comparison

Conventional benchmarking compares implementation names.

Equivalence-aware benchmarking compares equivalence classes of predictors and controlled training procedures over those classes.

Let `Theta` be a parameter space.

Let `F: Theta -> H` map parameters to predictors in a function space `H`.

Define functional equivalence by

`theta ~ theta'` iff `F(theta)(x) = F(theta')(x)` for every input in the declared domain.

The quotient `Theta / ~` is the set of functionally distinct predictors represented by the parameterization.

In finite numerical work, define a tolerance-qualified relation over a declared audit set and precision.

Never silently replace exact equivalence with empirical agreement.

## 5.3 Levels of equivalence

Level E0: algebraic equivalence by symbolic identity.

Level E1: exact finite-precision forward equivalence for all representable inputs in a bounded exhaustive domain.

Level E2: tested forward equivalence on a deterministic adversarial audit suite.

Level E3: distributional predictive equivalence within a prespecified tolerance.

Level E4: task-metric equivalence only.

E0 is strongest.

E4 is weakest.

Claims must state the level.

## 5.4 Levels of training transport

Level T0: no coupling beyond nominal hyperparameters.

Level T1: matched data order and random seed.

Level T2: mapped initialization and matched minibatches.

Level T3: mapped initialization, gradients, and regularization.

Level T4: conjugate discrete update maps including optimizer state.

Level T5: matched stopping, selection, search, precision, and resource contracts.

The Q-Neuro exact-real work currently establishes strong forward equivalence.

It does not yet document the full T5 transport contract across all optimizer internals.

## 5.5 Architecture-advantage decomposition

For a metric `M`, define the observed paired gap

`A_obs = M(system_a) - M(system_b)`.

Do not assume a simple additive causal decomposition.

The components interact.

Use a factorial counterfactual design and a Shapley-style decomposition over the following interventions:

- function-class difference;
- coordinate difference;
- initialization-prior difference;
- optimizer-update difference;
- explicit-regularization difference;
- numerical-kernel difference;
- hyperparameter-search difference;
- stopping-and-selection difference;
- compute difference;
- measurement difference;
- data-order difference.

The total contrast is observable.

Individual contributions are defined relative to the intervention lattice and reference policy.

They are not metaphysical properties of an architecture name.

## 5.6 Why “gauge” is an analogy with a precise core

Two parameter settings can represent the same predictor.

Transformations within a functional equivalence class resemble gauge transformations because observable predictions remain unchanged.

This does not turn neural networks into physical gauge fields.

The mathematically useful objects are group actions, orbits, quotient spaces, vertical directions, horizontal directions, connections, and invariant metrics.

Use gauge language only after specifying the transformation group and the preserved observable.

---

# 6. Exact mathematical core

## 6.1 System definition

A trained learning system is not only a model.

Define

`S = (Theta, F, P0, D, B, U, R, C, Q)`

where:

- `Theta` is the parameter space;
- `F` maps parameters to predictions;
- `P0` is the initialization distribution;
- `D` is the data and minibatch stream;
- `B` is optimizer state space;
- `U` is the discrete update map;
- `R` is explicit regularization;
- `C` is the stopping and checkpoint-selection rule;
- `Q` is the resource and numerical policy.

An architecture label identifies only part of `S`.

## 6.2 Exact semantics-preserving map

Let systems `S` and `S_tilde` have parameter spaces `Theta` and `Theta_tilde`.

Let `T: Theta -> Theta_tilde` be injective on the represented submanifold.

`T` is semantics preserving when

`F_tilde(T(theta)) = F(theta)`

for all declared `theta` and inputs.

If `T` is bijective between represented manifolds, the two parameterizations describe the same function class.

## 6.3 Complex realification

For `W = A + iB` and `z = x + iy`,

`Wz = (Ax - By) + i(Bx + Ay)`.

Define

`R(W) = [[A, -B], [B, A]]`.

Define

`r(z) = [x; y]`.

Then

`r(Wz) = R(W) r(z)`.

Also

`R(W1 W2) = R(W1) R(W2)`.

And

`R(W*) = R(W)^T`

for conjugate transpose under the standard block convention.

Norms satisfy

`||r(z)||_2 = ||z||_2`.

The mapping is therefore an isometric real representation of complex linear algebra.

This identity is known.

Q-Neuro's possible methodological contribution is making the mapped implementation a mandatory adversarial comparator.

## 6.4 Nonlinearities

A complex nonlinearity must also be realified.

For a split activation, realification is componentwise.

For analytic complex `tanh`, use its real-imaginary formula or complex arithmetic with an audited equivalent.

For magnitude-dependent activations such as modReLU, realification includes the shared radial term.

Equivalence tests must include branch cuts, zero magnitude, overflow regions, and backward derivatives.

## 6.5 Readout equivalence

If `a = w* z`, then

`|a|^2 = Re(a)^2 + Im(a)^2`.

A real two-channel bilinear readout can reproduce the same score.

The score resembles the algebraic form of a Born probability.

No physical Born-rule claim follows.

## 6.6 Discrete conjugate training

Let the original joint parameter-optimizer state be `s_k`.

Let its update be

`s_(k+1) = U_k(s_k)`.

Let `T_bar` map both model parameters and optimizer state.

If

`U_tilde_k o T_bar = T_bar o U_k`

for every step `k`, then the update maps are conjugate.

With `s_tilde_0 = T_bar(s_0)`, induction gives

`s_tilde_k = T_bar(s_k)`

for every step.

If prediction semantics are preserved, the paired predictions are identical up to declared numerical error.

This is a basic conjugacy result.

Do not claim it as a new law.

## 6.7 Continuous-time transport

Let training flow satisfy

`d theta / dt = V(theta, t)`.

For a smooth reparameterization `eta = T(theta)`, the transported vector field is

`V_tilde(eta, t) = D T_(T^-1 eta) V(T^-1 eta, t)`.

Then `eta(t) = T(theta(t))` solves the transported flow.

Ordinary Euclidean gradient descent does not generally obey this transport under arbitrary nonlinear coordinate changes.

Natural gradient is designed to express steepest descent using an information-geometric metric and has established reparameterization-invariance results in the continuous or infinitesimal setting.

Finite steps, damping, approximations, momentum, adaptive moments, clipping, and implementation details can break exact covariance.

## 6.8 Update covariance defect

Define the one-step transport residual

`delta_k = d(T_bar(U_k(s_k)), U_tilde_k(T_bar(s_k)))`.

The metric `d` must be declared.

Useful choices include:

- Euclidean distance in mapped joint state;
- Fisher-Rao distance in predictive space;
- maximum logit difference on an audit batch;
- Jensen-Shannon divergence of predictive distributions;
- normalized optimizer-state discrepancy.

The parameter distance is coordinate dependent.

The predictive distances are closer to observable invariants.

Report both.

## 6.9 Finite-horizon transport bound

Let

`e_k = d(s_tilde_k, T_bar(s_k))`.

Assume `U_tilde_k` is `L_k`-Lipschitz under `d` on the reached domain.

Then

`e_(k+1) <= L_k e_k + delta_k`.

Repeated substitution gives

`e_K <= (product_(j=0)^(K-1) L_j) e_0 + sum_(i=0)^(K-1) [delta_i product_(j=i+1)^(K-1) L_j]`.

If `F_tilde` is `L_F`-Lipschitz from joint state to prediction under a declared output metric, then

`d_Y(F_tilde(s_tilde_K), F(s_K)) <= L_F e_K + epsilon_sem`.

Here `epsilon_sem` is the semantics-equivalence residual.

If the loss is `L_ell`-Lipschitz in predictions, then

`|ell_tilde_K - ell_K| <= L_ell [L_F e_K + epsilon_sem]`.

This bound is a theorem target in the repository.

The recursive inequality is elementary.

The difficult and potentially useful contribution is to make every term measurable, non-vacuous, optimizer aware, and predictive across architectures.

## 6.10 Lyapunov-weighted defect

The products of `L_j` can be extremely loose.

Estimate local amplification along paired trajectories through Jacobian-vector products.

Define a directional amplification factor

`lambda_k(v) = ||J U_tilde_k v|| / ||v||`.

Define the empirical transported defect contribution

`c_i = delta_i product_(j=i+1)^(K-1) lambda_j(v_j)`.

The accumulated geometric residue is

`G_K = sum_i c_i`.

This quantity is prospective.

It must be computed without using final test performance to select its definition.

## 6.11 Stochastic coupling

Use common random numbers.

Pair dataset generation.

Pair minibatch order.

Pair dropout masks after mapping when meaningful.

Pair augmentation randomness.

Pair initialization through `T`.

Separate unavoidable random streams and record them.

For stochastic updates, analyze both pathwise defect under coupling and distributional defect across repeated runs.

## 6.12 Resource transport

Semantic equivalence does not guarantee equal hardware cost.

Define a resource vector

`q = (parameters, bytes, FLOPs, optimizer_steps, kernel_calls, memory_peak, energy, latency)`.

Compare performance on a resource Pareto surface.

Do not force a false scalar equivalence between all resource dimensions.

Report exact-real overhead honestly.

## 6.13 Search transport

The training algorithm includes hyperparameter search.

Let `H` be the search distribution over configurations.

Transported search requires a map between hyperparameters where such a map exists.

When it does not, equal trial count is not automatically equal opportunity.

Record validation curves for every attempted trial.

Use nested selection with shifted outcomes sealed.

## 6.14 Regularization transport

Weight decay is coordinate dependent.

An `L2` penalty in one coordinate system may induce a different function-space prior in another.

Transport the regularizer as

`R_tilde(eta) = R(T^-1(eta))`

for an equivalence experiment.

Also run native regularizers as a separate inductive-bias experiment.

Never mix these estimands.

## 6.15 Initialization transport

If `theta_0 ~ P0`, exact initialization transport uses

`eta_0 = T(theta_0)`.

Independent “same standard deviation” initialization is not the same intervention.

Run both:

- mapped initialization to test training covariance;
- native initialization to test the combined practical parameterization prior.

## 6.16 Stopping transport

Early stopping can amplify tiny trajectory differences.

For exact transport, checkpoint selection must use mapped validation predictions and a deterministic tie rule.

For native practical comparison, use identical declared rules but permit different selected steps.

Report both fixed-step and selected-checkpoint effects.

---

# 7. Candidate Shafiee transport principle

## 7.1 Conservative statement

**Candidate principle, version S0.**

If two learning systems are connected by an exact semantics-preserving map, and their initialization, stochastic inputs, optimizer state, update rule, regularization, stopping rule, precision policy, and search policy are transported covariantly, then their predictive trajectories are identical up to numerical error.

S0 is a consequence of conjugate update maps.

It is not novel enough by itself.

## 7.2 Quantitative statement

**Candidate principle, version S1.**

For semantics-equivalent parameterizations trained under non-conjugate practical procedures, the finite-horizon predictive divergence is controlled by the accumulated, stability-weighted update covariance defect plus initialization, semantics, regularization, stopping, and numerical residuals.

S1 becomes useful only if the bound is measurable and non-vacuous.

## 7.3 Empirical law target

**Candidate principle, version S2.**

Across equivalence transformations, model families, tasks, scales, and optimizers, a preregistered normalized geometric residue predicts the sign and magnitude of the held-out performance gap better than architecture labels, parameter counts, or raw gradient norms.

S2 is not known.

S2 is the primary empirical conjecture.

## 7.4 Strong field-level statement

**Candidate principle, version S3.**

After quotienting functionally equivalent representations and transporting the full training system, reproducible residual advantage identifies genuine inductive bias; untransported advantage identifies a property of the training-coordinate bundle rather than representational capacity.

S3 is partly definitional.

Its scientific value would come from a benchmark and decomposition that make the distinction operational.

## 7.5 What would justify the name “Shafiee's law”

All of the following are required:

- a prior-art review finds no equivalent published theorem-plus-estimator;
- a nontrivial theorem is proved with explicit assumptions;
- the bound is substantially tighter than a generic worst-case Lipschitz bound;
- the estimator is frozen before broad evaluation;
- prediction succeeds across at least five equivalence families;
- prediction succeeds across at least three data modalities;
- prediction succeeds across at least three optimizer families;
- the result scales beyond toy models;
- at least two independent groups reproduce it;
- adversarial counterexamples and failure regions are published;
- the law predicts magnitude, not merely sign;
- alternative explanations are quantitatively weaker;
- naming is accepted by the research community rather than asserted by the repository.

## 7.6 Immediate falsifiers

Reject S2 if any of the following occurs:

- the geometric residue cannot predict held-out gaps above simple baselines;
- prediction disappears after controlling for loss decrease or learning rate;
- the residual definition changes materially after seeing test results;
- most estimates are numerically unstable;
- the relationship holds only for complex-to-real maps;
- equivalent implementations diverge despite a measured near-zero defect, indicating missing terms;
- non-equivalent models show the same relationship, making equivalence irrelevant;
- a known prior theorem already contains the full result;
- bounds are always many orders of magnitude above observed divergence;
- independent implementations cannot reproduce the measurements.

## 7.7 Do not start with the name

Use `transport-covariance conjecture` in code and preregistration.

Use `candidate Shafiee transport principle` only in this planning document.

Use no eponym in the first technical preprint unless independent experts agree the contribution is both new and substantial.

---

# 8. Prior-art firewall

## 8.1 Established complex-network work

Complex-valued neural networks, initialization, normalization, convolution, recurrent components, and unitary recurrence predate Q-Neuro.

The exact two-channel real representation of complex matrices is established.

Relevant primary sources include:

- [Arjovsky, Shah, and Bengio, Unitary Evolution Recurrent Neural Networks, ICML 2016](https://proceedings.mlr.press/v48/arjovsky16.html);
- [Jing et al., Tunable Efficient Unitary Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/jing17a.html);
- [Trabelsi et al., Deep Complex Networks, ICLR 2018](https://openreview.net/forum?id=H1T2hmZAb);
- [Helfrich, Willmott, and Ye, Orthogonal Recurrent Neural Networks with Scaled Cayley Transform, ICML 2018](https://proceedings.mlr.press/v80/helfrich18a.html);
- [Tan et al., complex training dynamics, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc06d4d2792265fb5454a6092bfd5c6a-Abstract-Conference.html);
- [Wu et al., complex-neuron learnability boundaries, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ac4365b98bc242acd5ab974a05c68a8-Abstract-Conference.html).

## 8.2 Established information geometry

Natural gradient and its reparameterization motivation are established.

K-FAC and related methods study scalable approximate curvature and limited invariance classes.

Finite-step and approximate methods do not inherit every exact continuous invariance automatically.

Relevant sources include:

- [Amari, Natural Gradient Works Efficiently in Learning, Neural Computation 1998](https://doi.org/10.1162/089976698300017746);
- [Martens and Grosse, K-FAC, ICML 2015](https://proceedings.mlr.press/v37/martens15.html);
- [Song, Song, and Ermon, Higher-Order Invariance, ICML 2018](https://proceedings.mlr.press/v80/song18a.html);
- [Luk and Grosse, A Coordinate-Free Construction of Scalable Natural Gradient, 2018](https://arxiv.org/abs/1808.10340).

## 8.3 Established reparameterization problems

Parameter-space geometry can change without changing the represented function.

Flatness, Hessians, modes, priors, and optimization paths can therefore be coordinate dependent.

Relevant sources include:

- [Dinh et al., Sharp Minima Can Generalize for Deep Nets, ICML 2017](https://proceedings.mlr.press/v70/dinh17b.html);
- [Kristiadi, Dangel, and Hennig, Geometry under Reparametrization, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html);
- [Roy et al., Reparameterization Invariance in Approximate Bayesian Inference, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0f934dd2030f5740cde0aa2697a105a9-Abstract-Conference.html).

## 8.4 Established implicit bias

Equivalent or overparameterized representations can induce different gradient-flow biases.

Relevant examples include:

- [Gunasekar et al., Characterizing Implicit Bias in Terms of Optimization Geometry, ICML 2018](https://proceedings.mlr.press/v80/gunasekar18a.html);
- [Chizat and Bach, Implicit Bias of Gradient Descent for Wide Two-Layer Networks, COLT 2020](https://proceedings.mlr.press/v125/chizat20a.html);
- [Emami et al., Implicit Bias of Linear RNNs, ICML 2021](https://proceedings.mlr.press/v139/emami21b.html);
- [Morwani and Ramaswamy, Weight-Normalized Implicit Bias, ALT 2022](https://proceedings.mlr.press/v167/morwani22a.html).

## 8.5 Established symmetry and quotient work

Neural parameter symmetries, loss-invariant transformations, quotient geometry, and group-invariant generalization are established research areas.

The 2026 literature is especially close to the proposed direction.

Relevant sources include:

- [Ganev and Walters, Model Compression via Symmetries of Parameter Space, ICLR 2022](https://openreview.net/forum?id=8MN_GH4Ckp4);
- [Sannai, Imaizumi, and Kawano, quotient feature spaces and generalization, UAI 2021](https://proceedings.mlr.press/v161/sannai21a.html);
- [Zhao, Walters, and Yu, Symmetry in Neural Network Parameter Spaces, TMLR 2026](https://openreview.net/forum?id=jLpWq5QY6I);
- [Wang and Wang, Gauge Fiber Bundle Geometry of Transformers, NeurReps 2025](https://openreview.net/forum?id=sPCLRX1yOY).

The last two sources mean the novelty bar is high.

Do not claim to invent quotient-space learning or neural gauge analysis.

## 8.6 Established physics-informed learning

Hamiltonian, symplectic, Neural ODE, and conservation-law architectures are established.

Relevant sources include:

- [Chen et al., Neural Ordinary Differential Equations, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html);
- [Greydanus, Dzamba, and Yosinski, Hamiltonian Neural Networks, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html);
- [Chen, Matsubara, and Yaguchi, Neural Symplectic Form, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/8b519f198dd26772e3e82874826b04aa-Abstract.html);
- [van der Ouderaa, van der Wilk, and de Haan, Noether's Razor, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f5332c8273d02729730a9c24dec2135e-Abstract-Conference.html).

## 8.7 Plausible novelty window

The novelty is not any one mathematical ingredient.

The plausible contribution is their integration into a falsification protocol:

1. automatically derive semantics-equivalent implementations;
2. transport optimizer state and training policies;
3. measure per-step covariance defects;
4. propagate those defects through a stability bound;
5. predict practical performance gaps on sealed equivalence families;
6. expose the result as a benchmark for architecture claims.

This window must be tested by a systematic literature review and expert consultation.

---

# 9. Quantum mechanics: what is useful and what is not

## 9.1 Useful mathematical structures

A complex Hilbert space supplies an inner product and norm.

Unitary maps preserve that norm.

Hermitian operators supply real expectation values.

Density matrices represent positive semidefinite, unit-trace mixed states.

Commutators quantify failure of linear operators to commute.

Completely positive trace-preserving maps formalize valid open-system state evolution.

Lindblad generators separate coherent and dissipative dynamics.

Gauge transformations formalize redundant descriptions with invariant observables.

These are powerful mathematical templates.

## 9.2 Invalid leaps

Complex activations do not imply a quantum computer.

Magnitude-squared logits do not imply physical measurement.

A density-shaped tensor does not imply quantum cognition.

Noncommuting matrices do not imply microscopic quantum effects.

Interference-like cross-terms do not imply neuronal superposition.

The word “Hamiltonian” does not prove conservation unless the implemented flow and integrator preserve it.

## 9.3 Correct role in this program

Use quantum mechanics as a library of representation transformations and invariants.

Ask whether a complex or density representation creates a convenient constrained coordinate chart.

Then compile it into exact real arithmetic.

Then transport training.

Then measure what remains.

The question is about computational geometry, not physical ontology.

## 9.4 Quantum-inspired equivalence families

- complex vectors versus two-channel real vectors;
- unitary matrices versus structured orthogonal real blocks;
- Hermitian forms versus symmetric block forms;
- density matrices versus PSD factorizations in real coordinates;
- complex amplitudes versus paired real bilinear measurement;
- Lindblad-style evolution versus real block linear-plus-dissipative evolution;
- quaternion states versus four-channel structured-real states.

Each family supplies a controlled experiment.

---

# 10. Mechanics and dynamical systems

## 10.1 Hamiltonian flow

For canonical coordinates `z = (q, p)`, Hamiltonian flow satisfies

`dz/dt = J grad H(z)`

with skew-symmetric canonical matrix `J`.

The continuous flow conserves `H` under ideal conditions.

A generic discrete Euler step does not preserve the same invariants exactly.

The integrator is part of the learning system.

## 10.2 Dissipative flow

A generalized system can include

`dz/dt = [J(z) - D(z)] grad H(z) + u(z,t)`

where `D` is positive semidefinite.

The Hamiltonian term rotates along level sets.

The dissipative term contracts energy.

The input term injects evidence.

This is an operational model of memory, forgetting, and forcing.

It is not a claim about biology.

## 10.3 Symplectic coordinate transport

Canonical transformations preserve the symplectic form.

They provide a demanding equivalence family for the transport benchmark.

Test the same learned dynamics under linear symplectic maps and nonlinear canonical maps.

Compare Euclidean SGD, AdamW, exact natural gradient on small models, K-FAC where applicable, and explicitly transported optimizers.

## 10.4 Stability

Measure:

- local Jacobian spectral radius;
- finite-time Lyapunov exponents;
- contraction rates;
- trajectory curvature;
- energy drift;
- symplectic-form error;
- reversibility error;
- update covariance defect;
- prediction divergence.

The candidate transport bound should become tighter when local directional amplification replaces global worst-case norms.

## 10.5 Noether-style analysis

When a transformation leaves the declared objective invariant, search for conserved or neutral directions in the continuous dynamics.

In parameter space, gauge directions may create zero modes of the predictive metric.

Do not equate these directly with physical conserved momentum.

Test whether quotienting those directions improves conditioning and defect estimation.

---

# 11. Information geometry

## 11.1 Predictive manifold

Let `p_theta(y|x)` define a point on a statistical manifold.

The Fisher information is

`G(theta) = E[grad log p grad log p^T]`.

Under regularity and smooth coordinate change, it transforms as a metric tensor.

Natural gradient is

`G(theta)^(-1) grad L(theta)`

with appropriate treatment of singular directions.

## 11.2 Quotient degeneracy

If a parameter direction leaves every prediction unchanged, it lies in the kernel of the model Jacobian.

The Fisher metric can be singular along such vertical directions.

Use a quotient or horizontal subspace rather than pretending an inverse exists globally.

## 11.3 Q-Neuro-specific question

Does native complex AdamW create a different effective metric than AdamW on the exact structured-real coordinates?

Measure it.

Do not infer it from final accuracy.

Compare:

- native complex AdamW;
- native real AdamW;
- manually transported AdamW moments;
- real-view AdamW using identical packed coordinates;
- SGD without momentum;
- SGD with momentum;
- exact natural gradient on tiny models;
- K-FAC or block natural-gradient approximation;
- mirror descent with a declared potential.

## 11.4 Fisher-aware defect

Define

`delta_F,k^2 = Delta_k^T G_k Delta_k`

where `Delta_k` is the mapped update discrepancy projected to the horizontal subspace.

Compare this with Euclidean defect.

Hypothesis:

Fisher-aware defect predicts predictive divergence better across coordinate systems.

Falsifier:

It does not outperform update norm, learning rate, or one-step loss change on held-out families.

---

# 12. Statistical mechanics and stochastic optimization

## 12.1 SDE approximation

Approximate minibatch training as

`d theta = -P(theta) grad L dt + sqrt(2 T(theta)) B(theta) dW_t`.

Here `P` is a preconditioner and `B B^T` approximates gradient-noise covariance.

The effective temperature and diffusion are coordinate dependent unless transported properly.

## 12.2 What to measure

- gradient-noise covariance;
- drift vector;
- diffusion tensor;
- stationary basin occupancy where meaningful;
- escape times from controlled saddles;
- entropy production under non-reversible optimizers;
- path action under paired parameterizations;
- function-space diffusion rather than parameter-space diffusion alone.

## 12.3 Candidate insight

Equivalent parameterizations can induce different stochastic exploration even when deterministic gradients are locally related.

Adaptive optimizers further distort the effective metric through coordinatewise second moments.

The practical architecture gap may be a nonequilibrium property of the optimizer-coordinate system.

This is a hypothesis, not a result.

## 12.4 Experimental design

Use small models where full covariance and Hessian spectra are tractable.

Then test scalable estimators on larger models.

Freeze the estimator on small analytic cases before measuring large-model performance.

---

# 13. Causal model of an architecture experiment

## 13.1 Variables

Let:

- `A` denote architecture notation;
- `F` denote function class;
- `P` denote parameterization;
- `I` denote initialization prior;
- `O` denote optimizer;
- `R` denote regularizer;
- `H` denote hyperparameter search;
- `C` denote compute;
- `N` denote numerical implementation;
- `D` denote data and task;
- `Y` denote evaluation outcome.

The architecture label influences many of the others.

An ordinary leaderboard estimates a total bundled effect.

It does not identify the causal path.

## 13.2 Intervention lattice

Construct paired systems by progressively transporting:

1. forward semantics;
2. initialization;
3. data order;
4. raw gradients;
5. optimizer state;
6. regularization;
7. numerical precision;
8. stopping;
9. search policy;
10. compute.

At each rung, measure the remaining predictive and performance gap.

This produces a **causal comparator ladder**.

## 13.3 Shapley decomposition

Because factors interact, compute Shapley contributions over a feasible subset of interventions.

Use preregistered reference policies.

Report uncertainty across tasks and seeds.

Do not interpret the decomposition as unique outside its intervention set.

## 13.4 Mediation analysis

Potential mediators include:

- accumulated covariance defect;
- path length in Fisher geometry;
- local stability amplification;
- margin;
- calibration;
- gradient-noise scale;
- effective rank;
- representation norm;
- validation selection step.

Use cross-fitting.

Do not use the same cells to invent and confirm a mediator.

---

# 14. The Equivalence Compiler

## 14.1 Product definition

Build a system that accepts a model specification and an equivalence transformation, then emits:

- a mapped implementation;
- parameter conversion functions;
- optimizer-state conversion functions;
- forward equivalence tests;
- backward equivalence tests;
- update-conjugacy tests;
- resource accounting;
- adversarial numerical tests;
- a machine-readable equivalence certificate.

Working package name: `qneuro.equivalence`.

## 14.2 Certificate schema

Each certificate must include:

- source model identifier;
- target model identifier;
- transformation family;
- exact or approximate status;
- mathematical derivation reference;
- parameter-domain restrictions;
- input-domain restrictions;
- dtype;
- device;
- maximum forward residual;
- maximum logit residual;
- maximum probability residual;
- maximum loss residual;
- maximum gradient residual;
- maximum optimizer-state residual;
- maximum one-step update residual;
- exhaustive or sampled audit status;
- adversarial cases;
- code hashes;
- test hashes;
- timestamp;
- software environment;
- known failure modes.

## 14.3 Compiler architecture

Create:

`qneuro/equivalence/spec.py`

`qneuro/equivalence/maps.py`

`qneuro/equivalence/complex_real.py`

`qneuro/equivalence/quaternion_real.py`

`qneuro/equivalence/factorization.py`

`qneuro/equivalence/fourier.py`

`qneuro/equivalence/attention_gauge.py`

`qneuro/equivalence/optimizer_transport.py`

`qneuro/equivalence/regularizer_transport.py`

`qneuro/equivalence/stopping_transport.py`

`qneuro/equivalence/certificate.py`

`qneuro/equivalence/adversarial.py`

`qneuro/equivalence/quotient_metrics.py`

`qneuro/equivalence/stability.py`

`qneuro/equivalence/defects.py`

## 14.4 First supported map

Promote the existing complex-to-exact-real implementation into a formal map object.

Do not duplicate its algebra.

Wrap and test the existing mapping.

Add parameter round-trip tests.

Add state round-trip tests.

Add optimizer-moment transport tests.

Add fixed-minibatch multi-step conjugacy tests.

Add checkpoint conversion in both directions.

## 14.5 Numerical adversary

Generate cases near:

- zero magnitude;
- complex-tanh poles in finite precision;
- normalization epsilon boundaries;
- very large real and imaginary parts;
- cancellation regions;
- rank-deficient factors;
- repeated eigenvalues;
- nearly commuting operators;
- long sequence accumulation;
- mixed precision boundaries.

The certificate must report where equivalence degrades.

---

# 15. Benchmark families

## 15.1 Family A: complex versus real block

Models:

- Q-Neuro complex operator;
- exact structured-real block operator;
- native two-channel real operator;
- unrestricted real operator.

Questions:

- Does forward equivalence hold across all supported nonlinearities?
- Does SGD preserve paired training more closely than AdamW?
- Which optimizer-state component creates the first divergence?
- Does divergence predict final loss or only coordinate distance?
- Does native complex kernel behavior matter on CPU, MPS, and CUDA?

## 15.2 Family B: unitary versus orthogonal block

Map a complex unitary recurrence to its real orthogonal block representation.

Use long-memory tasks, controlled linear systems, and sequence modeling.

Measure norm preservation, gradient propagation, update covariance, and generalization.

## 15.3 Family C: quaternion versus four-channel real

Implement Hamilton-product structure and exact real block matrices.

Use 3D rotation data and non-geometric controls.

Test whether practical gains arise from structure, coordinate-specific optimization, or data symmetry.

## 15.4 Family D: Fourier versus time-domain parameterization

Construct exactly equivalent circular convolutions.

Compare frequency-domain complex coefficients with conjugate-symmetric real filters.

Transport initialization, weight decay, and optimizers.

Use signals with controlled spectral sparsity and off-grid shift.

## 15.5 Family E: dense versus factorized linear maps

Use `W = UV` versus direct `W`.

The represented function families match when rank is sufficient or restricted consistently.

Gradient descent induces different implicit biases.

This family tests whether the geometric-residue estimator generalizes beyond complex arithmetic.

## 15.6 Family F: matrix scaling symmetries

For homogeneous layers, use transformations such as

`W2 W1 = (W2 / c)(c W1)`.

Predictions remain fixed while Euclidean norms and sharpness can change.

Test optimizer covariance and quotient metrics across scaling orbits.

## 15.7 Family G: hidden-unit permutations

Permute neurons and compensate adjacent weights.

This is an exact discrete symmetry.

Properly implemented coordinatewise optimizers should be permutation equivariant when state is permuted too.

Use it as a zero-defect positive control.

## 15.8 Family H: attention gauges

Use exact query-key basis transformations that preserve attention logits under declared conditions.

Use value-output transformations that preserve the block function.

Audit interactions with LayerNorm, biases, RoPE, softmax precision, and fused kernels.

This is high-risk because the recent literature is close.

Treat it as validation, not an automatic novelty claim.

## 15.9 Family I: state-space representations

Compare diagonal complex SSMs with structured-real block SSMs.

Compare similar realizations of the same linear dynamical system under change of basis.

Use system-identification tasks where the exact transfer function is known.

## 15.10 Family J: coordinate-transformed neural ODEs

Apply smooth invertible state-coordinate maps.

Transport vector fields exactly.

Measure solver tolerance, adjoint error, and optimizer covariance separately.

## 15.11 Family K: canonical transformations in Hamiltonian models

Use symplectic coordinate changes that preserve physical trajectories.

Compare native Euclidean training and geometrically transported training.

Measure conservation and prediction in coordinate-invariant units.

## 15.12 Family L: normalization reparameterizations

Compare equivalent folded and unfolded affine-normalization layers at evaluation.

Training equivalence may fail because batch statistics change.

This family tests the compiler's ability to refuse an invalid equivalence claim.

## 15.13 Family M: probability-link reparameterizations

Compare redundant logit shifts and temperature-coordinate representations that preserve probabilities under constrained mapping.

Use calibration metrics and proper scoring rules.

## 15.14 Family N: tensor-network gauge freedom

Insert invertible matrices and their inverses on internal tensor-network bonds.

The contracted function remains fixed.

Test conditioning, canonicalization, and optimizer transport.

## 15.15 Family O: normalizing-flow coordinate changes

Use analytically invertible transformations with exact Jacobian accounting.

Compare parameterizations of identical densities.

This expands the program beyond discriminative prediction.

---

# 16. Experimental phases

## Phase 0: repository freeze

- Create a new preregistration for equivalence science.
- Do not modify the historical QN falsification records.
- Record the parent commit.
- Record the environment.
- Record the exact result-artifact hashes.
- Add the new claim ledger entries as prospective.
- Mark this planning document non-authoritative.

## Phase 1: exact Q-Neuro conjugacy

- Reconstruct one paired complex/exact-real checkpoint.
- Verify parameter-map round trip.
- Verify forward states at every evidence step.
- Verify logits.
- Verify probabilities.
- Verify loss.
- Verify raw gradients.
- Verify regularizer gradients.
- Inspect native AdamW state representation.
- Implement optimizer-state transport.
- Run one update.
- Run ten updates.
- Run one epoch.
- Run a complete deterministic tiny task.
- Identify the first nonzero discrepancy.
- Attribute it to a specific operation or numerical kernel.

## Phase 2: analytic microcosms

- Linear regression under invertible linear coordinates.
- Logistic regression under redundant coordinates.
- Two-layer matrix factorization.
- Complex scalar regression and exact realification.
- Unitary scalar recurrence.
- Homogeneous two-layer scaling orbit.
- Permutation symmetry control.
- Nonlinear diffeomorphic parameter map.
- Exact natural-gradient reference.
- Finite-step natural-gradient reference.
- Adam reference.
- AdamW reference.
- SGD reference.
- momentum reference.

## Phase 3: estimator discovery

- Define Euclidean update defect.
- Define Fisher-horizontal defect.
- Define function-space defect.
- Define local amplification.
- Define integrated geometric residue.
- Define initialization residual.
- Define regularization residual.
- Define stopping residual.
- Define search residual.
- Compare candidate estimators only on discovery families.
- Freeze one primary estimator.

## Phase 4: held-out equivalence families

- Seal at least five transformation families.
- Include one complex-real family.
- Include one factorization family.
- Include one permutation or scaling symmetry.
- Include one sequence model family.
- Include one continuous-dynamics family.
- Predict sign and magnitude before opening outcomes.
- Preserve failed predictions.

## Phase 5: practical architecture audits

- Choose published architecture claims with open code.
- Reproduce the claimed comparison.
- Construct the strongest semantics-equivalent or mechanism-stealing control.
- Match search and compute.
- Report how much of the headline gap survives.
- Seek author feedback before public criticism.
- Avoid accusatory language.

## Phase 6: scaling

- Start below 100,000 parameters.
- Advance only if the estimator is non-vacuous.
- Test 0.1M, 1M, 10M, and 100M scales subject to resources.
- Preserve a MacBook-compatible core benchmark.
- Use external compute only after the small-scale mechanism is frozen.
- Report scaling of defect, divergence, compute, and estimator error.

## Phase 7: independent replication

- Publish a minimal equivalence challenge.
- Provide only the theorem and interface, not hidden expected results.
- Ask independent teams to implement maps.
- Require different frameworks where possible.
- Compare PyTorch, JAX, and another stack.
- Include CPU and GPU implementations.
- Publish all divergences.

---

# 17. Primary hypotheses

## H0: no useful law

Generic stability bounds are too loose.

Measured covariance defects do not predict practical performance.

Architecture gaps remain idiosyncratic.

## H1: exact transport collapse

When the full training system is transported, semantics-equivalent implementations produce indistinguishable predictive trajectories.

## H2: optimizer geometry

Native practical optimizers create systematic divergence across equivalent parameterizations.

The divergence is predictable from update covariance defect.

## H3: prior and regularizer

Initialization and regularization transport explain more practical gap than optimizer update geometry.

## H4: numerical implementation

Kernel choice, dtype, and normalization dominate discrepancies in long sequential systems.

## H5: local stability

The same one-step covariance defect produces different final gaps because trajectory amplification differs.

Lyapunov-weighted defect predicts that difference.

## H6: quotient metric

Function-space or Fisher-horizontal metrics generalize better than Euclidean parameter metrics.

## H7: architecture residual

After all known transport operations, a residual performance difference remains because the systems were not actually function-class equivalent or because an unmodeled training component differs.

## H8: broad generality

One frozen estimator predicts across complex, factorized, Fourier, SSM, attention, and dynamical-system representations.

## H9: Q-Neuro-specific artifact

The relationship works only in Q-Neuro due to its normalization, task construction, or small scale.

## H10: research-audit utility

Even if no universal quantitative law exists, the equivalence compiler materially improves the validity of architecture comparisons.

---

# 18. Endpoints

## 18.1 Primary mathematical endpoint

A proven finite-horizon bound with explicit assumptions, exact residual definitions, and at least one non-vacuous regime.

## 18.2 Primary empirical endpoint

Out-of-family predictive accuracy of the frozen geometric-residue model for final paired performance gaps.

Use:

- held-out `R^2`;
- mean absolute error;
- sign accuracy;
- calibration of predicted versus observed magnitude;
- coverage of predicted upper bounds;
- improvement over simple baselines.

## 18.3 Co-primary integrity endpoint

Rate of false equivalence certificates.

The compiler must reject deliberately invalid mappings.

## 18.4 Secondary endpoints

- stepwise logit divergence;
- stepwise probability divergence;
- loss divergence;
- gradient divergence;
- optimizer-state divergence;
- Fisher path length;
- horizontal versus vertical update energy;
- directional amplification;
- final task metric gap;
- calibration gap;
- robustness gap;
- sample-efficiency gap;
- compute gap;
- memory gap;
- energy gap;
- reproducibility across frameworks;
- certificate generation time.

## 18.5 Negative-control endpoints

- permutation symmetry should have near-zero defect when optimizer state is permuted;
- deliberate optimizer-state mismatch should create detectable defect;
- invalid nonlinear maps should fail certificate tests;
- shuffled task outcomes should destroy performance-gap prediction;
- architecture labels alone should not appear competitive because of leakage;
- random residual features should not survive held-out families.

---

# 19. Statistical design

## 19.1 Top-level unit

The top-level unit is an independently specified equivalence problem within a transformation family.

Training seeds are repeated measurements.

Datasets or worlds are nested where appropriate.

Do not treat batches or examples as independent evidence for the law.

## 19.2 Split structure

Use four layers:

1. implementation fixtures;
2. estimator discovery transformations;
3. frozen confirmation transformations;
4. independent replication transformations.

No transformation family may move backward after outcomes are observed.

## 19.3 Sample-size logic

Power the main study on top-level transformation problems.

Do not power on millions of prediction rows.

Use simulation based on pilot heterogeneity.

Target useful precision for MAE and held-out correlation, not only a rejection p-value.

## 19.4 Baselines for the law predictor

- zero gap;
- mean discovery gap;
- architecture-family mean;
- optimizer-family mean;
- learning rate;
- total gradient norm;
- training loss decrease;
- parameter count;
- condition number;
- raw one-step prediction divergence;
- cumulative unweighted defect;
- cumulative stability-weighted defect.

## 19.5 Hierarchical model

Use a model with random intercepts or partial pooling for transformation family, task family, and implementation framework.

Allow optimizer interactions.

Keep the confirmatory fixed formula simple.

Use richer models only in discovery.

## 19.6 Uncertainty

Report:

- bootstrap intervals over top-level problems;
- leave-one-family-out sensitivity;
- exact or Monte Carlo sign-flip tests where paired units permit;
- posterior predictive checks if a Bayesian model is used;
- uncertainty in local amplification estimates;
- measurement-error sensitivity.

## 19.7 Multiplicity

Use one primary estimator and one primary held-out score.

Correct named secondary hypothesis families.

Treat model-search results as exploratory.

## 19.8 Missing and failed runs

Use intention-to-run reporting.

Failure rate is an adverse outcome.

Do not exclude divergence because it is inconvenient.

Separate mathematical-domain violations from infrastructure failure.

---

# 20. Q-Neuro implementation plan

## 20.1 Preserve historical modules

Do not rewrite the existing operator models merely to fit the new abstraction.

Wrap them.

Historical reproducibility depends on their current behavior.

## 20.2 Add new configuration

Create `experiments/configs/equivalence_transport_prereg.yaml`.

Include:

- immutable seed namespaces;
- transformation families;
- discovery and confirmation assignment;
- model constructors;
- map identifiers;
- optimizer contracts;
- regularizer contracts;
- stopping contracts;
- precision contracts;
- resource budgets;
- endpoints;
- thresholds;
- exclusion rules;
- hashes.

## 20.3 Add a new registry namespace

Use experiment IDs `QE-000001`, `QE-000002`, and so on.

Do not overload the QN series.

Each run records source and target system IDs.

## 20.4 Add tests before experiments

Create tests for:

- map composition;
- map inverse;
- parameter round trip;
- state round trip;
- exact forward equivalence;
- intermediate-state equivalence;
- loss equivalence;
- gradient pullback;
- optimizer-state conversion;
- conjugate update;
- certificate serialization;
- deliberate certificate failure;
- dtype boundaries;
- deterministic seeds;
- clean-worktree provenance;
- artifact hashing.

## 20.5 Extend compute accounting

Record:

- theoretical FLOPs by operation;
- measured kernel time;
- optimizer-state bytes;
- activation bytes;
- peak resident memory;
- device energy where reliable;
- compilation time;
- data-loading time;
- evaluation time;
- total search cost.

## 20.6 Raw trajectory artifacts

Every paired run must emit:

- mapped initial parameters;
- batch identifiers;
- per-step losses;
- per-step logits on a fixed probe set;
- per-step gradient summaries;
- per-step optimizer-state summaries;
- per-step covariance defects;
- local amplification estimates;
- checkpoint hashes;
- final raw predictions;
- failure events.

Use compressed arrays, not giant JSON where practical.

Include schema versions.

## 20.7 Analysis outputs

Generate:

- defect-versus-time curves;
- prediction-divergence curves;
- stability-weighted contribution plots;
- observed-versus-predicted gap plots;
- coverage plots for the bound;
- intervention-ladder waterfall plots;
- family-level forest plots;
- resource Pareto plots;
- negative-control dashboards;
- certificate summaries.

---

# 21. First twelve registered experiments

## QE-000001: complex-real forward certificate

Purpose: formalize the existing exact-real mapping.

No training.

Test all supported state operations and readouts.

Success: tolerance-qualified forward, state, loss, and gradient certificate.

Failure: any unexplained mismatch above dtype-specific threshold.

## QE-000002: SGD conjugacy

Purpose: test mapped initialization and plain SGD across complex and exact-real implementations.

Use deterministic full-batch training first.

Success: multi-step predictive trajectory agreement.

Failure: unexplained covariance defect.

## QE-000003: AdamW state audit

Purpose: locate divergence caused by native complex versus real adaptive moments.

Log moment tensors and packed real views.

No final-performance headline.

## QE-000004: transported AdamW

Purpose: implement conjugate optimizer-state updates where mathematically defined.

Compare native and transported variants.

Primary endpoint: stepwise predictive divergence.

## QE-000005: regularization ladder

Purpose: separate native weight decay from transported function-equivalent regularization.

Use fixed steps and mapped initialization.

## QE-000006: stopping ladder

Purpose: measure divergence created by early stopping and checkpoint ties.

Compare fixed-step and selected checkpoints.

## QE-000007: numerical stress test

Purpose: map equivalence error across dtype, sequence length, normalization scale, and device.

No task-accuracy claim.

## QE-000008: linear analytic transport

Purpose: verify the finite-horizon bound in a system with exact Jacobians and known Lipschitz constants.

## QE-000009: matrix-factorization transfer

Purpose: test whether the estimator generalizes beyond complex coordinates.

## QE-000010: permutation zero control

Purpose: show that correctly transported discrete symmetry creates near-zero measured residue.

## QE-000011: invalid-map negative control

Purpose: ensure the certificate catches a map that matches on random inputs but fails adversarially.

## QE-000012: sealed pilot law test

Purpose: freeze one geometric-residue estimator and predict held-out paired gaps across at least three families.

This remains a pilot and cannot establish a law.

---

# 22. Theorem agenda

## Theorem T1: exact realification functor

Formalize realification as an algebra-preserving map for the supported complex operator category.

Prove preservation of addition, composition, adjoint, and norm.

Mark known components as known.

The repository-specific result is an implementation correctness theorem.

## Theorem T2: nonlinear state-update equivalence

State sufficient conditions on injection, activation, normalization, and readout for exact Q-Neuro step equivalence.

Handle epsilon normalization explicitly.

Handle the complex activation explicitly.

## Theorem T3: discrete training conjugacy

Prove trajectory equivalence for a mapped joint model-optimizer state under conjugate updates.

Include stochastic coupling.

## Theorem T4: covariance-defect stability bound

Prove the finite-horizon recursive bound.

State local and global variants.

Separate parameter-state and predictive-state bounds.

## Theorem T5: Fisher-horizontal refinement

Derive a quotient-aware version that removes vertical symmetry directions.

Specify singular-metric treatment.

## Theorem T6: stochastic transport bound

Bound expected predictive divergence under coupled stochastic updates.

Include martingale noise or Wasserstein formulation.

## Theorem T7: optimizer-specific covariance

Characterize transformations under which SGD, momentum, Adam, AdamW, RMSProp, and natural gradient are equivariant.

Do not assume a single answer.

## Theorem T8: regularizer transport

Prove how explicit penalties transform.

Show counterexamples for naive weight decay.

## Theorem T9: early-stopping discontinuity

Show how arbitrarily small trajectory differences can produce finite checkpoint differences under discrete selection.

Define stable tie-breaking conditions.

## Theorem T10: resource non-equivalence

Formalize why semantic equivalence does not imply computational equivalence.

Use a Pareto partial order rather than a scalar theorem.

## Theorem T11: lower bounds or impossibility

Search for conditions under which no coordinate-invariant finite-step first-order optimizer exists under a chosen computational budget.

This may be more novel than the upper bound.

## Theorem T12: identifiability of decomposition

State when the factorial intervention lattice identifies optimizer, prior, regularizer, and numerical contributions.

Give counterexamples with unmeasured interactions.

---

# 23. Counterexample agenda

The project must actively search for counterexamples.

## 23.1 Vacuous bound

Construct an unstable linear system where global Lipschitz products explode while actual predictive divergence remains small.

## 23.2 Vertical defect

Construct large parameter movement entirely within a functional symmetry orbit.

Euclidean defect is large.

Predictive defect is zero.

## 23.3 Small local defect, large selection gap

Create validation ties where a tiny numerical difference selects different checkpoints.

## 23.4 Zero one-step defect, later divergence

Create an incomplete metric that misses optimizer-state mismatch.

## 23.5 Large defect, no task gap

Create divergent logits on irrelevant classes while top-1 remains unchanged.

This shows why proper scoring rules and prediction metrics are required.

## 23.6 Same function class, different reachable set

Use a singular or non-surjective parameter map.

Show that apparent equivalence at initialization does not imply global training equivalence.

## 23.7 Non-smooth map

Use a coordinate map with singularities.

Show failure of differential transport.

## 23.8 Stochastic decoupling

Use dropout masks that cannot be mapped naively.

## 23.9 Numerical branch mismatch

Use mathematically equivalent expressions with different overflow and cancellation behavior.

## 23.10 Search-induced reversal

Give one parameterization more favorable hyperparameter coordinates under the same nominal search distribution.

---

# 24. Real data without medical overclaim

## 24.1 Why external data matter

A law about learning systems should not depend on NeuroWorld.

Real data introduce noise, scale, modality, preprocessing, and implementation variation absent from synthetic generators.

## 24.2 Safe initial domains

- system identification;
- audio sequence classification;
- nonclinical sensor fault detection;
- weather or energy forecasting;
- character-level language modeling;
- image classification for Fourier/convolution equivalence;
- molecular-property prediction where licenses permit;
- algorithmic sequence tasks with analytic ground truth.

## 24.3 Medical boundary

Do not start patient-data experiments without governance, licensing, privacy, intended-use, bias, and clinical-collaborator review.

Even retrospective medical performance would not establish deployment safety.

The equivalence-audit tool can become medically relevant without claiming to diagnose patients.

Its near-term medical value would be improving the validity of model comparison.

---

# 25. Publication strategy

## 25.1 Paper one

Working title:

**When Architectures Are Coordinates: Falsifying Representation Advantage with Transported Equivalence Controls**

Core content:

- the Q-Neuro case study;
- the exact-real comparator ladder;
- the training-system definition;
- the equivalence certificate;
- the transport-defect bound;
- analytic and small-scale experiments;
- failures and scope.

Do not call this a Nature paper in the manuscript.

Submit where the evidence fits.

## 25.2 Paper two

Working title:

**An Equivalence Compiler for Architecture Claims**

Core content:

- software system;
- multiple equivalence families;
- adversarial certificates;
- audits of published comparisons;
- reproducibility across frameworks.

## 25.3 Paper three

Only if confirmed:

**Predicting Optimization Gaps Across Equivalent Neural Parameterizations**

Core content:

- frozen estimator;
- held-out families;
- scaling;
- independent replication;
- counterexamples;
- quantitative law.

## 25.4 Evidence required for a top general-science venue

- broad conceptual importance;
- rigorous theorem;
- multiple real domains;
- strong baselines;
- large-scale relevance;
- independent replication;
- public code and raw artifacts;
- a result surprising to domain experts;
- no dependence on branding;
- clear practical consequences.

## 25.5 Authorship and age

Being a high-school student is a human story.

It is not evidence for a scientific claim.

The paper should be judged on correctness.

Do not use age to shield the work from criticism.

Do not conceal age if Samyar wants it in a biography or press context.

Keep it out of the causal argument.

---

# 26. Claim language

## 26.1 Allowed now

“Q-Neuro's strongest exact-real control removes the earlier intrinsic-complex interpretation within the implemented model family.”

“The negative result motivates an equivalence-aware audit of architecture comparisons.”

“We propose a transport-defect framework and a falsifiable empirical conjecture.”

“The framework uses complex realification as its first case study.”

## 26.2 Not allowed now

“Shafiee's law has been discovered.”

“Q-Neuro is the most powerful medical AI.”

“Q-Neuro outcompetes every AI.”

“Q-Neuro will win a Nobel Prize.”

“Quantum mechanics explains diagnosis.”

“The model is clinically validated.”

“The complex architecture is fundamentally more expressive.”

## 26.3 Allowed after theorem only

“Under assumptions A through F, the accumulated covariance-defect bound holds.”

This still does not permit an empirical-law claim.

## 26.4 Allowed after preregistered confirmation

“The frozen geometric-residue estimator predicted held-out gaps across the declared families with the reported accuracy.”

This still does not permit universal language.

## 26.5 Allowed after independent replication

“Independent implementations reproduced the direction and approximate magnitude in the tested domains.”

Scope remains mandatory.

---

# 27. Claude Code execution protocol

## 27.1 Before editing

- Read `docs/CLAIMS.md`.
- Read `docs/FAILED_IDEAS.md`.
- Read `docs/NEXT_PHASE_AUDIT.md`.
- Read `docs/PREREGISTRATION_NEXT_PHASE.md`.
- Read `docs/PROVISIONAL_LAW_FREEZE.md`.
- Read `docs/MATHEMATICAL_FRAMEWORK_V2.md`.
- Read `docs/PRIOR_ART_V2.md`.
- Read `RESEARCH_LOG.md`.
- Read `ROADMAP.md`.
- Inspect `research/claims.json`.
- Inspect the QN-000033 result.
- Inspect the QN-000040 result.
- Inspect the QN-000042 result.
- Inspect QN-GRAND-001 preflight.
- Run the full test suite.
- Confirm the worktree state.

## 27.2 First output

Produce an audit memo, not code.

The memo must answer:

- Which exact maps already exist?
- Which maps lack optimizer transport?
- How are complex parameters represented by the installed PyTorch optimizer?
- Which historical paired seeds can be reconstructed?
- Which raw predictions are missing?
- Which artifact schemas should be migrated?
- What is the smallest deterministic conjugacy experiment?
- What prior work most threatens novelty?

## 27.3 First code change

Implement only the equivalence specification and certificate interface.

Do not launch a large training sweep.

Write failing tests first for unsupported equivalence claims.

## 27.4 First experiment

QE-000001 is deterministic and training-free.

It must finish on the M2 MacBook.

It must emit a certificate.

It must include adversarial numerical cases.

## 27.5 Advancement rule

Do not advance from exact equivalence to empirical law fitting until:

- parameter maps round-trip;
- state maps round-trip;
- forward equivalence passes;
- loss equivalence passes;
- gradient pullback passes;
- one-step update behavior is explained;
- raw artifacts are complete;
- the preregistration is frozen.

---

# 28. Atomic research backlog

The following backlog is intentionally explicit.

Each item should become an issue, test, experiment, proof obligation, or documented rejection.

## 28.1 Repository integrity

- [ ] Confirm parent commit hash.
- [ ] Record current branch.
- [ ] Record remote URL.
- [ ] Run `git status --short`.
- [ ] Verify historical result directories are immutable by policy.
- [ ] Hash all historical result summaries.
- [ ] Hash all frozen configurations.
- [ ] Hash all law records.
- [ ] Verify QN-GRAND-001 remains sealed.
- [ ] Verify no raw confirmation predictions were accidentally added elsewhere.
- [ ] Validate experiment registry against result directories.
- [ ] Validate figure sources against summary files.
- [ ] Validate paper tables against result summaries.
- [ ] Preserve the current paper build.
- [ ] Create a versioned next-program branch.

## 28.2 Exact map specification

- [ ] Define `EquivalenceLevel`.
- [ ] Define `TransportLevel`.
- [ ] Define `ParameterMap` protocol.
- [ ] Define `StateMap` protocol.
- [ ] Define `OptimizerStateMap` protocol.
- [ ] Define `RegularizerMap` protocol.
- [ ] Define `PredictionMetric` protocol.
- [ ] Define `ResourceContract`.
- [ ] Define domain restrictions.
- [ ] Define dtype restrictions.
- [ ] Define device restrictions.
- [ ] Define inverse-map status.
- [ ] Define map-composition semantics.
- [ ] Define certificate schema version.
- [ ] Define failure severity.

## 28.3 Complex parameter mapping

- [ ] Inventory every complex parameter in `ComplexOperatorState`.
- [ ] Inventory every exact-real target parameter.
- [ ] Match shapes in real scalar degrees of freedom.
- [ ] Derive embedding map.
- [ ] Derive low-rank left-factor map.
- [ ] Derive low-rank right-factor map.
- [ ] Derive state-injection map.
- [ ] Derive initial-state map.
- [ ] Derive normalization map.
- [ ] Derive readout map.
- [ ] Derive demographic-context map.
- [ ] Derive masking behavior.
- [ ] Derive padding behavior.
- [ ] Implement source-to-target checkpoint conversion.
- [ ] Implement target-to-source checkpoint conversion.
- [ ] Test exact round trip.

## 28.4 Forward equivalence

- [ ] Test empty-padding batch behavior.
- [ ] Test one-token sequences.
- [ ] Test maximum supported length.
- [ ] Test repeated tokens.
- [ ] Test positive evidence.
- [ ] Test negative evidence.
- [ ] Test missing evidence.
- [ ] Test chronology twins.
- [ ] Test all-zero initial state edge case.
- [ ] Test random states.
- [ ] Test adversarial cancellation.
- [ ] Test large magnitude.
- [ ] Test small magnitude.
- [ ] Test normalization epsilon boundary.
- [ ] Compare each intermediate state.
- [ ] Compare amplitudes.
- [ ] Compare logits.
- [ ] Compare probabilities.
- [ ] Compare masks.
- [ ] Compare final predictions.

## 28.5 Backward equivalence

- [ ] Compare loss values.
- [ ] Compare logit gradients.
- [ ] Compare state gradients.
- [ ] Compare embedding gradients.
- [ ] Compare factor gradients.
- [ ] Compare readout gradients.
- [ ] Verify conjugate-gradient convention.
- [ ] Verify Wirtinger interpretation.
- [ ] Verify PyTorch complex-autograd convention.
- [ ] Compare finite differences.
- [ ] Compare double-precision reference.
- [ ] Test batch reductions.
- [ ] Test masked reductions.
- [ ] Test regularization gradients.
- [ ] Record maximum relative and absolute errors.

## 28.6 Optimizer audit

- [ ] Inspect SGD parameter update.
- [ ] Inspect momentum buffer representation.
- [ ] Inspect Nesterov behavior.
- [ ] Inspect Adam first moment.
- [ ] Inspect Adam second moment.
- [ ] Inspect AMSGrad state.
- [ ] Inspect AdamW decoupled decay.
- [ ] Inspect epsilon placement.
- [ ] Inspect bias correction.
- [ ] Inspect complex tensor real-view behavior.
- [ ] Inspect fused optimizer availability.
- [ ] Inspect foreach optimizer behavior.
- [ ] Inspect MPS optimizer behavior.
- [ ] Inspect CPU optimizer behavior.
- [ ] Implement state packing.
- [ ] Implement state unpacking.
- [ ] Test one-step transport.
- [ ] Test ten-step transport.
- [ ] Test checkpoint resume transport.
- [ ] Test scheduler transport.
- [ ] Test gradient clipping transport.

## 28.7 Regularization audit

- [ ] Map L2 parameter penalty.
- [ ] Map decoupled weight decay.
- [ ] Map spectral norm penalty.
- [ ] Map commutator penalty.
- [ ] Map phase penalty.
- [ ] Map state-norm penalty.
- [ ] Map readout penalty.
- [ ] Distinguish native and transported penalties.
- [ ] Test penalty-value equality.
- [ ] Test penalty-gradient equality.
- [ ] Document non-invariant regularizers.

## 28.8 Search-policy audit

- [ ] Define hyperparameter coordinates.
- [ ] Define source search distribution.
- [ ] Define mapped target search distribution.
- [ ] Record every attempted trial.
- [ ] Seal shifted metrics during selection.
- [ ] Use deterministic tie breaks.
- [ ] Report search cost.
- [ ] Report failed trials.
- [ ] Compare equal-count and transported search.
- [ ] Test sensitivity to search-space scale.

## 28.9 Numerical audit

- [ ] Test float64 CPU.
- [ ] Test float32 CPU.
- [ ] Test float32 MPS.
- [ ] Test bfloat16 where supported.
- [ ] Test mixed precision where supported.
- [ ] Disable fused kernels for reference.
- [ ] Enable fused kernels for practical comparison.
- [ ] Record deterministic-algorithm status.
- [ ] Record BLAS backend.
- [ ] Record PyTorch version.
- [ ] Record hardware.
- [ ] Test repeated-run bit stability.
- [ ] Measure accumulation with sequence length.
- [ ] Measure normalization drift.
- [ ] Measure underflow.
- [ ] Measure overflow.
- [ ] Measure cancellation.

## 28.10 Stability estimation

- [ ] Compute exact Jacobians on tiny models.
- [ ] Compute power-iteration spectral norms.
- [ ] Compute directional Jacobian-vector products.
- [ ] Compute finite-time Lyapunov estimates.
- [ ] Compare global and local bounds.
- [ ] Compare parameter and prediction metrics.
- [ ] Project vertical symmetry directions.
- [ ] Estimate Fisher metric.
- [ ] Regularize singular Fisher blocks transparently.
- [ ] Report estimator variance.
- [ ] Freeze amplification estimator.

## 28.11 Analytic toy suite

- [ ] Scalar quadratic under linear rescaling.
- [ ] Two-dimensional quadratic under rotation.
- [ ] Ill-conditioned quadratic under whitening.
- [ ] Logistic regression under invertible features.
- [ ] Complex scalar linear regression.
- [ ] Complex scalar nonlinear regression.
- [ ] Matrix factorization with exact product equivalence.
- [ ] Homogeneous network scaling orbit.
- [ ] Hidden-unit permutation orbit.
- [ ] Linear recurrence under similarity transform.
- [ ] Unitary recurrence under realification.
- [ ] Hamiltonian oscillator under canonical transform.
- [ ] Neural ODE under state diffeomorphism.
- [ ] Deliberately singular map.
- [ ] Deliberately non-smooth map.

## 28.12 Metrics

- [ ] Absolute parameter residual.
- [ ] Relative parameter residual.
- [ ] Horizontal parameter residual.
- [ ] Vertical parameter residual.
- [ ] Logit infinity-norm residual.
- [ ] Probability total variation.
- [ ] Jensen-Shannon divergence.
- [ ] Predictive KL with support safeguards.
- [ ] Fisher-Rao local distance.
- [ ] Loss residual.
- [ ] Gradient cosine.
- [ ] Gradient pullback residual.
- [ ] Optimizer-state residual.
- [ ] One-step covariance defect.
- [ ] Accumulated unweighted defect.
- [ ] Accumulated stability-weighted defect.
- [ ] Bound coverage.
- [ ] Bound tightness ratio.
- [ ] Final metric gap.
- [ ] Calibration gap.
- [ ] Resource gap.

## 28.13 Statistical pipeline

- [ ] Define top-level unit.
- [ ] Define paired hierarchy.
- [ ] Define discovery families.
- [ ] Define confirmation families.
- [ ] Define independent replication role.
- [ ] Simulate power.
- [ ] Freeze minimum effect.
- [ ] Freeze held-out scores.
- [ ] Freeze baseline predictors.
- [ ] Implement hierarchical bootstrap.
- [ ] Implement family leave-one-out.
- [ ] Implement sign-flip test.
- [ ] Implement bound-coverage interval.
- [ ] Implement measurement-error sensitivity.
- [ ] Implement failure-rate analysis.
- [ ] Test statistic determinism.
- [ ] Test no held-out leakage.

## 28.14 Visualization

- [ ] Equivalence map diagram.
- [ ] Comparator ladder diagram.
- [ ] Training-system bundle diagram.
- [ ] Quotient-space schematic.
- [ ] Vertical-horizontal decomposition schematic.
- [ ] Per-step covariance-defect curve.
- [ ] Per-step prediction-divergence curve.
- [ ] Local amplification curve.
- [ ] Contribution waterfall.
- [ ] Observed-versus-bound plot.
- [ ] Observed-versus-predicted plot.
- [ ] Family forest plot.
- [ ] Optimizer heatmap.
- [ ] Dtype heatmap.
- [ ] Sequence-length scaling plot.
- [ ] Resource Pareto front.
- [ ] Certificate failure map.
- [ ] Counterexample gallery.

## 28.15 Documentation

- [ ] New preregistration.
- [ ] Mathematical specification.
- [ ] Prior-art systematic-search protocol.
- [ ] Equivalence compiler guide.
- [ ] Certificate schema guide.
- [ ] Optimizer transport guide.
- [ ] Numerical reproducibility guide.
- [ ] Benchmark data card.
- [ ] Model cards.
- [ ] Failure registry update.
- [ ] Claim ledger update.
- [ ] Research log update.
- [ ] README scientific pivot.
- [ ] Contribution guide.
- [ ] Independent replication guide.

## 28.16 Publication hygiene

- [ ] Separate observed results from proposals.
- [ ] Mark exploratory plots.
- [ ] Mark confirmatory plots.
- [ ] Link every table cell to an artifact.
- [ ] Publish raw paired predictions.
- [ ] Publish failed certificates.
- [ ] Publish excluded-domain reasons.
- [ ] Include compute totals.
- [ ] Include carbon or energy caveats where measured.
- [ ] Include author contributions.
- [ ] Include conflicts of interest.
- [ ] Include data licenses.
- [ ] Include code license.
- [ ] Include reproducibility commands.
- [ ] Include exact environment lock.

---

# 29. Decision gates

## Gate A: exactness

Advance only if the source-target map has a valid certificate.

If exactness fails, downgrade the equivalence level.

Do not hide the downgrade.

## Gate B: conjugacy

Advance only after the first update discrepancy is either eliminated or mathematically explained.

## Gate C: non-vacuous bound

Advance to law discovery only if the bound is within a prespecified multiplicative range of observed predictive divergence on analytic cases.

## Gate D: cross-family signal

Advance to confirmation only if the estimator beats simple baselines in discovery across more than one equivalence family.

## Gate E: frozen confirmation

One attempt.

No coefficient changes after opening.

## Gate F: scale

Advance beyond MacBook-scale models only if the measurement cost and predictive value justify it.

## Gate G: independent replication

No law naming without external implementation.

## Gate H: medical relevance

No patient-data claim without the full governance gate.

---

# 30. Expected ways this program may fail

The transport bound may be mathematically correct but uselessly loose.

The best defect metric may reduce to one-step prediction divergence.

The apparent cross-family law may vanish on attention or factorization.

Fisher estimation may be too expensive or unstable.

Vertical-horizontal decomposition may be ambiguous near singular strata.

Optimizer state may not admit a simple exact transport.

Finite-step discretization may dominate continuous geometry.

Early stopping and hyperparameter search may introduce discontinuities that resist smooth bounds.

Numerical kernels may dominate small models but disappear at scale.

The literature may already contain the main theorem.

Independent groups may not reproduce the effect.

The compiler may become useful engineering without a universal law.

Any of those outcomes must be published honestly.

---

# 31. Why this can still matter

Machine-learning papers routinely compare named architectures whose differences bundle representation, parameterization, optimizer compatibility, initialization, regularization, kernel efficiency, and search policy.

An equivalence compiler can turn some of those bundles into controlled interventions.

It can reveal three distinct outcomes:

1. exact collapse, showing that a supposed representational advantage was an implementation artifact;
2. training divergence, showing that the practical advantage belongs to optimizer-coordinate interaction;
3. persistent residual, motivating a search for genuine function-class or inductive-bias differences.

Q-Neuro already experienced the first transition.

That makes it an unusually honest case study.

The negative result is not baggage to overcome.

It is the empirical reason this program exists.

---

# 32. Final instruction to Claude Code

Do not try to make Q-Neuro look victorious.

Make it impossible to fool.

Begin with the exact complex-real pair.

Treat the entire training process as the experimental object.

Compile equivalences.

Transport what can be transported.

Measure what fails to transport.

Bound how the failure propagates.

Predict held-out consequences.

Search for counterexamples.

Preserve every negative result.

If a broad quantitative law survives, the community may eventually give it a name.

If it does not survive, release the compiler and the falsification evidence.

Either outcome advances science more than an unsupported revolutionary claim.

---

# Appendix A. Repository manifest

The manifest below is generated from the tracked repository state and is included so a successor can account for every artifact.

Each entry should be rechecked against the current commit before work begins.

Manifest columns are path, newline count, byte count, and Git blob hash.

| Path | Lines | Bytes | Blob |
|---|---:|---:|---|
| `.github/workflows/ci.yml` | 31 | 674 | `47e9cff329715af4195c8b95382cd9a0ddeef924` |
| `.gitignore` | 18 | 269 | `49faa24c58cb9c64855d08553175c3997059a0e3` |
| `ARCHITECTURE.md` | 56 | 2822 | `8c02e573da1d66fd9714bf839f941c81ee2f4f3a` |
| `CITATION.cff` | 23 | 915 | `6080c74442cb2517c91eafe04f732c780cf74b26` |
| `CONTRIBUTING.md` | 13 | 728 | `1ff2b366e49ebcf293a84754fe9b2dd300ed44d0` |
| `LICENSE` | 21 | 1071 | `09079db4d3fe7a31892fc72ba166e9c39b0b20f8` |
| `Makefile` | 183 | 7021 | `b0d6747a1bec0aea13063fd3d2b3a16928520541` |
| `README.md` | 94 | 5181 | `de9eeae26a4c06c470e25c44bcb991d803700a43` |
| `RELEASE_NOTES.md` | 38 | 2014 | `1a9c525daf92ae1ffa6fa4292a90846d6468b430` |
| `REPLICATION.md` | 110 | 4494 | `158574c4705fd905a208ac783f9f64abedf64e56` |
| `RESEARCH_LOG.md` | 407 | 24361 | `2a342ebaed914838e44bbd806eea24d37e4640e5` |
| `RESULTS.md` | 618 | 37993 | `c8f1f5fcc491c99b569fdd07beb46cb9b2434418` |
| `ROADMAP.md` | 31 | 1640 | `ff8110c94090b4c0b97699b54d94831ff11d6c61` |
| `SECURITY.md` | 8 | 459 | `5587b7614d69b3a74b3ae65060fefce4086b7cde` |
| `dashboard/app.js` | 112 | 8030 | `5b6ae6644740d56c9a0ffe5c4fc17bbecdc0b272` |
| `dashboard/data.js` | 2116 | 78910 | `c5464b12e8c26c42abd4f141afdd0aa7741fd649` |
| `dashboard/index.html` | 86 | 6575 | `7319ea9388ce2e69f0d60f04172c7b7e915c46a8` |
| `dashboard/styles.css` | 142 | 13642 | `d47dfd09eb26785dd52ea460f3c617aa22488439` |
| `docs/ARCHITECTURE_CANDIDATES.md` | 103 | 5321 | `ef068d1b66fcff23b849252ef07c658ff309c163` |
| `docs/CLAIMS.md` | 66 | 13553 | `08a2bce31caa41e705eed7a182e29134f82e90d7` |
| `docs/CLINICAL_VALIDATION_ROADMAP.md` | 74 | 3797 | `c74dec7396f7f0b3690a53c004ce32eb199fc2bc` |
| `docs/EXTERNAL_DATASETS.md` | 36 | 1961 | `1365d224ed7e1d34fe7ac0cda3e9df1a868bd53b` |
| `docs/FAILED_IDEAS.md` | 274 | 14894 | `7170848c72d869885e7d7a12dd456b43f662c672` |
| `docs/INDEPENDENT_TASKS.md` | 30 | 1491 | `2c0d4f92a61db1259caf1deb8ca7ce46d75bb11b` |
| `docs/INDEPENDENT_TASK_AMENDMENT_001.md` | 25 | 1419 | `a1571943754f7b1db5f76aa24e3060dee4df70c0` |
| `docs/INDEPENDENT_TASK_AMENDMENT_002.md` | 22 | 936 | `8576210307e5cddc956330f161e82eb0e4b45dbb` |
| `docs/MATHEMATICAL_FRAMEWORK_V2.md` | 96 | 4479 | `ac3168c75e8d848a7839d83cd949536deb2d941e` |
| `docs/MATHEMATICS.md` | 308 | 13169 | `9ea54469bec61dbb4d42af40b0dd279a14b358c7` |
| `docs/MECHANISM_PROTOCOL.md` | 38 | 2383 | `63fc746be03378024b16332dfeb59c9165ff7321` |
| `docs/MODEL_CARD.md` | 54 | 2773 | `ea4ddc59d5a6f8d33c2d661b2528bfa1f6100528` |
| `docs/NEXT_PHASE_AUDIT.md` | 466 | 27024 | `80a81792d4459bc482c43970401b226bb9bd9136` |
| `docs/PILOT_PROTOCOL.md` | 30 | 1731 | `f84ccc11d8aad2386537d87a13b021faf9dda77c` |
| `docs/PREREGISTRATION_AMENDMENT_001.md` | 42 | 2194 | `c848eaf8e226a99bed24ad41c8f37afb46d9d413` |
| `docs/PREREGISTRATION_NEXT_PHASE.md` | 415 | 19103 | `0fc48b814beb1addbe336e3aea0e4f849268b733` |
| `docs/PRIOR_ART.md` | 74 | 11117 | `518a52afd84af893c2b04cf89b6ecebd599b42ad` |
| `docs/PRIOR_ART_V2.md` | 60 | 6175 | `b01b5a44b30634e0c523566a836413afdd30b880` |
| `docs/PROVISIONAL_LAW_FREEZE.md` | 55 | 2783 | `99b6c079c8a4fc214085cae831a89b5779d78dd9` |
| `docs/QN_GRAND_PREFLIGHT_AMENDMENT_001.md` | 30 | 1266 | `71e3a89562f3a79d8cf8b6cf19bc5eb7ef68f3b5` |
| `docs/RESEARCH_QUESTIONS.md` | 71 | 3219 | `482244637f17fb1627208d5f090080fafb8281f4` |
| `docs/SHIFT_GAUNTLET.md` | 49 | 3285 | `c26eced39527a3e8c91d62a29693a331c6435efc` |
| `experiments/__init__.py` | 1 | 52 | `27508b309d076f6521cd78628e4b4d054fde5202` |
| `experiments/configs/ablation_suite.yaml` | 69 | 1750 | `007c9a3183f54a1bf626df01b51974c2e56fa80d` |
| `experiments/configs/active_evidence.yaml` | 48 | 1110 | `ed1e613192ec24805f5764a05417804b5ac46522` |
| `experiments/configs/computational_law_suite.yaml` | 21 | 865 | `5fdb33d3bca2d6314d3862763cee73fb14c77ed6` |
| `experiments/configs/discovery_engine.yaml` | 45 | 2416 | `cbc607469bc4893db6da85e18451e270505d71b9` |
| `experiments/configs/dynamics_suite.yaml` | 76 | 1892 | `92b7ff4e40017eb0c1c95e50c9d4e3831ad0bda4` |
| `experiments/configs/experiment_zero.yaml` | 34 | 764 | `fb105fd7e298c2112c59757544d377cbe7dff1cc` |
| `experiments/configs/experiment_zero_generator_shift.yaml` | 64 | 1569 | `50a3dec7e813f7d37f6456e0cbc9ce38e8c8df91` |
| `experiments/configs/experiment_zero_sample_efficiency.yaml` | 33 | 853 | `d2680a3180a8a717eeae1e2ac729e83f97be9870` |
| `experiments/configs/grand_falsification.yaml` | 340 | 10436 | `cdf62bd58c99234f834ca5878ad90722429caa72` |
| `experiments/configs/hard_halting.yaml` | 13 | 504 | `addbf6d23189532bd2670ba772013e9e85b29e1c` |
| `experiments/configs/independent_confirmation.yaml` | 106 | 2520 | `b7928cea783c2b0ad63b392ddaf9b17d9a189e4f` |
| `experiments/configs/independent_discovery.yaml` | 90 | 2178 | `172c2f7cf291eedb53b52e350a136049932ae7d3` |
| `experiments/configs/independent_task_audit.yaml` | 38 | 1081 | `cf3f450c8fda76ebbd9f2bbb25040a8aff945271` |
| `experiments/configs/mechanism_suite.yaml` | 101 | 2876 | `78fe698512a81e63a5dee2624a2507fd5ae0ba44` |
| `experiments/configs/neuro_task_suite.yaml` | 58 | 1304 | `175e5598ee23883a6d0c822164e873ce5a87be42` |
| `experiments/configs/observable_probe.yaml` | 41 | 930 | `ee6ff3fe86f0c676c0d09f769826bec6b736af87` |
| `experiments/configs/qn_grand_001.yaml` | 33 | 1331 | `1e22811c7bf9418405dd318359a7f08155eadb3b` |
| `experiments/configs/robustness_world_sweep.yaml` | 67 | 1705 | `cf997beaf04344eed9c176294d7f8923533763da` |
| `experiments/configs/shift_gauntlet.yaml` | 42 | 1239 | `291e336a99d672fca5e6a76b3663de5fb3bd2d97` |
| `experiments/configs/shift_pilot.yaml` | 91 | 2422 | `e5853af048696b70bfb61731d59abefdf3f9b479` |
| `experiments/configs/simulator_red_team.yaml` | 32 | 963 | `8262754eef0f9886e450436a37ebe86da14f83f8` |
| `experiments/configs/simulator_red_team_v2.yaml` | 38 | 1255 | `56790c62003370655471602fe01963074855f1a3` |
| `experiments/configs/training_laws.yaml` | 63 | 1578 | `c0a3b7fbf2c2d6d0ed82ed1760815c98ad92198e` |
| `experiments/configs/trajectory_study.yaml` | 13 | 496 | `674bddefe1db7076d2ad0b7d5d76b3e56c074d3a` |
| `experiments/results/QN-000001/config.yaml` | 34 | 756 | `ed3977070d910c175fe4825cb039a316aede1bc5` |
| `experiments/results/QN-000001/environment.json` | 9 | 257 | `28dcff06c54623f83d80c9a9e62ddf336f4d3fbc` |
| `experiments/results/QN-000001/metrics.json` | 648 | 17922 | `7c26d75af96ddde55b4987f357c286129b6864d5` |
| `experiments/results/QN-000002/VALIDITY.md` | 9 | 553 | `9e41016b222845dc33e7faf9bcf63930b52772bd` |
| `experiments/results/QN-000002/config.yaml` | 35 | 758 | `4712a4439f82961e8af9b643f523969a76b5d607` |
| `experiments/results/QN-000002/environment.json` | 11 | 280 | `3b2ee54478da23a5d83d6c0a976ecfaa4a29bed1` |
| `experiments/results/QN-000002/metrics.json` | 1856 | 56707 | `07ae9a56ef147a2955bf92f721a153948bd3c7ca` |
| `experiments/results/QN-000003/config.yaml` | 35 | 758 | `4712a4439f82961e8af9b643f523969a76b5d607` |
| `experiments/results/QN-000003/environment.json` | 10 | 279 | `75ef7928590209a9425694bc49f55a504fba3b48` |
| `experiments/results/QN-000003/metrics.json` | 1912 | 58418 | `0b61fa2772036864afffbef61938a41ac6b69b7d` |
| `experiments/results/QN-000004/config.yaml` | 40 | 863 | `281ac0eab1f80ebc258b05af655e52a727cbd1db` |
| `experiments/results/QN-000004/environment.json` | 10 | 279 | `72f12deaeab641bed3ce32c8713cfdd1470afe2f` |
| `experiments/results/QN-000004/metrics.json` | 12172 | 384586 | `022a4043ba87aa3bd5c4759c9d4462727a89ad50` |
| `experiments/results/QN-000005/config.yaml` | 71 | 1552 | `88cd4987557d61e99f914f4f8c60e7fd1f552186` |
| `experiments/results/QN-000005/environment.json` | 10 | 278 | `aa73d42e1802784e23de5b41839a2d3d07321365` |
| `experiments/results/QN-000005/metrics.json` | 2971 | 94707 | `43f45b39c8a5233186b1406728521831e3ec3585` |
| `experiments/results/QN-000006/config.yaml` | 77 | 1605 | `1470b1ab9d99de462c1c53534ca458852c0a9685` |
| `experiments/results/QN-000006/environment.json` | 10 | 279 | `baa09856ab202f4cdf1250fd20fedc68ea3d19dc` |
| `experiments/results/QN-000006/metrics.json` | 33049 | 1164783 | `e5e432bb985e3658e16921e4ed239f33e9e3751f` |
| `experiments/results/QN-000007/config.yaml` | 65 | 1409 | `45ffaf339283b513b2960ad7fdd78e674ead9794` |
| `experiments/results/QN-000007/environment.json` | 10 | 278 | `7bf081dc8879482272d51e5c12dc007edd091d1b` |
| `experiments/results/QN-000007/metrics.json` | 7436 | 239015 | `40b9ad6327fde195227c166b0e95a9d6bf146b6b` |
| `experiments/results/QN-000008/config.yaml` | 83 | 1746 | `6871b28a676731b0fad3f0dbdb8a94a79ea2c6f5` |
| `experiments/results/QN-000008/environment.json` | 10 | 279 | `2cd89022e97b5a18215d49e7bf0a16e985505ee5` |
| `experiments/results/QN-000008/metrics.json` | 36334 | 1315169 | `4e950eb0d64b5dfe37343c2db13bd67e41893452` |
| `experiments/results/QN-000009/config.yaml` | 64 | 1312 | `72cd26e629f1f269e703e13e8678b6dc90eb43fc` |
| `experiments/results/QN-000009/environment.json` | 10 | 278 | `1ead46f639114abd0b8fae574f34e4921cc372f5` |
| `experiments/results/QN-000009/metrics.json` | 4904 | 164486 | `587bf130cca73fab850bbbeaae5cc7c27bace88b` |
| `experiments/results/QN-000010/config.yaml` | 66 | 1320 | `1f033b25318697daab75646fb3f8650bbaf923aa` |
| `experiments/results/QN-000010/environment.json` | 10 | 279 | `2f7ebde0c48787419a2ea33a9a3ee91aa8bafc7b` |
| `experiments/results/QN-000010/metrics.json` | 19484 | 739648 | `42b26924c66edf44d408ee72046c084f0aa2f22e` |
| `experiments/results/QN-000011/config.yaml` | 55 | 1118 | `40239f9ac295ca376b96a1ef42cd1a1641a362a8` |
| `experiments/results/QN-000011/environment.json` | 10 | 278 | `9d7f3e3f61aac261571db634a203129efe1e8f7a` |
| `experiments/results/QN-000011/metrics.json` | 5072 | 145966 | `da781eacb546f1b4f0010c9231da99c3c3d02b32` |
| `experiments/results/QN-000012/config.yaml` | 56 | 1118 | `37e696e7d78bc363eaf4cf4b5416ae5bc78206d3` |
| `experiments/results/QN-000012/environment.json` | 10 | 279 | `9750c70f4cfd0d3657e91d6b8fe7d3b01a208ee7` |
| `experiments/results/QN-000012/metrics.json` | 21332 | 690001 | `76a665d7a6955340becaea383ac478195adb884b` |
| `experiments/results/QN-000013/config.yaml` | 96 | 1915 | `96a8bda5959568c0ba080888848ba22444fe9652` |
| `experiments/results/QN-000013/environment.json` | 10 | 278 | `b3724f917565c19a9e27b5e89017b5ce5d95bf65` |
| `experiments/results/QN-000013/metrics.json` | 10236 | 315155 | `9a48e82dab3d041c10e25f55f038b42cebb278a7` |
| `experiments/results/QN-000014/config.yaml` | 100 | 1940 | `a3e1872b3c214f29764b6a63d9c72b2f0067cc7e` |
| `experiments/results/QN-000014/environment.json` | 10 | 279 | `70c1b782ffaed03121b69419160498c692231546` |
| `experiments/results/QN-000014/metrics.json` | 29284 | 1019988 | `54e51166dcab479a6f938354ae2b6cf70b1710f3` |
| `experiments/results/QN-000015/config.yaml` | 86 | 1767 | `daf48e29c62ae512a162f543f9fbe4926a45c9ee` |
| `experiments/results/QN-000015/environment.json` | 10 | 278 | `91452e6265e9a5f0444accd0926393b4fa3c20dc` |
| `experiments/results/QN-000015/metrics.json` | 8637 | 265999 | `d8ef4ddeff2a14e9a0c9fee203f79f5f33b02e62` |
| `experiments/results/QN-000016/config.yaml` | 90 | 1792 | `a881b5ae3177012a3848aa443b25703fab2a383a` |
| `experiments/results/QN-000016/environment.json` | 10 | 279 | `37c353aa184719f17c02c054ed33101d89a9f74b` |
| `experiments/results/QN-000016/metrics.json` | 24645 | 861916 | `86598ac37e9f5ff55c3cdc430b08961442856160` |
| `experiments/results/QN-000017/VALIDITY.md` | 6 | 355 | `3778eaee4a8715df62fd12d88bba71918a7551aa` |
| `experiments/results/QN-000017/config.yaml` | 23 | 595 | `c50560002464f073654c27bc8128c1d52d4fc301` |
| `experiments/results/QN-000017/environment.json` | 10 | 278 | `f84c6c3e3b4434d2f96389f875c74f9eede17f87` |
| `experiments/results/QN-000017/metrics.json` | 665 | 19372 | `82d63769172a75625df1266a322cb9f954642989` |
| `experiments/results/QN-000018/config.yaml` | 23 | 595 | `c50560002464f073654c27bc8128c1d52d4fc301` |
| `experiments/results/QN-000018/environment.json` | 10 | 279 | `05eaf00a1b5abe728dec4b46e57d4e27f8ebf233` |
| `experiments/results/QN-000018/metrics.json` | 665 | 19539 | `6887726f39594300f18606c08ff8e20a4ab7f0bf` |
| `experiments/results/QN-000019/config.yaml` | 41 | 885 | `1d279a7d729c9c534244bb0f9d02e831ae286ded` |
| `experiments/results/QN-000019/environment.json` | 10 | 279 | `d7437b0a866d56e8840c082b94990f359949e419` |
| `experiments/results/QN-000019/metrics.json` | 4929 | 141736 | `02c74a93bffbb488932dfb4de773629401897ff6` |
| `experiments/results/QN-000020/config.yaml` | 66 | 1563 | `7fd624cae4d942677eb04e8d76eecfa99832b950` |
| `experiments/results/QN-000020/environment.json` | 10 | 278 | `7b029d95cdea49de6e9e62515427a339e08ae1eb` |
| `experiments/results/QN-000020/metrics.json` | 4111 | 131973 | `95a815e05fa9198a133e2c19107b31c2328e83b6` |
| `experiments/results/QN-000021/config.yaml` | 70 | 1593 | `ba57a6b2a6759ffe24e065b598c9bb1215ce8a7d` |
| `experiments/results/QN-000021/environment.json` | 10 | 278 | `7b029d95cdea49de6e9e62515427a339e08ae1eb` |
| `experiments/results/QN-000021/metrics.json` | 19263 | 682964 | `130c18b7460d1de5764c1213ef3600777cf5f408` |
| `experiments/results/QN-000022/config.yaml` | 15 | 461 | `0ce29d4d1c6f66f9ca896f5d9daddc3e2a0efad4` |
| `experiments/results/QN-000022/environment.json` | 10 | 278 | `7da548032e7984ad1acb03994b1a44d987305b7f` |
| `experiments/results/QN-000022/metrics.json` | 3205 | 112767 | `075abd02fb520cb273e1dc9dfc43d0bea1b88b09` |
| `experiments/results/QN-000023/config.yaml` | 25 | 526 | `56b1bf349ef7b02372432e59e0714afc559f9a1d` |
| `experiments/results/QN-000023/environment.json` | 10 | 278 | `7da548032e7984ad1acb03994b1a44d987305b7f` |
| `experiments/results/QN-000023/metrics.json` | 3895 | 138138 | `7951d7f939fec81fcc8cf2e7ccfcca9a48b56f2d` |
| `experiments/results/QN-000024/config.yaml` | 16 | 497 | `fb97a49a35c976a0fc22d36b7a2fd7b71ad9a1c6` |
| `experiments/results/QN-000024/environment.json` | 10 | 278 | `a0b19e7e5815a32d1b610fb0e150205a4ebe3ca4` |
| `experiments/results/QN-000025/config.yaml` | 16 | 497 | `fb97a49a35c976a0fc22d36b7a2fd7b71ad9a1c6` |
| `experiments/results/QN-000025/environment.json` | 10 | 278 | `a0b19e7e5815a32d1b610fb0e150205a4ebe3ca4` |
| `experiments/results/QN-000025/metrics.json` | 146 | 5077 | `c6c4b097554a5f7698a0fff2446969de4ac4ad24` |
| `experiments/results/QN-000025/selected_trajectories.json` | 3747 | 101799 | `2d7b293ec37f261d8f1ba005c85343aa35dee8ed` |
| `experiments/results/QN-000026/candidate_registry.json` | 888 | 26748 | `611ec5586eea2bdeceb672e92909ffa3b1f2291c` |
| `experiments/results/QN-000026/config.yaml` | 55 | 2387 | `eb311c60d02cca139b26b7069cef96830da989f0` |
| `experiments/results/QN-000026/environment.json` | 10 | 278 | `8b3df8035903cbe49f12e26136d72fe497d6466e` |
| `experiments/results/QN-000026/metrics.json` | 37 | 869 | `12c285c22e82c7ccbbfb92c23219ee859eabea28` |
| `experiments/results/QN-000026/pareto_frontiers.json` | 26 | 575 | `c6b6b561c5c1ac512809dfef20063c93844182d8` |
| `experiments/results/QN-000026/proposals.yaml` | 36 | 1749 | `5adfdabef4f21dcfdf5c6ccae1d71e3d58f4f8c8` |
| `experiments/results/QN-000026/surprises.json` | 158 | 5462 | `be873e45e4adc0b7e1ffa51c5bf2729d6bd137af` |
| `experiments/results/QN-000027/config.yaml` | 33 | 959 | `d4d46146426f0ae856fec83241099d54218c3592` |
| `experiments/results/QN-000027/environment.json` | 48 | 1184 | `7cd20c99f74f0fc14e1af1c16268811541d732a5` |
| `experiments/results/QN-000027/metrics.json` | 67 | 1818 | `797e940375b569224df7ee7d043ce4e0fc60d124` |
| `experiments/results/QN-000028/config.yaml` | 39 | 1251 | `2bb1ab0a2847ef84563057c32e7b2a25fc50e963` |
| `experiments/results/QN-000028/environment.json` | 46 | 1145 | `3e7a42f4fd18f44feccce5c4cdecf0b609347dbe` |
| `experiments/results/QN-000028/metrics.json` | 70 | 1931 | `f47315f99a407515adbbb9a7592447e97f37d0a2` |
| `experiments/results/QN-000029/config.yaml` | 58 | 1294 | `0ea18f249f97063eb86b0986fdd699237772c212` |
| `experiments/results/QN-000029/environment.json` | 46 | 1141 | `765719a07c1138210ce9755a11841103945efb54` |
| `experiments/results/QN-000029/metrics.json` | 7886 | 189069 | `37d50ef04270876afebaeb7e4608384dbb2d178d` |
| `experiments/results/QN-000030/config.yaml` | 108 | 2357 | `5f527ec4b715711114c81d2ce2aeb0b33946f3e7` |
| `experiments/results/QN-000030/environment.json` | 46 | 1138 | `8a5fc63304add9e343be1b76b87cb114a7e53730` |
| `experiments/results/QN-000030/metrics.json` | 8903 | 245705 | `b182ffeca99d53ee6ad591cd91ccc9cb363fface` |
| `experiments/results/QN-000031/config.yaml` | 123 | 2513 | `f760870c5ea70ae3a498394199d0975c96c8f42a` |
| `experiments/results/QN-000031/environment.json` | 46 | 1138 | `1a0229bbf0c6a87f27cc27cd57d1a022aafd453b` |
| `experiments/results/QN-000031/metrics.json` | 327112 | 9118678 | `faa3fcf54862508340f4f2f72f44a7e618f00ff4` |
| `experiments/results/QN-000032/config.yaml` | 82 | 1979 | `82c0730e8bc3bfdd77559b84b0a96a69a626f4b4` |
| `experiments/results/QN-000032/environment.json` | 46 | 1142 | `ff47206453de714f8d2d621048026f1c1b8422af` |
| `experiments/results/QN-000032/metrics.json` | 2649 | 68548 | `d260aebc721a54947945d8548e369e611f29b9fb` |
| `experiments/results/QN-000033/config.yaml` | 129 | 2929 | `cdf8226b5c2779d0f777595ba0885b952ab59c67` |
| `experiments/results/QN-000033/environment.json` | 46 | 1142 | `9030eed67702436abdfd9dda965907f38e3ad0bb` |
| `experiments/results/QN-000033/metrics.json` | 52426 | 1474944 | `55a671633f72c047b43f73857afb0a2e60a88b0c` |
| `experiments/results/QN-000034/config.yaml` | 53 | 888 | `951a26d29e78b163b75ca3df81f0f17fd5b8879f` |
| `experiments/results/QN-000034/environment.json` | 46 | 1150 | `8c8e1de3b96998b200db4d7b9bcf7184d9103752` |
| `experiments/results/QN-000034/metrics.json` | 163 | 4693 | `7944fdb288029cf15303e3be89e3c434943e71c9` |
| `experiments/results/QN-000034/pipeline_frozen_law.json` | 13 | 288 | `16bcff4cf4e2f9de6d785a666e09034b6d9c66b5` |
| `experiments/results/QN-000035/config.yaml` | 35 | 1033 | `4cfeba72dd745f1d53abdeb6c6627da48bbdfeb3` |
| `experiments/results/QN-000035/environment.json` | 46 | 1149 | `cb4b1d6180a2794bb4244a2dac52c4f8e625c4b9` |
| `experiments/results/QN-000035/metrics.json` | 483 | 15640 | `1086f76f6978285d7acf968535a5a508c5f61543` |
| `experiments/results/QN-000036/config.yaml` | 36 | 1058 | `36041bde5d421e102cb6ab0a45f8f5d821292655` |
| `experiments/results/QN-000036/environment.json` | 46 | 1149 | `de65bc7589b922658e79474bbdb12ff2a8214612` |
| `experiments/results/QN-000036/metrics.json` | 516 | 17623 | `b882e04519cac89d98b39b684ad01832d2ba3b71` |
| `experiments/results/QN-000037/config.yaml` | 37 | 1076 | `867343e263d2be02662230d561833acb29f67505` |
| `experiments/results/QN-000037/environment.json` | 46 | 1149 | `8d9499eed68fb3d486a7120c143bf754c15dca75` |
| `experiments/results/QN-000037/metrics.json` | 532 | 18095 | `446e7d0b89ab19de932b7f531411da9ad773d5af` |
| `experiments/results/QN-000038/config.yaml` | 63 | 1599 | `7fb561dae47b4a7dccfba2471b4b0e6c4bd477ef` |
| `experiments/results/QN-000038/environment.json` | 46 | 1148 | `4ebb30e57fef92a0a42d7e754d712a4a9d9323ad` |
| `experiments/results/QN-000038/failure.json` | 7 | 281 | `fb636455db4d20dc05cac01815cf3ad678f6f38f` |
| `experiments/results/QN-000039/config.yaml` | 69 | 1748 | `131680853c8a6455930e0ae8afaa50c86bd48dfb` |
| `experiments/results/QN-000039/environment.json` | 46 | 1148 | `4eb1733f5329e056d25f44a2097aff0ed822be4c` |
| `experiments/results/QN-000039/metrics.json` | 2343 | 70098 | `2f8562bed385f61c56be2400e619dac9a813bdf1` |
| `experiments/results/QN-000040/config.yaml` | 105 | 2152 | `15fac29e20b651d8b1d0fffbd132b7686aa0deb2` |
| `experiments/results/QN-000040/environment.json` | 46 | 1148 | `4fdbdb95140580d2f5d4ef45b6ff02c3aa3d2264` |
| `experiments/results/QN-000040/metrics.json` | 358450 | 10499524 | `ebd4328e5e9aaf04435b061864c90d4b30fdae74` |
| `experiments/results/QN-000041/config.yaml` | 75 | 2003 | `9f768af348cf7e5a9fdf8cd649e3a70248c22fbe` |
| `experiments/results/QN-000041/environment.json` | 46 | 1151 | `f361046fe786d7fc875b457714715d09d09d775a` |
| `experiments/results/QN-000041/metrics.json` | 1578 | 48592 | `6019dc5baac376a4589136c03ffa61a458762159` |
| `experiments/results/QN-000042/config.yaml` | 119 | 2469 | `428424b3553abee8301ad45b9212398b0914862f` |
| `experiments/results/QN-000042/environment.json` | 46 | 1151 | `f52b2b6395b5d147c90feff949c8401e3b844116` |
| `experiments/results/QN-000042/metrics.json` | 234351 | 6864171 | `9b8eeb6361a5d7c382b92ae44e1b141dc3b110f2` |
| `experiments/results/QN-GRAND-001/config.yaml` | 33 | 1321 | `34ec01392185052575b5ff69bd76b0d7ad6c0bad` |
| `experiments/results/QN-GRAND-001/decision.json` | 17 | 731 | `4d17a6e0080d6db0acf64c5cd1806bf1f53c3225` |
| `experiments/results/QN-GRAND-001/environment.json` | 48 | 1181 | `876f9a9e4d27e3f3789f77f643e89b4be8c4babc` |
| `experiments/results/QN-GRAND-001/preflight.json` | 108 | 5229 | `71e400fab4529e6a631b1cffc2a64e4f6a98bf4b` |
| `experiments/run_active_evidence.py` | 248 | 10188 | `f35776ccd05675c606ed7dd0f1c613193dab7b98` |
| `experiments/run_computational_law_suite.py` | 183 | 7593 | `74e532de6cc765b2415c9d92fe840f32ed6ba2a6` |
| `experiments/run_discovery_engine.py` | 320 | 12884 | `68a86e427f599eebda22aa025e5edc1bfd886dbb` |
| `experiments/run_dynamics_suite.py` | 299 | 13611 | `8637b96ff1a68fc81f2764c4e4c4d35faf688cd7` |
| `experiments/run_experiment_zero.py` | 468 | 16887 | `a502f2f067f1c9b6859b8150582d4f176be2618c` |
| `experiments/run_generator_shift.py` | 286 | 11773 | `e2ce361055a38876c0ef1ab28054b3deced8d37d` |
| `experiments/run_hard_halting.py` | 466 | 17821 | `c53808eb8bea0d916444500771517f5a4dbdd43e` |
| `experiments/run_independent_confirmation.py` | 85 | 2735 | `51887ad601cb81c128fbfad2df489d86eea6718b` |
| `experiments/run_independent_discovery.py` | 459 | 19734 | `00f749238ecdf07cec17af128807532345584e2f` |
| `experiments/run_independent_task_audit.py` | 199 | 8329 | `ff9e53902483134e817231bb64bd089c0bd877e1` |
| `experiments/run_mechanism_suite.py` | 344 | 14137 | `803f75052c1a240bfe353723e13021bd77d527c8` |
| `experiments/run_neuro_task_suite.py` | 403 | 16867 | `1af37afdc4fd3fa44d2c811423a38b1316e77294` |
| `experiments/run_observable_probe.py` | 244 | 11341 | `f27905292f1670e799bf5e6154795627394394b1` |
| `experiments/run_qn_grand_001.py` | 275 | 10903 | `0205777cef0391cabfc09dcf295bcd8aa6b6fab0` |
| `experiments/run_robustness_sweep.py` | 346 | 14723 | `2998e08c7976a93ff43020a5f0db01a72572e2ce` |
| `experiments/run_sample_efficiency.py` | 178 | 7136 | `155859059c73a4fa15bef4636add6e50d835bd06` |
| `experiments/run_shift_gauntlet.py` | 178 | 7161 | `d4310cd31d0a08751502275fb92c7b311523aaa3` |
| `experiments/run_shift_pilot.py` | 404 | 17469 | `f812dd59f874fbc9c1921848f0ea2c674ed79651` |
| `experiments/run_simulator_red_team.py` | 161 | 6290 | `2bb1d91ddabe73b1a1ac026ef4533611cba1d856` |
| `experiments/run_training_laws.py` | 559 | 23563 | `3c9620c0502e73c57b604e1c7ebd76f4ddb3b775` |
| `experiments/run_trajectory_study.py` | 308 | 13107 | `ec90e7f832a22b34addabe154aa92143cac3612d` |
| `independent_tasks/__init__.py` | 17 | 389 | `3bd1d61e05d289348e604004a2691be4ab09f2de` |
| `independent_tasks/generators.py` | 373 | 13615 | `4cb624e21b1d5fbf916b11ac5b2a5a569b93ccc3` |
| `neuroworld/__init__.py` | 29 | 786 | `c9d3eca0b9ece3e8fc952e331491270fb2cf2fd7` |
| `neuroworld/shifts.py` | 517 | 20667 | `f41e333959451e91d26fb6550216a394b7562720` |
| `neuroworld/simulator.py` | 273 | 11696 | `d5db142ac82cf63442e081cbc4050665eae63282` |
| `neuroworld/tasks.py` | 174 | 6037 | `e2a20e82162aa41364e0ff033142bfdb2864e93c` |
| `neuroworld/validity.py` | 215 | 8175 | `56d1f17a938f987faac27a6585e7283db471585b` |
| `paper/MANUSCRIPT_METADATA.json` | 30 | 1981 | `71a4ebd21886cbbfc0c34b6526a1ef13b2f6607e` |
| `paper/README.md` | 46 | 2258 | `c948343dcac9ef20407821fd5b9534b5dda04867` |
| `paper/ablations.tex` | 18 | 2765 | `5ea88d9d7f61b96f6081e8159003085d605916fc` |
| `paper/abstract.tex` | 5 | 2040 | `631af4649f9b2042112b1b151f60ce46d5422f55` |
| `paper/appendices.tex` | 57 | 5660 | `483971def5d0bb0d07b0702749308cb3b13beb45` |
| `paper/architecture.tex` | 17 | 3044 | `dd22ae2831462b1f16beaabe12f6a0f66a82e0e1` |
| `paper/build_manuscript.py` | 998 | 38154 | `946b987745ce2669f20934b1d626cc9ea37a327f` |
| `paper/build_tables.py` | 433 | 14256 | `cda4d9a33198a2050133ca9a71984493bac70ffb` |
| `paper/conclusion.tex` | 5 | 1065 | `ac13d24dc49becd538e73e4a3ba4b418757889aa` |
| `paper/discussion.tex` | 13 | 3075 | `6192b1f6ca17f96cc5d6762a9ee8f6ec221d34f4` |
| `paper/experiments.tex` | 28 | 3639 | `b8727fdd17df8238113db638d42131d311b7d3e0` |
| `paper/figures/README.md` | 7 | 351 | `9311b441150a7a978935f8f2ade9184014e25a7a` |
| `paper/figures/active_evidence.pdf` | 1098 | 47814 | `80a0f39c38b944f0a6bf2e38fc689a5fa45f3c00` |
| `paper/figures/active_evidence.png` | 1532 | 395453 | `6483e01264be4ebcfc19fbd90e0e86de616f8f71` |
| `paper/figures/ambiguity_differential.pdf` | 723 | 26238 | `3d359e5df4f92d48ba4230a7a418eca4c710cc7f` |
| `paper/figures/ambiguity_differential.png` | 192 | 85137 | `a99fdb01c0359ee7124368648427b67357ab088d` |
| `paper/figures/architecture_overview.pdf` | 769 | 26751 | `95a1af5d4aa7d50939beca3f7dc1f0b5ccb0916e` |
| `paper/figures/architecture_overview.png` | 353 | 87399 | `f10660af52f38b27552d5df41200f59700312215` |
| `paper/figures/architecture_pareto_field.pdf` | 635 | 27340 | `632e94948ae4b19708e72341a47c1094d8d340f1` |
| `paper/figures/architecture_pareto_field.png` | 165 | 88362 | `967df20efdecebfa2a01fe16331860d4571595b9` |
| `paper/figures/calibration_transport.pdf` | 586 | 20749 | `f7e113bad83ef8aa436175e8aafbb3935ed1ed00` |
| `paper/figures/calibration_transport.png` | 128 | 51401 | `a603d5e81ad3fecf1c9088010b2ea036fb83e7d4` |
| `paper/figures/claim_status_audit.pdf` | 533 | 19317 | `3c6fb710bb2c9f813316f9c05255d8528eb5d695` |
| `paper/figures/claim_status_audit.png` | 89 | 49094 | `b1a7c34c73d5eba18eabaf19ebe904eac766cff8` |
| `paper/figures/critical_ablation_suite.pdf` | 923 | 37376 | `11de2a96352210f91342fbff5f20830c90b96166` |
| `paper/figures/critical_ablation_suite.png` | 611 | 231663 | `321e0d24888e3bba65e8e7fb3e2d8dc04e471da7` |
| `paper/figures/dynamics_suite.pdf` | 985 | 42838 | `bbcc44238746eac7f200f4e2af064e48db952c96` |
| `paper/figures/dynamics_suite.png` | 670 | 253915 | `92b2236e77c696033ed6691acff630cee93065ee` |
| `paper/figures/experiment_evidence_map.pdf` | 630 | 22553 | `ed83edc8959dfc27b323f32ccd48beef95e1c15f` |
| `paper/figures/experiment_evidence_map.png` | 286 | 81839 | `fa11e406697da0ef2e94cc5cb5b1c94fd0af8035` |
| `paper/figures/experiment_zero_learning_curves.pdf` | 1045 | 42580 | `71eeb06f04d6c36c5c81786a26adb869f7c30bf2` |
| `paper/figures/experiment_zero_learning_curves.png` | 991 | 335784 | `7e833a1d56ac26848c9942fdded1588ba1ef7694` |
| `paper/figures/falsification_phase.pdf` | 1043 | 39485 | `6a3a1732c4bf68c717f245a1db18836bbcacf7b7` |
| `paper/figures/falsification_phase.png` | 896 | 261328 | `a3f605e95c3fedc908469f750c38adfcdfab74eb` |
| `paper/figures/generator_shift_replication.pdf` | 1086 | 42846 | `3159dc3727febb350d53f3c001a2a180db7eca93` |
| `paper/figures/generator_shift_replication.png` | 1642 | 358217 | `42e919207ab8eccb1110e316bdaf9a5c7249d214` |
| `paper/figures/hard_halting.pdf` | 1022 | 39663 | `59228f4d7c9d110eb27f175a16790e8dfef3bb80` |
| `paper/figures/hard_halting.png` | 812 | 251541 | `088a2eb3b2612ac29a654e9e6cc524e7dc9f8aa3` |
| `paper/figures/manifest.json` | 11 | 296 | `d2e49019aa2991bd0d9ed3a8de72e3477d6bf5c0` |
| `paper/figures/neuro_task_suite.pdf` | 905 | 37091 | `09a1ae440c9c2fc694725b1ff3b8cf29c8b4de63` |
| `paper/figures/neuro_task_suite.png` | 660 | 261126 | `cb2c055686060568ab52187f72f7150135ce6678` |
| `paper/figures/observable_probe.pdf` | 1086 | 50788 | `e5c44be37200b3243dc60dfdd6d530fd60428934` |
| `paper/figures/observable_probe.png` | 1055 | 304485 | `21107855a1d905139c983aad84acb332e35181eb` |
| `paper/figures/ood_separability.pdf` | 657 | 24179 | `e50730ff95a1f0eedc93dee6f3319bb76c121f43` |
| `paper/figures/ood_separability.png` | 137 | 73319 | `98bcc6db91326108b52b6277d434aee4359be0e5` |
| `paper/figures/robustness_world_sweep.pdf` | 1078 | 43010 | `9029fb16f0efeccde7819b4759c2aa8b1db208c9` |
| `paper/figures/robustness_world_sweep.png` | 1154 | 326501 | `fae8e3783e098945a34a266b4c78331dd48d4673` |
| `paper/figures/sample_compute_frontier.pdf` | 667 | 27709 | `fe67b1fa0a798ab2127a96b85617f56b8b4915f7` |
| `paper/figures/sample_compute_frontier.png` | 149 | 71429 | `e6c8b6afb09ffe3899d70578c3247330f35a725d` |
| `paper/figures/surprise_taxonomy.pdf` | 532 | 19482 | `4bba2b96c66e501a8eeb2e94f028ae7d88e6e378` |
| `paper/figures/surprise_taxonomy.png` | 265 | 55364 | `6532cf2878151362863d66338e07efc464375490` |
| `paper/figures/training_law_suite.pdf` | 1057 | 43113 | `6a333138f12a2bb8fabe7228d5fd3457e87861a4` |
| `paper/figures/training_law_suite.png` | 913 | 256888 | `a5d60f8a2dc118517622dc7f01c3dc654cf43998` |
| `paper/figures/trajectory_signature.pdf` | 1134 | 58484 | `2b5258576b453e36c0ec20c72f239c4eb8fdbbba` |
| `paper/figures/trajectory_signature.png` | 1465 | 440215 | `55bf8528a98dff95f93fe06052170412252931a5` |
| `paper/interpretability.tex` | 13 | 2293 | `bfc4cd8573b679a9fdf68e3c53b3158e32c10c33` |
| `paper/introduction.tex` | 11 | 3777 | `2762ec5f9b6c55fa66c212e63daafd7955ad059a` |
| `paper/limitations.tex` | 25 | 3749 | `ec4d14c2a2e3df690a6c1f5c40c205454b9f7179` |
| `paper/main.tex` | 49 | 1388 | `b10e478466a01bcd1c8d7bde7f07d13ae84491fd` |
| `paper/neuroworld.tex` | 33 | 3463 | `c37e56a3d123813bc1e869767e5d8840822a9400` |
| `paper/qneuro.docx` | 1057 | 290291 | `7a016badb1c39e0ad396f0ac2c886b172acc7dd6` |
| `paper/qneuro.pdf` | 8276 | 520972 | `74bbecad1d355853755d8e6b468b87e955b4118a` |
| `paper/references.bib` | 27 | 3368 | `dbd37793394c471ac3d3e9b2872f46e491252f2f` |
| `paper/references.json` | 72 | 6258 | `8bbd57edc6e15759e1011a0d9cd747f5d9fc4dec` |
| `paper/related_work.tex` | 15 | 3893 | `66ae69dc7a6ca3e116b2074fe6bfe6623515d194` |
| `paper/results.tex` | 33 | 5257 | `58c099b74c44d9036b183e229fb21393d2673a83` |
| `paper/safety.tex` | 15 | 2233 | `a7be4ce8ef040d20bd4455a3a3456fa9a53583c2` |
| `paper/source/ablations.md` | 23 | 2661 | `bdfc0ed845b7da788bbae79d532710aaf28944b2` |
| `paper/source/abstract.md` | 3 | 1949 | `1d3558fcb631733cff0b060d59e152f574186763` |
| `paper/source/appendices.md` | 59 | 5533 | `f493dd1897e74cb9d02c712766c952c074dbc1ea` |
| `paper/source/architecture.md` | 21 | 2893 | `423203a7faf037e48a97b9467825db3091eb29ac` |
| `paper/source/conclusion.md` | 5 | 1006 | `3b859ba524b97053851ef9f342469a86d4f1de12` |
| `paper/source/discussion.md` | 13 | 3013 | `0a104df7ab4a1b970ce67babb0c86a4cfbd60685` |
| `paper/source/experiments.md` | 31 | 3445 | `fb62d469e3729b4147dc59475f9100bcf7b28190` |
| `paper/source/interpretability.md` | 17 | 2198 | `fdd0e845c2eb1ed0d3a63e6cacd76b2266c62385` |
| `paper/source/introduction.md` | 11 | 3713 | `5fe4a3e2c681a0ec8110fc04244e40218eaa44fc` |
| `paper/source/limitations.md` | 33 | 3618 | `8714e50024160889c9f3ec659c85a533daf30418` |
| `paper/source/neuroworld.md` | 33 | 3236 | `a71e112e6874d9743ad989ba5da07096352acb4b` |
| `paper/source/related_work.md` | 19 | 3751 | `99f2302264ae20f16afef7322240d12ea7572e52` |
| `paper/source/results.md` | 33 | 5096 | `6034dabafccef67711eef56f28e079b07b3e3805` |
| `paper/source/safety.md` | 19 | 2138 | `1fbb6095d90a6153d5b8c40957d28033e8206a0c` |
| `paper/source/theory.md` | 43 | 4011 | `dbe86e37ec4149062eae0a7904d2da6f3161e75c` |
| `paper/source/training.md` | 23 | 2819 | `56201a5531c86b86be803409410eea93b64783f1` |
| `paper/tables/architecture_results.json` | 111 | 1724 | `d483876d75308e9dc57a62f03bf29f59727dabd3` |
| `paper/tables/architecture_results.tex` | 24 | 1169 | `6b7b6040039cd1eb60d5d1093767195688998b64` |
| `paper/tables/critical_ablations.json` | 70 | 1206 | `8025283b87e926b187a2b26a1889c9eb33889cdc` |
| `paper/tables/critical_ablations.tex` | 20 | 923 | `98c1a7e41a78c6d67654b63174d7c88434b0d2a6` |
| `paper/tables/full_data.json` | 52 | 846 | `958c4a87bf59133f9a709e5a8f569765cce7a696` |
| `paper/tables/full_data.tex` | 16 | 677 | `7ad2ead45832918384a4807b4a4f720d3a0c962f` |
| `paper/tables/hard_halting.json` | 43 | 774 | `a357be1d44270ac53a402d33ce6d20272b04033a` |
| `paper/tables/hard_halting.tex` | 15 | 665 | `38dc6ec4155ca42abb1bd1968db0c9ee4d2e7c69` |
| `paper/tables/sample_efficiency.json` | 49 | 756 | `0c99cfc0e77e10992d0787757e137eaed66b9435` |
| `paper/tables/sample_efficiency.tex` | 17 | 605 | `7eb514610bdf9ef001392251ab8609976bbc0567` |
| `paper/tables/training_laws.json` | 87 | 1358 | `df019b6e8a75c8ab4ca7d31a2a2d04a8c5d7f848` |
| `paper/tables/training_laws.tex` | 21 | 962 | `7b760f5b27bd96e27dcff47a7b546f9319472651` |
| `paper/tables/trajectory_metrics.json` | 57 | 1110 | `71733f8292c885663598691786f46b6ee24e0f59` |
| `paper/tables/trajectory_metrics.tex` | 21 | 902 | `a9bbde69135dfc671d9c3ece44a9c75497d644b4` |
| `paper/tables/uncertainty_ood.json` | 56 | 943 | `29928c2fa248d4d2e8a31b60b7497b9232371abd` |
| `paper/tables/uncertainty_ood.tex` | 18 | 752 | `7168e8066ddc9ab21ad311bcc7ce4f393a0e4c1c` |
| `paper/tables/world_robustness.json` | 63 | 1019 | `5fc86a0932e1a8d25e8c7a951a34e3b1bac54d7e` |
| `paper/tables/world_robustness.tex` | 18 | 776 | `8a54e550c83ef5e0ddaac0d359353bab51afdde9` |
| `paper/theory.tex` | 48 | 4065 | `1336d21b7a9d65315e876859c4a653067289e6d3` |
| `paper/training.tex` | 18 | 2923 | `5fb3252fae3dffbf82793bb30ad0ff6998c31ecf` |
| `pyproject.toml` | 40 | 775 | `5362d610fc16f98549f5b3b14b9fd47c9d46b7f4` |
| `qneuro/__init__.py` | 3 | 74 | `960daae737a8a78c7b8f7166053743b5a6e79951` |
| `qneuro/calibration.py` | 38 | 1197 | `f66db3f92923780a4442363643742a820a40f0ed` |
| `qneuro/data.py` | 60 | 2274 | `7d351ae6ed279ebfd300a82d7800fed45b7cabcd` |
| `qneuro/discovery/__init__.py` | 5 | 198 | `84c3d2fa0219f743357a7ab46e2b9f0714dab5a9` |
| `qneuro/discovery/engine.py` | 80 | 2928 | `7d4f29c49545e7080d15373d48ac4ce4969decd0` |
| `qneuro/evaluation/__init__.py` | 34 | 925 | `847e7b54636a9c25a4c0bcd8ddfdc9f937c7abf6` |
| `qneuro/evaluation/active.py` | 230 | 9256 | `b55538c92141a66293768434ba3e3f4fe333fc1e` |
| `qneuro/evaluation/ambiguity.py` | 33 | 1487 | `80caca67d8ba5cf6249d6b172daa37c3cf876eef` |
| `qneuro/evaluation/ood.py` | 44 | 2180 | `c2f1e4ecc3954d05946f00590a6f9afe97d64ca1` |
| `qneuro/evaluation/representation.py` | 64 | 2471 | `f732ec2f2ec51d53d2d1dbbd239b43e4104e612d` |
| `qneuro/learning/__init__.py` | 19 | 427 | `e9af0d4340907b75e348cf115c70b902d92f691f` |
| `qneuro/learning/rules.py` | 331 | 14866 | `54a9aaff4eb889e3133d591090985d381e7819f0` |
| `qneuro/metrics.py` | 115 | 3772 | `03f4d77834567e787619d8c2c9edd061b2a484ed` |
| `qneuro/model_factory.py` | 476 | 15960 | `c0a74b0cf82b3d64b05898790361b92a944547e3` |
| `qneuro/models/__init__.py` | 93 | 2566 | `c5e92709123c99b7a9c1b7b02c8008396deb8592` |
| `qneuro/models/advanced.py` | 738 | 32365 | `c713197a5fdc05d821dd186280712fbc2d53985a` |
| `qneuro/models/baselines.py` | 388 | 14315 | `fab03d324a36a23c124904e04c4a064d0a78dc81` |
| `qneuro/models/equivalent.py` | 338 | 14731 | `b6eacdd3fa02d252b435a593a90ca9a6ef347ed7` |
| `qneuro/models/mechanisms.py` | 290 | 12875 | `fc41579b42902d45e39d985948361af01a1f2dea` |
| `qneuro/models/operators.py` | 347 | 14399 | `d565b1dad8aaf3433929c394e6ae9a6e8aaadb70` |
| `qneuro/observables/__init__.py` | 15 | 327 | `9a6d967f46fef9f1baa441b9b8abab805c6f09c8` |
| `qneuro/observables/probes.py` | 133 | 5437 | `4b904768db11e22b5754ae0ced9593b9fc2d65f7` |
| `qneuro/provenance.py` | 115 | 3654 | `3163206a82fb4a368f32cde8423b2a89c5c842c2` |
| `qneuro/registry.py` | 454 | 16574 | `d6a97956a40c56c5840ca6f786dd5b7bdb4dbe9d` |
| `release/manifest.json` | 275 | 8913 | `f471d65c662d864e37b8841e35e806fb4dca0c3b` |
| `release/replication_report.json` | 62 | 2155 | `8eedd8c2d216939b13a9e0d244ec5d7997c677cf` |
| `release/verification_report.json` | 194 | 5203 | `c7e7c7105ed40e72f373624e4a141ce78e194f89` |
| `research/__init__.py` | 1 | 56 | `d5009329d66337153d88598db6e004c7269c925d` |
| `research/adversarial_reviewer.py` | 90 | 3223 | `01593489624882f151669dcece9264a071386e8b` |
| `research/analyses/__init__.py` | 1 | 45 | `f9d4a6c0963954dd02c0c5236fdef4b5c826d848` |
| `research/analyses/analyze_ablation_suite.py` | 133 | 4878 | `a1b28301c737ca80d1430f64102cadc61073626e` |
| `research/analyses/analyze_active_evidence.py` | 137 | 4652 | `9d667ffbad0036e5ea4018b2f3a3b47488a0ecc1` |
| `research/analyses/analyze_dynamics_suite.py` | 129 | 4738 | `1527edfd8a3861ba5b9bbdb65ab12b44bc20e06c` |
| `research/analyses/analyze_falsification_phase.py` | 176 | 7799 | `3961d0460db88b39f18815e95e49eb0a8734f8ff` |
| `research/analyses/analyze_generator_shift.py` | 92 | 3479 | `ea36daa8c1f664de1dcb447f783fc1ad16c95b35` |
| `research/analyses/analyze_hard_halting.py` | 116 | 4487 | `88b4013ea417469e54934080b5fcaa47e99dc451` |
| `research/analyses/analyze_neuro_task_suite.py` | 150 | 5114 | `52c22d347f59c498203381836924f63e7b38ee62` |
| `research/analyses/analyze_observable_probe.py` | 136 | 5028 | `cec7205a8f06ee33ad6374d9356f4ca5c0416f7e` |
| `research/analyses/analyze_training_laws.py` | 190 | 7218 | `411cd3595b7f6f57718ea1de9451778809ac1d88` |
| `research/analyses/analyze_trajectories.py` | 83 | 3238 | `e2a78d33d400a64256abe36367d7dd15b867e145` |
| `research/analyses/generated/active_evidence_paired_effects.json` | 2355 | 72700 | `9eb3178bb499ce9d6ecd87b99f1bae68ef97acb5` |
| `research/analyses/generated/critical_ablation_paired_effects.json` | 1323 | 40676 | `34273cc78a3a9624a46d63230a341f3362621379` |
| `research/analyses/generated/dynamics_suite_paired_effects.json` | 1227 | 37723 | `9c6baf93781612f71247ea8c189a443e786591fc` |
| `research/analyses/generated/falsification_phase.json` | 131 | 4702 | `674d671f5623fe9f9a5ca7bf598af479c0ea392a` |
| `research/analyses/generated/generator_shift_paired_effects.json` | 1258 | 40479 | `1cc4ffe296fce0a653099e91929c1c8dca5d5a5b` |
| `research/analyses/generated/hard_halting_effects.json` | 233 | 6857 | `b469ae5478354af83854c96bd93168abd5822a2b` |
| `research/analyses/generated/neuro_task_suite_paired_effects.json` | 1028 | 32744 | `6f02099fbc3e944d7042ea68aaf04e40d48a9f57` |
| `research/analyses/generated/observable_probe_effects.json` | 609 | 17887 | `f408ac73da749b1f3c8c075e4e8c616ae7c67f55` |
| `research/analyses/generated/training_law_effects.json` | 819 | 25366 | `2a7d9508ca8809779c74c131293b8b1a6b7ecbac` |
| `research/analyses/generated/trajectory_effects.json` | 114 | 3508 | `ade0bcadede874fff9d5289af3c2ea21ae7359c5` |
| `research/claims.json` | 126 | 8602 | `18ec3f8296df046203de51d7f2d1785120206375` |
| `research/computational_laws.py` | 348 | 14202 | `a89192125ce4e665d105b74e0df192333e369179` |
| `research/discovery/architecture_catalog.yaml` | 44 | 3427 | `e3bc7f970c072928b9fae0b53fed5a82caacf0c0` |
| `research/discovery/generated/candidate_registry.json` | 888 | 26748 | `611ec5586eea2bdeceb672e92909ffa3b1f2291c` |
| `research/discovery/generated/pareto_frontiers.json` | 26 | 575 | `c6b6b561c5c1ac512809dfef20063c93844182d8` |
| `research/discovery/generated/proposals.yaml` | 36 | 1749 | `5adfdabef4f21dcfdf5c6ccae1d71e3d58f4f8c8` |
| `research/discovery/generated/surprises.json` | 158 | 5462 | `be873e45e4adc0b7e1ffa51c5bf2729d6bd137af` |
| `research/failures.json` | 45 | 2104 | `0eea7a0cbbbb9b1a8356f2ee66811fcd53e23852` |
| `research/figures/__init__.py` | 1 | 37 | `5247299034c3f3e124951c6b6aad497a024ddd03` |
| `research/figures/generate_ablation_suite.py` | 204 | 7264 | `1d5477da9fd0a4b7dc25ce4d3c36cf9d26cce615` |
| `research/figures/generate_active_evidence.py` | 214 | 7852 | `fdefe456e5639e8f3f351554bcc4439fb3f75ed3` |
| `research/figures/generate_dynamics_suite.py` | 183 | 7449 | `2edfa393446c384f8f8a2e11251f1f710ca4997a` |
| `research/figures/generate_experiment_zero.py` | 161 | 5650 | `414b275173ae8cd04778d2ea6890102ae443a8df` |
| `research/figures/generate_falsification_phase.py` | 153 | 5324 | `eb6730f1fc6ee0975693a350f61828b93741058b` |
| `research/figures/generate_generator_shift.py` | 144 | 5042 | `d9785925765aa70f12dc13bbe1259fc890f7c1cd` |
| `research/figures/generate_hard_halting.py` | 145 | 6001 | `31fdcc00560ff79347d6c6419463e76c77c0d841` |
| `research/figures/generate_neuro_task_suite.py` | 194 | 6540 | `fcf592e5258d947a65b366da73ff4b5c06f2af3a` |
| `research/figures/generate_observable_probe.py` | 211 | 7747 | `f088fece862c3cfa1b263d13697ea3ec0511433c` |
| `research/figures/generate_paper_extended.py` | 402 | 13878 | `db862aeda8c14e637bb1955a72d68d6a892b1713` |
| `research/figures/generate_robustness_sweep.py` | 206 | 7431 | `e68daa875f79466f50664d9f4cc6b9bd7daabcaf` |
| `research/figures/generate_training_laws.py` | 203 | 7378 | `51b4d185406f615cdcb8d325859268a4bed42435` |
| `research/figures/generate_trajectory_signature.py` | 187 | 7420 | `b1dd2678b9de515369d859736df53d54b3a5a1a1` |
| `research/figures/generated/active_evidence.pdf` | 1098 | 47814 | `80a0f39c38b944f0a6bf2e38fc689a5fa45f3c00` |
| `research/figures/generated/active_evidence.png` | 1532 | 395453 | `6483e01264be4ebcfc19fbd90e0e86de616f8f71` |
| `research/figures/generated/ambiguity_differential.pdf` | 723 | 26238 | `3d359e5df4f92d48ba4230a7a418eca4c710cc7f` |
| `research/figures/generated/ambiguity_differential.png` | 192 | 85137 | `a99fdb01c0359ee7124368648427b67357ab088d` |
| `research/figures/generated/architecture_overview.pdf` | 769 | 26751 | `95a1af5d4aa7d50939beca3f7dc1f0b5ccb0916e` |
| `research/figures/generated/architecture_overview.png` | 353 | 87399 | `f10660af52f38b27552d5df41200f59700312215` |
| `research/figures/generated/architecture_pareto_field.pdf` | 635 | 27340 | `632e94948ae4b19708e72341a47c1094d8d340f1` |
| `research/figures/generated/architecture_pareto_field.png` | 165 | 88362 | `967df20efdecebfa2a01fe16331860d4571595b9` |
| `research/figures/generated/calibration_transport.pdf` | 586 | 20749 | `f7e113bad83ef8aa436175e8aafbb3935ed1ed00` |
| `research/figures/generated/calibration_transport.png` | 128 | 51401 | `a603d5e81ad3fecf1c9088010b2ea036fb83e7d4` |
| `research/figures/generated/claim_status_audit.pdf` | 533 | 19317 | `3c6fb710bb2c9f813316f9c05255d8528eb5d695` |
| `research/figures/generated/claim_status_audit.png` | 89 | 49094 | `b1a7c34c73d5eba18eabaf19ebe904eac766cff8` |
| `research/figures/generated/critical_ablation_suite.pdf` | 923 | 37376 | `11de2a96352210f91342fbff5f20830c90b96166` |
| `research/figures/generated/critical_ablation_suite.png` | 611 | 231663 | `321e0d24888e3bba65e8e7fb3e2d8dc04e471da7` |
| `research/figures/generated/dynamics_suite.pdf` | 985 | 42838 | `bbcc44238746eac7f200f4e2af064e48db952c96` |
| `research/figures/generated/dynamics_suite.png` | 670 | 253915 | `92b2236e77c696033ed6691acff630cee93065ee` |
| `research/figures/generated/experiment_evidence_map.pdf` | 630 | 22553 | `ed83edc8959dfc27b323f32ccd48beef95e1c15f` |
| `research/figures/generated/experiment_evidence_map.png` | 286 | 81839 | `fa11e406697da0ef2e94cc5cb5b1c94fd0af8035` |
| `research/figures/generated/experiment_zero_learning_curves.pdf` | 1045 | 42580 | `71eeb06f04d6c36c5c81786a26adb869f7c30bf2` |
| `research/figures/generated/experiment_zero_learning_curves.png` | 991 | 335784 | `7e833a1d56ac26848c9942fdded1588ba1ef7694` |
| `research/figures/generated/falsification_phase.pdf` | 1043 | 39485 | `6a3a1732c4bf68c717f245a1db18836bbcacf7b7` |
| `research/figures/generated/falsification_phase.png` | 896 | 261328 | `a3f605e95c3fedc908469f750c38adfcdfab74eb` |
| `research/figures/generated/generator_shift_replication.pdf` | 1086 | 42846 | `3159dc3727febb350d53f3c001a2a180db7eca93` |
| `research/figures/generated/generator_shift_replication.png` | 1642 | 358217 | `42e919207ab8eccb1110e316bdaf9a5c7249d214` |
| `research/figures/generated/hard_halting.pdf` | 1022 | 39663 | `59228f4d7c9d110eb27f175a16790e8dfef3bb80` |
| `research/figures/generated/hard_halting.png` | 812 | 251541 | `088a2eb3b2612ac29a654e9e6cc524e7dc9f8aa3` |
| `research/figures/generated/neuro_task_suite.pdf` | 905 | 37091 | `09a1ae440c9c2fc694725b1ff3b8cf29c8b4de63` |
| `research/figures/generated/neuro_task_suite.png` | 660 | 261126 | `cb2c055686060568ab52187f72f7150135ce6678` |
| `research/figures/generated/observable_probe.pdf` | 1086 | 50788 | `e5c44be37200b3243dc60dfdd6d530fd60428934` |
| `research/figures/generated/observable_probe.png` | 1055 | 304485 | `21107855a1d905139c983aad84acb332e35181eb` |
| `research/figures/generated/ood_separability.pdf` | 657 | 24179 | `e50730ff95a1f0eedc93dee6f3319bb76c121f43` |
| `research/figures/generated/ood_separability.png` | 137 | 73319 | `98bcc6db91326108b52b6277d434aee4359be0e5` |
| `research/figures/generated/robustness_world_sweep.pdf` | 1078 | 43010 | `9029fb16f0efeccde7819b4759c2aa8b1db208c9` |
| `research/figures/generated/robustness_world_sweep.png` | 1154 | 326501 | `fae8e3783e098945a34a266b4c78331dd48d4673` |
| `research/figures/generated/sample_compute_frontier.pdf` | 667 | 27709 | `fe67b1fa0a798ab2127a96b85617f56b8b4915f7` |
| `research/figures/generated/sample_compute_frontier.png` | 149 | 71429 | `e6c8b6afb09ffe3899d70578c3247330f35a725d` |
| `research/figures/generated/surprise_taxonomy.pdf` | 532 | 19482 | `4bba2b96c66e501a8eeb2e94f028ae7d88e6e378` |
| `research/figures/generated/surprise_taxonomy.png` | 265 | 55364 | `6532cf2878151362863d66338e07efc464375490` |
| `research/figures/generated/training_law_suite.pdf` | 1057 | 43113 | `6a333138f12a2bb8fabe7228d5fd3457e87861a4` |
| `research/figures/generated/training_law_suite.png` | 913 | 256888 | `a5d60f8a2dc118517622dc7f01c3dc654cf43998` |
| `research/figures/generated/trajectory_signature.pdf` | 1134 | 58484 | `2b5258576b453e36c0ec20c72f239c4eb8fdbbba` |
| `research/figures/generated/trajectory_signature.png` | 1465 | 440215 | `55bf8528a98dff95f93fe06052170412252931a5` |
| `research/freeze_candidate_law.py` | 160 | 6901 | `a5f01e132a161214672d0a60b062d4e2175d01e6` |
| `research/laws/FROZEN_CANDIDATE_001.json` | 116 | 3362 | `bb4b0cbe167293dfb31d2298ec182addf45bd4eb` |
| `research/review_report.json` | 7 | 105 | `3aa21375f74e25996e0a078f51fb20be9d608b1a` |
| `research/statistics.py` | 284 | 11467 | `e636127e814ad98198c61a1ad52412d69e92b016` |
| `scripts/build_dashboard_data.py` | 178 | 6227 | `185e0b47992c7ba6f2ec6ef96c2abb38bb076a28` |
| `scripts/verify_release.py` | 355 | 13668 | `d13e61c8a9db6693fdbe507e7719241673601a35` |
| `tests/test_active.py` | 53 | 1789 | `fc910cd0b1889ac257dc27e9ec251789821b9579` |
| `tests/test_advanced_models.py` | 101 | 4078 | `f30ee87c1716124026b85f05324d69cffa72381f` |
| `tests/test_adversarial_reviewer.py` | 7 | 201 | `8e38ed0f9df7295f38e19cb3d296647054c76701` |
| `tests/test_computational_laws.py` | 96 | 3769 | `2a245213affeb2ed2b673f6ece75c6c5d7048e4c` |
| `tests/test_dashboard.py` | 14 | 573 | `8aa176c91beec907f513ee512a8f5ccacc2aa819` |
| `tests/test_discovery.py` | 25 | 933 | `0002e3a4b5be45eb7367e945cb2ec166ddb75fe6` |
| `tests/test_evaluation.py` | 37 | 1371 | `5dca0bbb9c9064fc0bb6464f93ba0152299b0d48` |
| `tests/test_falsification_phase.py` | 19 | 592 | `b1519003198e52729289af4e35e37e14e62c1dde` |
| `tests/test_independent_confirmation.py` | 31 | 1226 | `3e6ebf8e1b6e2dada8b9269b5970a0dafbe8863e` |
| `tests/test_independent_discovery.py` | 81 | 2759 | `4e632c98d9d578e65d93f7427bb69174c5e61981` |
| `tests/test_independent_tasks.py` | 84 | 3972 | `d2378d73edd41e63a19f4f5c4ba972a0c25ebb55` |
| `tests/test_learning.py` | 40 | 1547 | `52d252dde5a5e7015fa3a3d84abbe482bc1b6de1` |
| `tests/test_mechanisms.py` | 140 | 5199 | `5c6bf9f80eddb334a810c785f762fd3411e7fc50` |
| `tests/test_models.py` | 163 | 6963 | `2d5c6ccfd46bed078f927b2fb1e80d24aaddd3c4` |
| `tests/test_neuroworld.py` | 54 | 2191 | `993e6b1bc3970d493b549d8fc6900323b51b3bb4` |
| `tests/test_observables.py` | 39 | 1290 | `f77e52949713192ae20c494fb4edfa6c0c79856f` |
| `tests/test_paper_artifacts.py` | 55 | 2040 | `5a9f68dbb7b150857c1e0990acbfc009619728f1` |
| `tests/test_provenance.py` | 42 | 1364 | `7eccdaa26380e35d5e34b041608698bbcac4023e` |
| `tests/test_qn_grand_preflight.py` | 23 | 791 | `8af40330e15401be47dd2c5e62ce74e0d55cb6f6` |
| `tests/test_registry.py` | 95 | 3455 | `9d0a3ae3c149fbc50600b622800b5ee9fa6675ac` |
| `tests/test_release_verifier.py` | 42 | 1293 | `c3d355ee09bfba7adc62b92a30b8628322b077cc` |
| `tests/test_shift_pilot.py` | 52 | 1861 | `cf15a65e5eef68f06a1f07a2d9f368ecfc75c858` |
| `tests/test_shifts.py` | 100 | 4206 | `8e16ffbb78c2f03748786f025c2f6c97341d6f15` |
| `tests/test_statistics.py` | 73 | 2366 | `948baffbcc9382e7eb9979a7144b051dc6f6e7c7` |
| `tests/test_tasks.py` | 43 | 1723 | `da3646bf4d94f53ee4953ea11b9f8537ab31dbf3` |
| `tests/test_validity.py` | 42 | 1522 | `55bc9301fcfcbfa3d54de6ccfaea362bb9a17e66` |
| `uv.lock` | 879 | 131423 | `138e3764ecc623851d54109cc915f24ed29657d2` |
