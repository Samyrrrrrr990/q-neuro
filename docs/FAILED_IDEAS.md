# Failed or Non-Decisive Ideas

## Full-data Experiment Zero as a discriminator

**Why it seemed useful:** A matched 20,000-case comparison was the fastest way to test whether
ordered operator states could learn the synthetic causal task.

**What happened:** Transformer, real operator, and complex operator all reached approximately 0.995
or higher top-1 and perfect chronology-pair accuracy.

**Why it failed as a discriminator:** The task saturated. A difference of a few errors cannot
identify the mechanism or support an architectural headline.

**Decision:** Retain it as a correctness and order-sensitivity result; use lower-data curves and
harder generator shifts for architecture comparisons.

## Complex phase as an automatic advantage

**Why it seemed plausible:** Complex relative phase can express constructive/destructive
interference and might encode order more compactly than a real state.

**What happened:** Learned phase is necessary for the trained complex model, and complex top-1
exceeds real from 500–5,000 cases. However, complex is worse at 250 cases and has worse NLL, ECE,
and runtime at every tested size.

**Why the broad idea failed:** Functional use is not comparative benefit. The current complex model
does not establish a new Pareto frontier.

**Worth revisiting:** Yes, only with a trained two-channel real control, calibration-aware loss, and
changed causal generators.

## Asymmetric-input comparison (QN-000002)

**Issue:** The MLP alone received demographic context.

**Resolution:** Preserved and marked the run superseded. QN-000003 equalized inputs and replicated
the qualitative result. The failure is methodological, not architectural, and remains documented to
prevent accidental citation of the wrong run.

## Operator states as the strongest low-data baseline

**Why it seemed plausible:** QN-000004 showed both operator models learning the chronology task with
far fewer examples than the tiny Transformer.

**What happened:** After validation tuning, a 19,656-parameter GRU reached 0.920 in-domain top-1 at
250 cases, compared with 0.774 real operator and 0.699 complex operator.

**Why the idea failed:** The original Transformer was not a sufficient proxy for conventional
sequence learning. A compact recurrent inductive bias matches this task extremely well.

**Scientific value:** The GRU then collapses under generator shift, revealing that in-domain sample
efficiency and robustness are different questions. Future claims must report both.

## In-domain temperature scaling as a shift-calibration fix

**Why it seemed plausible:** Scalar temperature scaling is cheap, preserves class rankings, and can
correct over- or under-confidence using validation data.

**What happened:** Temperatures fitted on the original world worsen shifted ECE for every tested
model. For complex operators, moderate-shift NLL rises from 1.459 raw to 4.341 calibrated.

**Why it failed:** The confidence distortion changes with the evidence distribution. A scalar fitted
to the source distribution encodes the wrong correction under shift, especially for
magnitude-squared measurements.

**Decision:** Report raw and calibrated metrics, but do not use source-fitted temperature scaling as
a robustness intervention. Future calibration work must be shift-aware without using test labels.

## Held-out composition at 3,000 cases as an architectural discriminator

**Why it seemed useful:** Excluding selected finding conjunctions should test whether coupled or
interference-like states recombine evidence more effectively than conventional sequence models.

**What happened:** Complex and two-channel real both reach 1.000 held-out top-1; real reaches 0.999
and GRU 0.995. Operator-model reference-versus-held-out gaps are effectively zero.

**Why it failed as a discriminator:** The causal factors and individual evidence items remain easy
to learn, and 3,000 examples saturate the tested construction. Selecting test cases containing a
conjunction does not itself require an inseparable representation.

**Decision:** Keep the result as a competence check. A future composition experiment must reduce
data, require an XOR-like or tensor-coupled factor, and include a parameter-matched multiplicative
baseline.

## Complex states as automatic protection against premature collapse

**Why it seemed plausible:** Amplitude and phase could retain several hypotheses even when the
final measurement is uncertain.

**What happened:** On observationally identical chronology twins, complex pair NLL is 2.581 and
valid-twin probability mass is 0.212. The ordinary real operator obtains 1.148 NLL and 0.836 mass.

**Why it failed:** A rich internal state does not force a calibrated measurement. Cross-entropy on
fully observed cases rewards sharp decisions, and the current magnitude-squared readout has no
objective encouraging metastable mass over observational equivalence classes.

**Worth revisiting:** Yes, with ambiguity-aware training, energy/coherence observables, or an
explicit set-valued target. The current architecture alone is not the solution.

## Expected information gain as a universally superior query rule

**Why it seemed plausible:** Selecting the finding with the lowest expected posterior entropy is
the standard rational objective when all query costs are equal.

**What happened:** It improves complex and MLP evidence-curve AUC over a fixed information order,
but reduces Transformer AUC from 0.528 to 0.359. The real operator is essentially unchanged, and
GRU/two-channel effects vary across seeds.

**Why it failed broadly:** The policy evaluates counterfactuals through the same model whose
partial-evidence calibration may be wrong. Entropy minimization can steer toward model artifacts
rather than resolve the true label. Its diagnosis-weighted outcome model also ignores conditional
dependencies.

**Decision:** Treat acquisition policy and predictor as a coupled system. Future variants require
held-out policy validation, calibrated outcome models, and query-cost sensitivity; never infer
active competence from full-information accuracy alone.

## Dissipation as an automatic diagnostic-elimination advantage

**Why it seemed plausible:** Diagnosis must eliminate hypotheses, so learned damping could provide
the irreversible component absent from coherent evolution.

**What happened:** Dissipative-only dynamics reach 0.438 moderate-shift top-1 and almost completely
fail chronology pairs (0.008). Hamiltonian reaches 0.556/0.983. Adding dissipation to Hamiltonian
produces 0.550/0.923, not an improvement.

**Why it failed:** Elementwise damping plus renormalization does not encode targeted contradiction;
it can erase path information without shaping a useful energy landscape.

**Decision:** Remove generic damping from headline variants. Revisit only as diagnosis-conditional
Lindblad-like channels or explicit contradiction operators with a matched ablation.

## Soft adaptive depth as compute savings

**Why it seemed plausible:** ACT-style halting weights provide a differentiable way to assign less
diagnostic time to easy cases.

**What happened:** The adaptive attractor learns 5.25 expected steps out of eight, but all eight are
still evaluated and then mixed. Shift accuracy differs from fixed attractor by only +0.011.

**Why it failed as a compute claim:** Expected depth is not executed depth. Without hard batched
early exit or sparse per-case execution, the wall-clock graph is unchanged.

**Decision:** Do not report compute reduction. Implement a hard validation-tuned velocity exit and
measure actual operator calls and latency before revisiting.

**Follow-up:** QN-000023 implements that exit and does realize a 75% state reduction and roughly
80% latency reduction versus soft ACT. However, all cases halt at the minimum of two states. The
compute result is therefore fixed truncation, not evidence for adaptive per-case reasoning. Later
steps worsen calibration without improving accuracy.

**Revised decision:** Replace the eight-state attractor with a fixed two-state control in future
comparisons. Reopen adaptive time only on a task where validation selects a non-degenerate halt-step
distribution and where difficulty predicts the executed depth.

## Fixed NeuroWorld factor graph

**Why it seemed plausible:** Message passing over mechanism, localization, temporality, and context
groups should align with the simulator's causal construction.

**What happened:** The GNN obtains 0.319 in-domain and 0.184 shifted top-1, below logistic, MLP, and
operator controls.

**Why it probably failed:** The hand-built adjacency is coarse and shared message updates
oversmooth finding identity. Matching the generator's broad groups is not equivalent to recovering
its label-specific causal parameters.

**Decision:** Retain as a negative baseline; do not tune the graph against test worlds.

## Density rank as useful relational capacity

**Why it seemed plausible:** A higher-rank density factor can represent more independent coherent
hypothesis relations and should reduce the restriction of a nearly pure state.

**What happened:** Ranks 1, 2, and 4 reach 0.449, 0.453, and 0.441 shifted top-1. Rank 4 has more
parameters but lower in-domain accuracy, worse ambiguity NLL, and lower chronology-pair accuracy.

**Why it failed:** Cross-entropy supervises only the diagonal measurement. Off-diagonal capacity is
free to vary without a target that makes relational state predictive, and extra factor channels can
make optimization less identifiable.

**Decision:** Do not scale density rank. Revisit only with a later-resolution or multi-observable
objective that can falsify whether off-diagonals contain useful information.

## Complex state as a uniquely hierarchical representation

**Why it seemed plausible:** Phase-sensitive composition might separate mechanism, localization,
temporality, and context into readable observables even though only diagnosis is supervised.

**What happened:** Linear probes recover every factor accurately from the complex operator state,
but GRU and diagonal state-space representations are generally stronger. Complex-minus-GRU probe
accuracy is negative on all four factors. Across architectures, probe strength and shift robustness
are only moderately related (`r=+0.45`).

**Why the uniqueness claim failed:** The simulator makes its causal factors broadly predictive, so
many sufficiently expressive sequence models retain them. Linear accessibility does not imply an
architecture-specific decomposition or show that the final head relies on the probed factor.

**Decision:** Retain factor probes as state diagnostics. Do not describe them as uniquely
Q-Neuro, disentangled, or intrinsically interpretable. Require causal interventions on observables
before claiming that a factor controls a diagnosis.

## Phase-coded gradients as a robustness optimizer

**Why it seemed plausible:** Mechanism, localization, and diagnosis gradients could conflict, and
rotating auxiliary updates in explicit real/imaginary parameter planes might preserve orthogonal
information that scalar gradient averaging cancels.

**What happened:** PGO reaches 0.633 shifted top-1, slightly above diagnosis-only AdamW but below
ordinary multi-objective AdamW (0.635) and PCGrad (0.635). It takes 5.43 seconds versus 2.96 for the
same-objective AdamW control. Task-gradient cosines are weakly positive, so the proposed conflict
mechanism is mostly solving a conflict that is not present.

**Decision:** Do not promote PGO. Retain it as a tested optimizer and revisit only on a preregistered
task with measured negative gradient cosines. A future version also needs a real/two-channel
control to isolate whether complex rotation matters.

## Local pretraining as a benign initialization

**Why it seemed plausible:** Fast transition-local association followed by slow global learning
could combine continual plasticity with end-to-end refinement.

**What happened:** The hybrid reaches almost perfect source accuracy and materially better
ambiguous-pair NLL, but its 1,000-case shifted top-1 is 0.419 versus 0.620 for AdamW. At 250 cases it
reaches 0.845 source accuracy and only 0.087 shifted accuracy. The failure repeats across seeds.

**Why it failed:** The local prototype target strongly associates source-world tokens with class
codes. Global cross-entropy sharpens that basin rather than rebuilding a transferable evidence
geometry.

**Decision:** Treat local initialization as a potential source-locking risk. Future local rules must
use counterfactual or multi-world signals and be evaluated before global fine-tuning, not merely
after it.

## Centroid ZeroBackprop as an end-to-end alternative

**Why it seemed plausible:** Random complex dynamics may provide enough nonlinear features for a
closed-form class prototype readout, eliminating reverse-mode differentiation.

**What happened:** The prototype is fast and gradient-free but reaches only 0.133 in-domain top-1,
0.139 shifted top-1, and zero chronology-pair accuracy at 1,000 cases.

**Decision:** The centroid version is refuted. Further ZeroBackprop work requires plastic dynamics,
contrastive negative phases, random feedback, or perturbation-based credit assignment—not another
readout-only variation.

## Final complex state as a converged disease attractor

**Why it seemed plausible:** Repeated noncommutative evidence updates might settle into stable
hypothesis basins, making the final state an attractor-like diagnostic object.

**What happened:** QN-000025 measures a mean normalized final-token velocity of 0.175. States follow
long paths (2.956 normalized units) and remain sensitive to the final observation. Chronology twins
separate reliably, but that is path dependence rather than convergence.

**Decision:** Use “operator state,” not “attractor,” for the complex model. Reserve attractor claims
for an explicit fixed-point, basin, perturbation-recovery, or Lyapunov experiment.

## Automated Pareto ranking as a universal winner selector

**Why it seemed useful:** A single ranked candidate would make the next architecture decision
simple and could appear to summarize the completed matrix.

**What happened:** QN-000026 retains 10 of 24 architectures and seven of nine training laws on
their separate six-objective Pareto fronts. Accuracy, robustness, chronology, calibration, and CPU
time trade against one another; inference timing is also not commensurate with training timing.

**Why the idea failed:** Pareto analysis removes dominated choices but cannot supply scientific
values or collapse incompatible objectives. A broad frontier is the correct result, not a search
failure.

**Decision:** Use the engine to expose tensions and preregister falsifiable mutations. Do not call
its proposals discoveries, optimize against a hidden aggregate score, or compare timing across
contexts.

## A single cross-family calibration for the transport-defect estimator

**Why it seemed plausible:** If accumulated covariance defect really controls how far two
semantics-equivalent systems drift apart, one calibration should predict the final gap wherever the
defect is measurable. That is the whole content of the `transport_covariance_conjecture` at the
level Gate D tests it.

**What happened:** QE-000009 generated 216 transport traces across four discovery families
(permutation, scaling orbit, native complex, dense-vs-factorized) and scored every candidate and
baseline by leave-one-family-out held-out R². Every out-of-family fit was worse than predicting the
mean. `cumulative_defect` beat all baselines on one family; Gate D requires two. `amplified_defect`
beat all baselines on none.

**Why it failed:** Not for lack of signal. *Within* family, `cumulative_defect` reaches R² 0.962 on
factorization and 0.812 on the scaling orbit and beats every baseline on both. The problem is
calibration: family medians on the target scale span about 6.5 orders of magnitude, from ~1e-7 for
permutation — where the map is conjugate and there is essentially nothing left to predict — to
~1e-0.6 for the scaling orbit. Forcing one global slope and intercept across that range is worse
than the mean baseline everywhere.

**A correction made along the way:** `one_step_defect` was originally listed as a candidate. With a
mapped initialization `e_0 = 0`, so the first re-coupled step's defect is bit-for-bit the first
step's predictive divergence — identical in all 216 rows. It was a baseline wearing a candidate's
label, and it would have "tied" its way toward the gate. It is now excluded, with the reason
recorded in `qneuro/equivalence/estimators.py`.

**Decision:** QE-000010 refused to freeze an estimator and left rungs 5–8 sealed. No features were
added after the outcome was seen. Recorded as `FAIL-006`.

**Worth revisiting:** Yes, but only as a *new* registered attempt with a stated reason. The obvious
directions are a per-family random intercept — which changes what is being claimed, since a law
needing per-family calibration is a weaker object than the conjecture states — or restricting the
population to families where the map is non-conjugate, since two of the four sit at the rounding
floor by construction and contribute no predictable variation.

## The equivalence-breaking phase boundary as a general phenomenon

**Why it seemed plausible:** DISCOVERY-001 found a sharp, analytically derived boundary at
`rho = eta*lambda_max/2 = 1` separating "equivalent models train alike" from "one converges and its
exact equivalent diverges." On least squares it predicted the target's stability with 99.12%
accuracy, zero false alarms in 1,476 cells, and a confirmed differential prediction that Adam shows
no boundary. It also explained the Gate D failure. It looked like a real phenomenon.

**What happened:** Two frozen, hashed predictions were opened against an untouched nonlinear system
(ReLU MLP, cross-entropy, full-batch GD, float64).

- **P1 was vacuous.** The frozen grid used learning rates 0.05–0.2 against a measured curvature
  range of 0.51–10.9, so `rho` never reached 1.1. There were 197 opportunities for a false alarm and
  **zero** for a miss. It reported "passes: True" and that reading is meaningless. Preserved, not
  re-tuned; P2 added a non-vacuity guard requiring cells on both sides of the band.
- **P2 failed outright.** With the learning rate placed per-cell so that `rho` equalled its target
  by construction, **96 of 96** cells at `rho >= 1.1` converged. The SGD divergence rate was
  0.0000 even at `rho = 3.0`, with growth ratios of 1.03–1.27 against a threshold of 2.0.

**Why it failed:** Exactly the failure mode frozen in P2 beforehand. `rho` is computed from the
Hessian at the mapped initialization, but curvature is not constant for a nonlinear model. A ReLU
network under cross-entropy relocates to flatter regions and the loss saturates, so an initially
over-large step does not produce exponential growth. The defect is not calibration — it is that one
initialization-time curvature does not govern a nonlinear trajectory.

**What survived:** The `rho < 1 => stable` direction. Zero false alarms across 269 scored cells over
both attempts. That half has never produced a counterexample.

**Decision:** DISCOVERY-001 does not promote. It stands as a statement about **quadratic objectives
only** and may not be described as a general property of equivalence breaking. Recorded as
`FAIL-007`. No third prediction was issued: two attempts are consumed, and a third would require a
materially different estimator — which would be a new discovery, not a rescue of this one.

**Programme consequence:** This converts a stated risk into a measured one. Gate C passed on affine
maps with exact Lipschitz constants, and that pass was explicitly scoped as saying nothing about
nonlinear models. P2 is direct evidence that transport intuition built on analytic microcosms does
not extrapolate.

---

## FAIL-023 / FAIL-024 / FAIL-025 — Adaptive depth: a real speedup that cannot be reported

**Idea:** Let a recurrent model decide for itself how long to think. On `chase_to_goal` the model
must report how far away a goal node is along a hidden cycle; the answer is discoverable only by
following the chain, so the required depth genuinely varies per example and nothing in the input
announces it. A fixed-depth model must always pay the worst case. An adaptive one should not.

**What worked.** `Q3Arrival` halts when it detects arrival and treats the halt step itself as the
answer. It reached **0.9995 accuracy at 4.53 average steps** against a fixed-depth baseline's
1.0000 at 8.00 — a **1.77x** reduction in inference compute at 27,970 parameters against 28,360.
That number is real. Seeds 0, 2 and 3 all reproduce it to within 0.0006 accuracy and 0.00 steps.

**FAIL-023 — the frozen distribution law died.** `QNEURO3-Q3-P1` predicted in advance that the
saving is a property of the workload rather than the architecture: `average steps = E[distance]`
to within 0.25. Opened once against three untouched difficulty distributions, it failed three of
four cells. What broke was **accuracy**, not step count — 0.4356, 0.2461 and 0.3334 against a
required 0.99. For Q3 the halt step *is* the answer, so those step counts were untestable rather
than merely wrong. The Q3b answer-decoupling control was not run, because it would have measured
training reliability a second time under a different name; the question of whether Q3's result
depends on answer-equals-step-count therefore **remains open**.

**FAIL-024 — the headline was one seed in a bimodal distribution.** The boring explanation was
tested first, and it was right. Re-running the *original* task at the *original* training budget
across seeds 0–4 gives 0.9994, **0.4308**, 0.9998, 1.0000, **0.4913**. Over twenty runs spanning
two task constructions and two training budgets, only **7 of 20** reach 0.99. The distribution is
bimodal — every run is either 0.9994–1.0000 at 4.54 steps or 0.42–0.57 at 5.2–6.1 steps, and
nothing lands in between. Training volume changes nothing; seed decides.

The part worth carrying to any other adaptive-compute system: **the failure mode is silent and
mimics success.** A collapsed run reports 5.2–6.1 average steps — a plausible, non-degenerate,
adaptive-looking allocation comfortably under the fixed depth of 8. Read the step counter alone
and all thirteen failed runs look like working elastic models delivering a 1.3–1.5x saving. Only
the accuracy column shows the model is wrong more than half the time.

**The matched control is what settles it.** `Q0Fixed` under identical conditions reaches ≥0.99 on
**10 of 10** seeds, minimum 0.9919. The task is not flaky and the budget is sufficient; the
unreliability belongs to the adaptive architecture. So Q3 buys 1.77x less compute in exchange for
a 40% chance of a silently broken model, against a baseline that never breaks. Expected accuracy
across seeds: **0.78 for Q3, 1.00 for Q0.**

**FAIL-025 — the repair failed, and failed informatively.** `QNEURO3-Q4-P1` was frozen before any
code was written: Q3 is bimodal because nothing forces its state to track position on the chain,
so a training-only per-step position readout (zero inference cost, identical halting rule) should
lift reliability to ≥9/10. Result: **0 of 10** on both tasks. The kill condition fired as written
and no second variant was issued.

It did half of what was predicted, and the useless half. The collapse mode is genuinely **gone** —
no Q4 run falls below 0.6322 where four Q3 runs sat at 0.42–0.51. But the good mode went with it:
Q4's best run anywhere is 0.9500. Grounding the state cured the variance by pulling both tails to
the middle. The likely reason, offered as explanation and not as an established result, is that a
24-way cross-entropy at every one of eight steps dominates the single scalar arrival signal, so
the state is optimised to name the node rather than to make the one binary decision that matters.
An easier auxiliary task can absorb the representation.

**Decision:** Q-Neuro 3.0 cycle 1 closes negative. Adaptive depth does not beat fixed depth on
this task family once reliability is counted alongside compute. Nothing here shows adaptive depth
cannot win elsewhere; it shows that this instance does not, and that a speedup figure without a
matched accuracy *and* a matched seed-reliability rate is not a result.

---

## FAIL-026 … FAIL-031 — Cycle 2: six ways to be wrong before one prediction survived

Cycle 1 closed with adaptive depth losing to fixed depth. Cycle 2 found the cause, fixed it, and
then spent almost all of its effort trying to destroy what followed. Six things died on the way.

**The fix, first, because everything else is a reaction to it.** Q3's bimodality was diagnosed by
looking at accuracy *by distance*: failing runs are perfect at distance 1–2 and collapse from 3
onward. The state stops carrying position. **RMS-normalising the state after each hop** took the
variant sweep from 11 of 24 seeds to **6 of 6**, and every successful run lands on exactly
1.0000 at 4.54 steps. Three other single-variable interventions were run at the same time and did
not help: a goal-match feature (3/6), dense per-step halting supervision (3/6), both together
(2/6). Layer normalisation in recurrent networks is textbook and no novelty is claimed for it.

**FAIL-027 — nothing predicts the mode earlier than accuracy does.** With a genuinely bimodal
system finally in hand — the thing `DISCOVERY-002`'s committor branch needed and never had — 24
runs were instrumented with 21 mechanistic observables logged every 100 optimizer steps: linear
decodability of the current node, of hops remaining, of the goal indicator; halting-distribution
entropy and per-step profile; attention entropy; state effective rank. **None separates the modes
earlier than validation accuracy.** First perfect separation is step 1900 of 4000 for accuracy,
decodability, loss and halting entropy alike. The best early AUC at step 500 is 0.731 for
`dec_remaining` against accuracy's 0.521 — inside multiple-comparison noise for 21 observables
across 7 timepoints. Early validation accuracy is the operative predictor, kill-and-restart on it
is standard practice, and the branch closed without a prediction being issued.

**FAIL-026 — a control that everything passed.** The first decoupled-answer task kept the goal at
node 0 and asked for the goal's label. A fixed, known goal identity makes its label directly
addressable by attention at any step, with no walking. Every configuration scored 1.0000 — including
runs whose halting was 30% correct, which is what gave it away. *A control that every model passes
is not a control.* Rebuilt content-addressed: walk to the **first** node whose label matches a
per-example query, then report **which node** that is. Shortcut audit on the rebuilt task: distance
alone 0.064, guessing any node carrying the query label 0.291, chance 0.042.

**FAIL-028 — then it was unlearnable, twice over.** The rebuilt task defeated every model at chance
for two separate reasons. The label of the node just moved to was never available to the halting
head, so the predicate was literally unanswerable; and the head was a linear map on a
concatenation, which cannot express *"these two are equal"* at all. The second defect is the same
expressivity gap the goal-match variant had exposed days earlier, reintroduced from scratch in a
new task.

**FAIL-029 — the mechanism was false.** With the task fixed, the separation was dramatic: on
associative lookup a final-state readout scores 0.22 while reading the answer from an input-selected
step scores **1.0000**. `QNEURO3-ATTRIB-P1` was frozen to explain it — the fixed model must
*transport* the matched identity through the remaining iterations and degrades in proportion to the
carry distance. Opened once: accuracy by distance is **flat at ~0.22**, and the required
`d=max_depth` minus `d=1` margin of 0.30 measured 0.02 and 0.07. The model fails uniformly,
including where nothing is carried at all. Kill condition triggered exactly as the frozen
`anticipated_failure_modes` predicted in its first line.

Diagnosing it exposed a confound in our own comparison — the attribution models were told which
step was the match and the fixed model was not. The matched-supervision control removed it, and the
separation survived: `fixed_supervised` still scores 0.232–0.243, an explicit learned latch scores
0.222–0.239, and mean pooling reaches 0.847 on one seed of three. Five distinct fixed-depth
alternatives fail where input-selected readout succeeds on every seed.

**FAIL-030 — and it did not generalise.** `QNEURO3-TRANSFER-P1` opened the question on an untouched,
qualitatively different family: streaming threshold-crossing, no attention anywhere in the core.
Required: a final-state readout at least 0.20 below per-step selection. Measured: **0.9351 against
0.9425 — a gap of 0.007.** The kill condition applied verbatim. The attribution result is scoped to
associative-lookup tasks and no principle is claimed from it. The first anticipated failure mode
named the reason before the run: a 9-value answer alphabet and a one-dimensional running sum make
latching easy enough that the final state suffices. The sharpest detail is that the *best* model on
this family is the explicit latch at 0.9760 — one of the controls that failed on query-chase. **The
two families reward opposite designs.**

**FAIL-031 — and it buys no capability.** `QNEURO3-EXTRAP-P1` asked the one thing halting could
uniquely offer: iterating past the depth it was trained at, since its stopping rule is a local
condition rather than a fixed count. Trained at length 12, evaluated at 16: **0.8328 / 0.3528 /
0.6548** on the unseen indices against a required 0.80 on every seed. And E2 *inverted* — the
**unnormalised** model extrapolates at 0.9136 against the normalised model's 0.6135. Normalisation
costs 0.30 of extrapolated accuracy while buying about 0.05 in distribution.

That inversion bounds the fix that opened the cycle. Normalisation rescued halting on associative
lookup (11/24 → 20/20) and damages extrapolation on streaming. It is a family-specific engineering
choice with a real trade-off, not a principle, and the final architecture exposes it as a flag
rather than a default.

**What was left standing.** One thing, and it had already transferred prospectively as T3 of a
prediction that otherwise failed: halting on a supervised predicate costs nothing in accuracy and
returns the workload's full compute saving. Mean steps track `E[predicate index]` to within 0.1
across six settings — 4.45 of 8, 6.55 of 12, 2.51 of 4, 4.54 of 8, 6.60 of 12, 6.14 of 24. That is
optimal allocation, which also means **the size of the saving belongs to the workload, not to the
architecture** — precisely what the long-dead `QNEURO3-Q3-P1` said and could not test.

---

## FAIL-032 / FAIL-033 — The ceiling moved, then stopped moving

The confirmed result came with a ceiling: adaptive halting's advantage decayed to 0.97× at batch
256. That was measured under **lockstep** execution, and the wording it was written up in —
"batching destroys the advantage" — said more than the evidence.

**The correction, before any new work.** Heterogeneous halt depths create a *straggler effect under
lockstep batching*. The per-example FLOP saving is untouched; what changes is whether the execution
policy realises it. Lockstep executes `n · max_i d_i` rows where useful work is `Σ_i d_i` — at batch
256 on the lookup family, 5888 rows against 1415, a **4.16× pure waste ratio** that a better runtime
could in principle avoid.

**Prior art first, and it is decisive.** Active-set compaction is the standard early-exit loop, is
what MoE dispatch does at every layer, and is what sequence packing does for variable-length
attention. Continuous batching is iteration-level scheduling as deployed in LLM serving. Length
bucketing predates transformers. **No novelty was available and none is claimed**
(`docs/PRIOR_ART_RUNTIME.md`). All four were implemented as baselines, and each is verified to
reproduce lockstep's answers exactly — a check that immediately caught a real bug, where deferred
compaction let a fired row overwrite its own answer.

**What compaction actually recovers.** On the lookup family, against the same matched-accuracy
baseline, the advantage stops decaying and starts growing: 1.28× at batch 16, 1.46× at 32, 1.59× at
64, 1.60× at 128, **1.95× at 256** — against 1.04×/1.10×/1.10×/0.98×/1.01× under lockstep. At batch
1 and 4 compaction is a small *loss* (0.80–0.92×), because there is nothing to compact.

**FAIL-032 — the cost model missed.** `QNEURO3-RUNTIME-P1` froze
`T = c_step·rows + c_launch·iterations + c_compact·compactions` with all three constants measured on
the target family's raw forward, and predicted the crossover at **batch 45**. Measured: the
crossover is **below 16**. Relative errors of 55.1% and 45.1% at batches 16 and 32 against a 25%
band. The model is accurate exactly where compute dominates — 1.0% at batch 128, 11.5% at 256 — and
wrong where overhead does, because it over-charged compaction at small batch (15 modelled
compactions against 10 measured) using a `c_compact` taken from a synthetic gather of all six state
tensors. Kill condition applied: **no predictive runtime equation is claimed**, and the equation was
not patched and re-issued.

*A procedural defect was caught before this one opened.* The first hash recorded for it could not be
re-verified from disk: integer keys in the predicted curve sort numerically in memory and
lexicographically after a JSON reload, so the two serialisations differed. Found with no evidence in
existence and re-issued by hashing the round-trip. **A frozen prediction whose hash cannot be
re-verified from disk is not frozen.**

**FAIL-033 — and it does not transfer.** `QNEURO3-RUNTIME-P2` predicted the recovery would appear on
the streaming family: compaction ≥1.5× over the matched-accuracy baseline at batch 256. Measured
**1.065×**. Compaction over lockstep was 1.289× against a required [1.3, 2.2] — missing by 0.011.
Accuracy was not even matched: `select` scored 0.9336 against arrival's 0.9189, so the comparison
*flattered* arrival and it still failed.

The reason was in the frozen anticipated failure modes before the run: *"at c_step = 0.3343 µs a
removed row saves a third of a microsecond while a gather costs tens."* The streaming core is
attention-free and eight times cheaper per example-step.

**The boundary this establishes.** Compaction removes the lockstep straggler ceiling **when per-step
cost is large relative to gather cost** — 1.95× at 2.66 µs/example-step, 1.07× at 0.33. That is a
second boundary of exactly the same shape as the first. The advantage is real; every time it is
measured under a stronger control it turns out to be bounded by something that has nothing to do
with the mechanism.

---

## FAIL-034 / 035 / 036 — The final phase: a real dataset, and three dead branches

Sixteen frozen predictions in, with one pass and a niche that had survived every control, the last
phase asked the two questions that could still overturn it: does the mechanism beat *strong*
adaptive-compute baselines, and does it work on data I did not design?

**FAIL-034 — on real data it comes fourth of five.** UCI Human Activity Recognition, the dataset's
own canonical subject-disjoint split (17 train / 4 validation / 9 test subjects, chosen by the
dataset, not by me). 128-step inertial windows delivered in 16 chunks; classify the activity as
early as possible. Five arms over one identical core at identical parameter counts (63,271), three
seeds, protocol frozen and hashed before the test subjects were read:

| arm | test accuracy | mean chunks | train s |
|---|---:|---:|---:|
| fixed depth | 0.9127 | 16.00 | 3.6 |
| **ACT** (Graves 2016) | **0.9006** | **3.61** | 4.5 |
| confidence exit | 0.8811 | 2.57 | 3.4 |
| confidence exit @ matched compute | 0.8747 | 2.28 | 3.4 |
| **supervised halting (ours)** | **0.8112** | 2.39 | **8.4** |
| PonderNet | 0.5220 | 16.00 | 4.3 |

Real data supplies no ground-truth halt step, so the target had to be distilled from a teacher's
earliest confident-correct chunk — declared as prior art in the frozen record beforehand, and the
reason our arm costs 2.3× the training time. **The kill condition fired.** Supervised halting is
beaten by a method from 2016 that needs no teacher, and by a softmax threshold that needs neither.

The narrowing is the headline, not a caveat: **the mechanism earns its place only where the task
supplies a halt target.** On the three synthetic families where it does, the mechanism attains
optimal allocation at matched accuracy with 10/10 seed reliability. On a real workload where it does
not, it loses.

*What did transfer* is the runtime characterisation, completely: 6.38× at batch 1, 0.70× at batch
256 under lockstep, 1.63× once compaction is used. The execution-policy findings hold on real data
even though the mechanism does not win.

PonderNet's collapse to 0.5220 is almost certainly my implementation's fault rather than the
method's. ACT working well on the same core suggests the harness is fair — and a correctly tuned
PonderNet would only rank *above* us.

**FAIL-035 — complex fields add exactly nothing.** A genuinely complex state carried as
`complex64`, with Hermitian attention so phase can affect the halting decision, at matched *real*
parameter count (56,545 against 55,001): **1.0000 accuracy, 1.0000 halt accuracy, 6.15 mean steps on
3 of 3 seeds** — indistinguishable from the real control's 1.0000 at 6.14. Expressivity was already
known to be identical by exact realification (`FAIL-005`); the last remaining route was optimisation
dynamics, and that is now measured identical too.

**FAIL-036 — adaptive width is beaten by a smaller fixed model.** Letting the model choose both
depth and width, `C(x) = T(x)·N(x)`, with a per-step router over eight hidden groups:

| config | cost | accuracy | params |
|---|---:|---:|---:|
| adaptive depth, full width | 6.15 | 1.0000 | 57,065 |
| adaptive depth, **routed** 4/8 | 3.07 | 1.0000 | 57,065 |
| adaptive depth, **static** 4/8 | 3.07 | 1.0000 | **32,425** |
| adaptive depth, **static** 2/8 | **1.54** | **1.0000** | **20,105** |
| adaptive depth, routed 2/8 | 2.13 | 0.8143 | 57,065 |

A statically narrow model matches the router at identical cost with 43% fewer parameters, and beats
it outright at the aggressive setting. At binding capacity (`d = 16`) routing is *strictly worse*
than static narrowing — 0.032 against 0.115 at equal cost. Killed at both regimes.

Before the control was run, an accounting bug made routing look like a clean 2× Pareto win: the
fixed-depth arms were charged only up to their *selected* step rather than the full depth they
actually execute, understating their cost fourfold. Found and fixed before any conclusion was drawn.

**What these three have in common.** Each was killed by the simplest available control — a smaller
model, an exactly equivalent real model, a 2016 baseline. None needed a subtle argument.

---

## FAIL-037 / 038 / 039 — Nova: thirty-two architectures, three hypotheses, no principle

Nova began from a clean architectural slate with one objective: find a genuinely new principle of
neural computation. It searched thirty-two architectures across six mechanism families and returned
**no**. What makes the failure worth reading is how each hypothesis died.

**The instrument came first, and it disqualified two of its own tasks.** Eight algorithmic tasks
with known optimal procedures, evaluated at 1×, 2× and 4× the trained length. Before any candidate
was compared, a shortcut audit asked what a degenerate predictor could score: **position alone gets
0.887 on `cummax` and 0.598 on `sort` at length 64.** Both were dropped. Had they stayed, the
`causal_mlp` control's apparent 0.917 on sort would have looked like a discovery.

**The baselines then did something inconvenient.** The obvious answer to the measured gap — linear
attention, retention, selective state-space models, the family that is *supposed* to combine
recurrence and retrieval — turned out mediocre at both: 0.55 parity, 0.30 mod-sum, 0.58 needle. The
gap was real, and the established answer did not close it. That is what made the search worth
running at all.

**FAIL-037 — H-DILUTION.** Softmax attention is not length-invariant: adding non-matching keys takes
probability mass and shifts the read. A read that ignores non-matching candidates should therefore
extrapolate better. The operator-level property is **real and measurable** — read drift when 24
distractors are inserted is 0.236 for max/threshold against softmax's 0.724.

It made no difference. The confound control — ordinary softmax with the *same* post-read RMS
normalisation the unnormalised readers require — moves copy from 0.172 to 0.305, capturing
essentially the entire apparent gain. Max lands at 0.321, inside noise of the control. Needle is
flat across all four variants.

Two bugs were found before that verdict was reachable. The first `max` normaliser divided by the
sum at the end, which is *algebraically exactly softmax*; it produced numbers identical to the
control to three decimals, which is how it was caught, and the primary hypothesis had not been
tested at all up to that point. The second was the confound itself: normalisation applied only to
the candidate arms.

> **The lesson: an operator can have a property without the property mattering.** Those are separate
> questions and only a control separates them.

**FAIL-038 — H-INTERFERENCE.** An LSTM alone reaches 0.992 on mod-sum at 4× length. Add a parallel
attention branch and it collapses — identically for three different attention normalisers. A test-time
branch ablation showed why: turning attention *off* makes it **worse** (0.291 → 0.157), and turning
the recurrence off is equally bad. Neither branch works alone. The model found a joint solution that
needs both routes and does not extrapolate.

The frozen prediction said handicapping the shortcut route with dropout would let the recurrence be
learned. **All four clauses failed.** Dropout does not de-conflict anything; it slides the model
along a trade-off until it simply *is* an LSTM again — needle 0.841 → 0.260 against an LSTM-alone
0.283.

*And a validity threat surfaced here that touched everything.* The dramatic version of this effect —
mod-sum at 0.291 — was an artifact of an 800-step training budget. At 2400 steps it reads **0.776**.
Every headline number in Nova was re-measured. The prediction still fails as scored, but its
narrative was corrected rather than kept.

**FAIL-039 — H-COMPOSE.** If the competition is an artifact of *which two* routes were combined,
a model with all three — recurrence, location cursor, content attention — should approach the
per-task best everywhere. Mean 0.692 against a required 0.75, and reverse fell to **0.146**, which
is chance, against a per-task best of 0.371.

The third clause *passed*: adding attention relieved the state-tracking conflict exactly as
predicted, mod-sum 0.776 → 0.998 with needle at 0.977. And ordered memory died in the same change.

> **Capability competition is conserved. Relieving it between one pair reintroduces it elsewhere.**

**And the winner was a 2014 paper.** The single best-performing mechanism Nova produced, the
`cursor`, is Neural Turing Machine relative-shift location addressing — reproduced *more weakly*
than the original, on copy, the task the NTM paper introduced it for. The prior-art firewall caught
this before any novelty was claimed, which is the entire reason the firewall runs before the
comparison rather than after.

**What is left.** Two capabilities remain unsolved by every architecture tested: copy (best 0.470)
and reverse (best 0.371), against a chance level of 0.126. That gap is real, it is open, and Nova
did not close it.
