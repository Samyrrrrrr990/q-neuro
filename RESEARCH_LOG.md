# Research Log

This log is append-only in intent. Superseded runs remain visible.

## 2026-08-09 — Foundation and hypothesis selection

**Question:** Is a complex hypothesis state evolved by non-commuting evidence operators measurably
better than conventional computation at similar parameter count?

**Hypothesis:** Explicit ordered operators will solve chronology twins that an aggregate MLP cannot;
complex relative phase may improve sample efficiency beyond real operators.

**Method:** Defined five architecture candidates. Selected low-rank real and complex residual
operators because they isolate non-commutativity with minimal compute. Deferred Hamiltonian,
density-matrix, attractor, adaptive-time, local-learning, and UI branches.

**Result:** Mathematical specification, preliminary prior-art ledger, NeuroWorld, four matched model
families, registry, tests, and experiment runners implemented on Python 3.12/PyTorch on the M2 Mac.

**Next:** Run the controlled full-data comparison.

## 2026-08-09 — QN-000001 pipeline smoke test

**Question:** Does the complete registry/train/evaluate/artifact path execute without numerical or
packaging failures?

**Method:** 800 training cases, one seed, two epochs. This profile was not intended for scientific
comparison.

**Result:** All four models completed; normalization, phase ablations, shuffling, counterfactuals,
and SQLite registration worked. Seven initial invariant tests passed.

**Next:** Commit the source state and run the preregistered setup.

## 2026-08-09 — QN-000002 preliminary full-data run

**Question:** Do ordered models learn the chronology-twin task?

**Method:** 14,000/3,000/3,000 split, three seeds, approximately 20k parameters.

**Result:** MLP 0.752 top-1 and 0.499 order accuracy; Transformer 0.995 top-1; real operator 1.000;
complex operator 0.999. Ordered models solved all explicit counterfactual pairs.

**Confounder discovered:** The MLP received age/sex covariates while sequence models did not. These
covariates were matched within twins and were unlikely to cause the order result, but the comparison
violated the same-evidence control.

**Decision:** Preserve and mark the run superseded; equalize inputs and replicate.

## 2026-08-09 — QN-000003 corrected replication

**Question:** Does the result survive equal demographic context for every model?

**Method:** Same setup and seeds as QN-000002; demographic context added to every initial state.

**Result:** Qualitative result replicated. Mean top-1 was 0.7524 MLP, 0.9947 Transformer, 0.9993
real operator, and 1.0000 complex operator. Every ordered model scored 1.0 on order twins and 500
counterfactual pairs. Complex phase ablations collapsed accuracy, but real operators achieved lower
NLL/ECE and trained about 40% faster than complex.

**Interpretation:** Non-commutative sequential computation matters in the designed task. The task
saturates, so it does not identify a complex advantage.

**Next:** Test nested training-set sizes.

## 2026-08-09 — QN-000004 sample-efficiency study

**Question:** Which model learns the chronology relation with fewer examples?

**Method:** Nested training sizes 250, 500, 1,000, 2,000, and 5,000; fixed 1,500 validation and
3,000 test cases; three seeds; up to 25 epochs with early stopping.

**Result:** Real operator led at 250 cases (0.7159 top-1; 0.308 counterfactual-pair accuracy). Complex
led at 500 cases (0.9493; 0.9933) and reached 0.9971 at 1,000 cases, versus 0.9846 real and 0.9070
Transformer. Complex NLL/ECE was worse than real at every size and runtime was roughly 1.5–1.7x.

**Interpretation:** Operator-state computation shows an interesting sample-efficiency signal on the
fixed simulator. Complex dynamics show a top-1/counterfactual crossover after 250 cases but not a
dominant Pareto improvement.

**Possible confounders:** Single generator seed, limited Transformer tuning, no GRU/state-space
baseline, no two-channel real equivalent, only three training seeds, saturated medium-data task.

**Next experiment:** Change the generator seed and causal templates; add a parameter-matched GRU
and an explicitly two-channel real operator; tune calibration/temperature without changing top-1.

## 2026-08-09 — QN-000006 generator-shift replication

**Question:** Does the medium-data complex result survive a stronger recurrent baseline, an
explicitly two-channel real control, and changed generator conditions?

**Method:** Nested sizes 250/500/1,000; three seeds; approximately 20k real scalars; learning rates
for GRU and Transformer chosen exclusively by in-domain validation NLL. Evaluated on the original
world, a nuisance-seed shift, and a noisy/sparse-evidence shift. The latter mixes probabilities
toward 0.5, reduces observation rate, increases timing noise, and independently hides chronology
markers.

**Result:** The tuned GRU is the strongest 250-case in-domain model (0.920 top-1) and disproves the
broad operator sample-efficiency interpretation. It collapses under generator shift. At 1,000 cases,
complex operator top-1 is 0.995 in-domain, 0.896 under nuisance shift, and 0.660 under noisy/sparse
shift. The two-channel real control obtains 0.986, 0.828, and 0.597. Complex solves 0.991 of shifted
counterfactual pairs versus 0.844 two-channel and 0.663 ordinary real.

**Interpretation:** The new evidence supports an exploratory robustness hypothesis rather than an
in-domain sample-efficiency headline. The complex-versus-two-channel paired difference is positive
under both shifts at 1,000 cases, but the study has only three seeds and project-designed shifts.

**Negative result:** A well-tuned conventional recurrent model can dominate all Q-Neuro variants
in low-data in-domain accuracy. Architecture comparisons without tuning that baseline were weak.

**Next experiment:** Replicate over a preregistered grid of at least five unseen world seeds and
multiple shift severities; add post-hoc temperature calibration fitted only on validation data; do
not add Hamiltonian or density dynamics yet.

## 2026-08-09 — QN-000008 multi-world robustness confirmation

**Question:** Does the complex robustness signal replicate across multiple unseen worlds and shift
severities when world seed is treated as the statistical unit?

**Method:** Trained all six controls at 1,000 cases with three training seeds. Evaluated five unseen
world seeds under nuisance, mild, moderate, and severe shifts. Each world has 2,000 cases and 300
counterfactual pairs. Fitted one temperature using only in-domain validation logits.

**Result:** Complex top-1 is 0.909 nuisance, 0.806 mild, 0.645 moderate, and 0.468 severe. The
two-channel control reaches 0.846, 0.745, 0.585, and 0.414. Complex-minus-two-channel paired effects
are +0.054 to +0.063 with all five-world 95% intervals above zero. Complex also retains 0.987
counterfactual-pair accuracy at moderate shift.

**Negative result:** In-domain temperature scaling does not transfer. It worsens moderate-shift ECE
for every model and catastrophically worsens complex NLL in one training seed.

**Interpretation:** A robustness phenomenon is now replicated within the NeuroWorld simulator
family. The result justifies expanding to orthogonal task constructions, not escalating to clinical
or novelty claims.

**Next:** Build composition, ambiguity, OOD disease, and hidden-syndrome task suites with explicit
task-specific controls. Keep Hamiltonian/density and novel optimizers behind those gates.

## 2026-08-09 — QN-000010 orthogonal NeuroWorld task suite

**Question:** Does the replicated shift advantage extend to composition, irreducible ambiguity,
unknown diseases, and an unlabeled hidden syndrome?

**Method:** Trained all six models in three independent contexts across seeds 11/23/37: a 3,000-
case ordinary dataset; a split excluding declared evidence conjunctions from training; and a split
excluding diagnosis label 19 entirely. Evaluated 300 observationally identical ambiguity pairs and
400 cases from a separately generated, unlabeled syndrome. Measured output uncertainty and
nearest-centroid representation distance.

**Result:** Operator models saturate held-out composition (0.995–1.000), making that test
non-discriminative. Complex rejects the unknown disease with 0.9988 maximum-softmax AUROC and
separates the hidden syndrome with 0.9990 representation AUROC. The two-channel output-space
unknown AUROC is nearly identical at 0.9974, while its representation metrics are unstable across
seeds. Complex performs poorly on irreducible ambiguity: pair NLL 2.581 versus 1.148 for the real
operator, and it assigns only 0.212 total probability mass to the two valid twin labels.

**Interpretation:** Complex representation geometry is a promising anomaly signal, but unknown-
disease rejection does not yet require complex arithmetic. More importantly, hypothesis-state
dynamics do not automatically preserve multiple plausible diagnoses; the present complex
measurement can be less calibrated than a real operator when evidence is fundamentally
insufficient.

**Statistical limitation:** Three paired seeds permit a minimum two-sided exact sign-flip p-value
of 0.25. All task-suite effects remain exploratory.

**Next:** Add active evidence acquisition and explicit trajectory diagnostics. Then test whether
adaptive dynamics or uncertainty-aware objectives repair ambiguity without sacrificing robustness.

## 2026-08-09 — QN-000012 active evidence acquisition

**Question:** Can hypothesis-state models reach a correct differential with fewer findings, and
does model-conditioned expected information beat random or fixed acquisition?

**Method:** Restricted the benchmark to factorial labels 8–19 so evidence order is not an outcome.
Trained all models on 3,000 partially observed cases over three seeds. For 200 fully specified test
cases, sequentially revealed 12 of 40 outcomes using random order, a global mutual-information
order, or counterfactual expected-entropy minimization. Recorded predictive curves and runtime.

**Result:** Full-information top-1 is 0.950–0.997 and poorly predicts active performance. Complex
expected-information AUC is 0.590 and final 12-query accuracy is 0.833. MLP is 0.585/0.803 and
two-channel is 0.568/0.807, so complex does not separate from those controls. Expected information
improves complex AUC by 0.073 over fixed ordering but lowers Transformer AUC by 0.169. GRU has
0.950 full-case top-1 yet only 0.282 expected-information AUC.

**Interpretation:** Evidence acquisition is a distinct computational law. Model uncertainty can
guide acquisition only when its partial-evidence geometry is trustworthy. Complex is promising
but not uniquely efficient; the MLP result rules out a broad hypothesis-state claim.

**Confounders:** Binary findings have equal query cost; conditional outcome probabilities use a
naive diagnosis-weighted approximation; policies do not model real clinical questions; only 12
synthetic diagnoses and three seeds are used.

**Next:** Expose trajectories and implement adaptive diagnostic time, energy/attractor, and
Hamiltonian/dissipative variants. Use active AUC and ambiguity NLL as gates rather than optimizing
saturated full-case accuracy.

## 2026-08-09 — QN-000014 computational-law mechanism suite

**Question:** Do attractor, adaptive-time, Hamiltonian, dissipative, or density laws improve the
robustness/ambiguity frontier beyond conventional and operator controls?

**Method:** Implemented and tested 18 laws at roughly 20k real scalars. Trained on 1,000 cases over
three seeds. Evaluated in-domain performance, 300 ambiguous twins, 200 chronology pairs, and three
unseen moderately shifted worlds. Shift inference treats world seed as the unit.

**Result:** Complex operator remains strongest under shift at 0.647. Hamiltonian reaches 0.556 and
hybrid 0.550, versus dissipative-only 0.438. Hybrid-minus-dissipative is +0.112 across worlds, while
hybrid-minus-Hamiltonian is −0.006. D3 reaches 0.453 shift top-1 and 1.468 ambiguity NLL, preserves
density invariants, and has 0.620 mean off-diagonal coherence. Adaptive attractor has 5.25/8 soft
expected steps but no clear performance gain over fixed attractor.

**Interpretation:** Coherent low-rank Hermitian rotation is useful relative to damping alone, but
dissipation adds nothing to the coherent model. Complex operator dynamics still lead the tested
Hamiltonian approximation. Density structure creates a legitimate inspectable state without yet
creating predictive benefit. Adaptive depth is not validated until it produces actual compute
savings at matched performance.

**Negative results:** Fixed factor-graph message passing underfits; energy attractors ignore order;
dissipation alone loses chronology; hybrid dynamics do not beat pure Hamiltonian; D3 does not beat
real operators; complex and Transformer remain badly calibrated on irreducible ambiguity.

**Next:** Run targeted ablations for phase, ordering, interference, damping, density rank, and hard
adaptive halting. Then evaluate unconventional learning rules against AdamW.

## 2026-08-09 — QN-000016 critical mechanism ablations

**Question:** Which parts of the complex operator robustness signal survive retrained ablations?

**Method:** Compared a commutative complex accumulator, two-channel real operator, phase-insensitive
magnitude readout, removal of negative evidence, and the full complex operator. Repeated
attractor/Hamiltonian controls and trained D3 at ranks 1/2/4. Used the same three seeds and unseen
worlds as QN-000014.

**Result:** Full complex shifted top-1 is 0.647 versus 0.415 commutative, 0.499 two-channel, 0.543
magnitude-only readout, and 0.575 without negative evidence. World-paired effects are +0.232,
+0.148, +0.104, and +0.072 respectively, with all descriptive intervals above zero. Density ranks
1/2/4 reach 0.449/0.453/0.441; higher rank does not help. Hamiltonian again exceeds dissipative by
0.119, while hybrid does not exceed Hamiltonian.

**Interpretation:** The robustness phenomenon depends jointly on ordered state-conditioned
composition, phase-sensitive measurement, and negative evidence. No single removed mechanism
accounts for the full effect. Higher density rank is not beneficial without an objective that uses
relational state.

**Caveats:** Accumulator versus operator changes more than commutator structure; the magnitude-only
readout also changes optimization geometry; density ranks are not parameter-matched; only three
worlds and one simulator family are used.

**Next:** Implement observable probes, hard halting, and targeted contradiction/revival tests before
moving to local and nonstandard learning laws.

## 2026-08-09 — QN-000019 emergent hierarchical observable probes

**Question:** Do diagnosis-trained latent states expose mechanism, localization, temporality, and
context, and are Hermitian quadratic observables useful on complex states?

**Method:** Froze all 54 QN-000014 checkpoints (18 architectures × three seeds), extracted final
states on factorial-label test cases, and selected linear-probe regularization using a training-only
validation split. For complex operator, Hamiltonian, and hybrid states, also fitted constrained
Hermitian observables `z†Az`. No simulator-factor label entered end-to-end diagnostic training.

**Result:** Complex linear-probe accuracy is 0.932/0.933/0.907/0.918 across the four factors, but
GRU reaches 0.969/0.954/0.964/0.981 and state-space reaches 0.948/0.922/0.908/0.992. Hermitian
observables improve complex-state accuracy most for temporality (+0.039) and context (+0.023), but
usually worsen NLL. Mean probe accuracy correlates only descriptively with moderate-shift top-1
across models (`r=+0.45`).

**Interpretation:** Hierarchical simulator information is accessible from Q-Neuro states, yet it is
neither unique to complex arithmetic nor sufficient for robustness. Quadratic Hermitian readouts
can expose additional structure, but need calibration-aware training before they qualify as useful
diagnostic observables.

**Caveats:** Probe labels come from the same simulator; models are non-independent; three seeds
limit inference; probe accessibility does not show that the diagnosis head uses the factor; final
states may encode nuisance correlations.

**Next:** Measure realized hard halting and state trajectories, then compare global, local,
phase-coded, and zero-backprop learning laws.

## 2026-08-10 — QN-000020 training-law smoke profile

**Purpose:** Integration-only validation of nine training paths, checkpoint persistence, resource
counters, ambiguity/shift evaluation, and registry writes.

**Result:** Every method completed. Local plasticity and ZeroBackprop recorded zero backward and
zero autograd-gradient calls. Two training epochs are intentionally inadequate for global methods;
the metrics are not used scientifically.

## 2026-08-10 — QN-000021 Experiment Six training-law suite

**Question:** Can local, hybrid, phase-coded, or no-backprop learning preserve diagnosis while
changing compute or transfer behavior?

**Method:** Held architecture fixed to the 20,304-scalar complex operator. Compared AdamW, SGD,
equal-effective-batch gradient accumulation, multi-objective AdamW, PCGrad, PGO, local complex
delta/Hebbian updates, local→global training, and a frozen-random centroid ZeroBackprop prototype.
Evaluated 250/1,000 cases, three seeds, three unseen worlds, chronology twins, and ambiguity.

**Result:** At 1,000 cases, AdamW shifted top-1 is 0.620; multi-objective AdamW/PCGrad/PGO reach
0.635/0.635/0.633. PGO costs 5.43 s versus 2.96 s for multi-objective AdamW. Its gradient cosines
are weakly positive, and it does not exceed the same-objective control. Local/ZeroBackprop reach
0.642/0.133 in-domain and 0.137/0.139 shifted. The hybrid reaches 0.998 in-domain and ambiguity NLL
1.317 but only 0.419 shifted top-1, 0.201 below AdamW across worlds.

**Interpretation:** Auxiliary labels modestly help robustness; phase rotation and conflict
projection do not. Pure local learning acquires source structure without useful scaling. Local
pretraining creates a reproducible source-specialization basin that global fine-tuning retains—a
negative result relevant to two-timescale learning.

**Caveats:** Fixed method-specific learning rates were not exhaustively tuned; auxiliary methods
receive extra labels; local rules are prototypes rather than an exhaustive local-learning family;
RSS is too coarse for memory claims; only one architecture and simulator family are tested.

**Next:** Add realized hard halting and trajectory-level state visualizations, then build automated
mechanism search and surprise detection around the now-complete experiment matrix.

## 2026-08-10 — QN-000022 hard-halting smoke profile

**Purpose:** Verify checkpoint reconstruction, validation-only threshold selection, active-index
execution, step histograms, paired evaluation, latency timing, and registry persistence.

**Result:** All three source checkpoints completed and reduced mean execution to about 3.2/8 states
without an obvious source-accuracy loss. The sparse candidate grid is integration-only.

## 2026-08-10 — QN-000023 realized hard velocity halting

**Question:** Can soft expected diagnostic depth be converted into actual state-update and latency
savings without degrading diagnosis?

**Method:** Applied active-index velocity stopping to the three frozen QN-000014 adaptive-attractor
checkpoints. Selected thresholds from 12 velocity quantiles on source validation, constrained
against the eight-state final readout. Evaluated source, three unseen worlds, chronology pairs,
state counts, and 15-repeat CPU latency.

**Result:** Every selected threshold halts every case at two states. Source/shift top-1 is
0.722/0.432 versus 0.724/0.431 for the soft mixture. Latency falls from 0.0171 to 0.0035 ms/case,
and shifted NLL/ECE fall from 2.518/0.330 to 2.127/0.172. Hard inference uses 20.2% of soft-path
latency across seeds. Chronology-pair accuracy remains zero.

**Interpretation:** The compute saving is real, but it is fixed truncation rather than adaptive
reasoning. Later attractor steps are actively harmful to calibration and add no accuracy. The
appropriate architecture change is a shallower attractor, not a more elaborate halting network.

**Caveats:** Millisecond timings are CPU- and batch-specific; thresholds were selected on the source
world; the attractor model is weaker than the complex operator; all active cases happen to stop at
the same boundary; latency does not include data loading.

**Next:** Render evidence-level complex trajectories and counterfactual paths, then turn the
experiment space into a configuration-search and surprise-detection system.

## 2026-08-10 — QN-000024 failed trajectory extraction

**Failure:** The first registered trajectory run called `torch.flatnonzero`, which does not exist in
the supported PyTorch API. The registry marks the run failed and its configuration/environment are
preserved. Replaced it with `torch.nonzero(...).flatten()` and reran under a new experiment ID.

**Scientific impact:** None; failure occurred before any completed seed or metrics artifact.

## 2026-08-10 — QN-000025 complex-state trajectory study

**Question:** What does the trained complex state actually do as ordered positive and negative
evidence arrives?

**Method:** Replayed 1,000 cases and 200 chronology pairs through all three frozen QN-000014
complex checkpoints. Logged state path length, final velocity, entropy, per-token true-label
probability changes, operational negative-evidence drops/revivals, and counterfactual state
distance. Selected visual examples by deterministic generation order, not outcome.

**Result:** Entropy falls by 1.266 nats while normalized path length is 2.956 and final velocity is
0.175. Positive/observed-negative tokens change true-label probability by +0.0467/+0.0109 on
average. A >0.05 drop after negative evidence occurs in 5.0% of cases, with 75.9% later recovery.
Chronology pairs are all correct and finish 0.841 normalized state distance apart.

**Interpretation:** The state exposes real, reversible evidence dynamics and order-dependent
bifurcation. Observed-negative evidence commonly helps by excluding alternatives; the rare drop
and recovery pattern is measurable but not a semantic contradiction label. Final states remain
dynamic rather than demonstrably converged attractors.

**Caveats:** Synthetic evidence order and labels; no human interpretability study; probability
changes are context-dependent and not causal token effects; one deterministic example is a visual
illustration, while aggregate metrics use all cases.

**Next:** Build configuration-level computational-law search, Pareto ranking, and anomaly flags,
then consolidate the complete evidence into the dashboard and paper.
