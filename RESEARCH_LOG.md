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
