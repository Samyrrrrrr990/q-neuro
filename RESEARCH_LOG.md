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
