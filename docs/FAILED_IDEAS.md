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
