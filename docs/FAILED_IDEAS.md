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
