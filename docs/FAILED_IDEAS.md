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

