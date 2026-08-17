# Mechanism discovery protocol

The mechanism suite asks causal questions about the constrained complex parameterization. It is a
discovery experiment and cannot validate the primary robustness claim.

## Interventions

- **Forced linear commutation:** token operators are diagonal in one shared complex coordinate
  basis. Their linear commutators are exactly zero; nonlinear injections and normalization remain.
- **Commutator penalty:** the four declared chronology-marker commutators receive a normalized
  positive penalty.
- **Noncommutative real dynamics:** an unrestricted dense real transition receives a bounded
  incentive for marker noncommutation.
- **Phase destroyed in training:** every post-update state is projected to zero phase.
- **Phase destroyed in inference:** the conventionally trained complex model is evaluated with
  zero or randomized phase at measurement.
- **Magnitude destroyed:** every component is projected to unit magnitude after each update.
- **Conjugation removed:** both transition projections and learned measurement omit conjugation.
- **Fixed random/frozen dynamics:** dynamics remain at initialization and only the readout learns.
  These two preregistered names intentionally share one implementation and are reported as aliases.
- **Frozen readout:** dynamics learn while the random readout is fixed.
- **Mechanism stealing:** explicit noncommutative real, real polar, and real rotation controls test
  whether any useful structure is uniquely complex.
- **Ambiguity preservation:** real and complex positive-evidence heads and a two-state mixture test
  the accuracy/calibration tension under observationally identical targets.
- **Negative controls:** magnitude-only readout, no-negative-evidence dynamics, exact real block,
  and a diagonal state-space model remain visible.

## Measurements

The suite records performance and calibration across independent worlds, controlled order,
MNAR-like missingness, unseen factor combinations, contradiction, and irreducible ambiguity. It
also records marker commutator norms, state magnitude/phase summaries, ambiguity-pair metrics,
parameter counts, trainable counts, and optimization cost.

Any relationship selected here must be formalized before testing on untouched task generators.
Because all current data are synthetic, no mechanism result can support a clinical or biological
claim.
