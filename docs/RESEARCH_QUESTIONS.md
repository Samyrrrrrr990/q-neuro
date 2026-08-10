# Research Questions

Status: pre-experimental. These are hypotheses, not findings.

## Primary question: Experiment Zero

Does evolving a hypothesis state through ordered, non-commuting evidence operators provide a
measurable benefit over an equivalently budgeted conventional classifier?

The initial claim is falsified if a tuned MLP and sequence baseline match or outperform the
operator models in predictive performance, counterfactual response, calibration, and efficiency,
with no compensating advantage in interpretable state dynamics.

### Confirmatory outcomes

1. Top-1 accuracy and top-3 recall on a held-out generator split.
2. Negative log-likelihood and expected calibration error (ECE).
3. Counterfactual flip accuracy on paired cases differing in one causal variable.
4. Order-pair accuracy on cases where `A -> B` and `B -> A` have different targets.
5. Wall time, peak resident memory, and learned scalar parameter count.

### Required controls

- MLP using the same observed evidence without order.
- Real-valued low-rank operator-state model.
- Complex-valued operator-state model in which phase affects both evolution and measurement.
- Shuffled-order evaluation without retraining.
- Randomized-phase and phase-disabled ablations.
- At least three fixed seeds for any reported headline comparison.

## Secondary questions

### RQ2 — Non-commutativity

Do learned commutator norms correlate with performance on order-dependent cases? A positive result
requires more than order sensitivity: the model must respond in the causally correct direction.

### RQ3 — Complex phase

Does complex phase improve sample efficiency, calibrated ambiguity, or counterfactual behavior
after matching real scalar parameter counts? Accuracy alone is insufficient to identify the
mechanism; randomized-phase and equivalent two-channel real controls are necessary.

### RQ4 — Adaptive diagnostic time

Can a state-velocity halting rule reduce mean operator applications without degrading performance,
and does it allocate more steps to ambiguous cases? This is deferred until fixed-step dynamics are
stable.

### RQ5 — Attractor and density dynamics

Do explicit energy minima or density-matrix off-diagonal terms predict later resolution better
than ordinary hidden states? These mechanisms are too expensive and underidentified for the first
experiment and require their own matched controls.

## Leakage and simulator checks

- Split by generated case, not by evidence permutation.
- Do not expose target disease IDs in evidence tokens or padding patterns.
- Hold prevalence fixed across compared models.
- Audit whether sequence length, missingness, or token position alone predicts labels.
- Repeat with changed generator parameters before treating a gain as architectural evidence.
- Distinguish simulator-specific success from evidence about general clinical reasoning.

## Earliest decision gate

Advance the complex architecture only if it either (a) replicably improves a preregistered metric
at similar compute/parameters, or (b) exposes a measurable state phenomenon with predictive value.
Otherwise retain the simpler real operator model or conventional baseline and record the negative
result.

