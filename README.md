# Q-Neuro

Q-Neuro is a computational research project testing whether ordered evidence can act on an
evolving diagnostic hypothesis state more effectively than conventional fixed-vector
classification. Its first controlled study compares an MLP with real- and complex-valued
low-rank evidence-operator models on a causal synthetic neurology environment.

This repository does **not** claim that clinical reasoning is quantum mechanical, that the
proposed mechanisms are novel, or that the software is clinically valid. Complex arithmetic is
included only where phase changes the computation, and every architectural claim is treated as a
falsifiable hypothesis.

> **RESEARCH PROTOTYPE — NOT FOR CLINICAL USE.** All initial cases are synthetic. Outputs are
> experimental measurements, not medical advice.

## Current milestone

- Foundational research questions and mathematical definitions
- A causal, structured `NeuroWorld` simulator with explicit missingness and ordered evidence
- Parameter-budgeted MLP, real operator-state, and complex operator-state models
- A reproducible, resource-bounded Experiment Zero runner
- Mathematical-invariant and generator-validity tests

## Current evidence

Corrected Experiment Zero (`QN-000003`, 14,000 training cases, three seeds) found mean top-1
accuracy of 0.752 for the unordered MLP, 0.995 for the tiny Transformer, 0.999 for the real
operator-state model, and 1.000 for the complex operator-state model. All ordered models solved the
held-out chronology counterfactual pairs; the MLP solved none because each pair has identical
aggregate evidence by construction.

The learning-curve study (`QN-000004`) is more informative than the saturated full-data result. At
250 cases the real operator model was strongest (0.716 mean top-1 versus 0.673 complex). At 1,000
cases complex reached 0.997 versus 0.985 real, but its mean ECE was 0.176 versus 0.028 and it trained
about 1.6 times longer. This is preliminary evidence for an operator-state inductive bias on one
synthetic generator—not evidence that complex arithmetic, Q-Neuro broadly, or a medical system is
superior.

![Experiment Zero learning curves](research/figures/generated/experiment_zero_learning_curves.png)

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run python -m experiments.run_experiment_zero \
  --config experiments/configs/experiment_zero.yaml
```

Run artifacts are written to a never-overwritten `experiments/results/QN-XXXXXX/` directory and
indexed in `experiments/registry.sqlite3`.

## Research documents

- [`docs/RESEARCH_QUESTIONS.md`](docs/RESEARCH_QUESTIONS.md)
- [`docs/PRIOR_ART.md`](docs/PRIOR_ART.md)
- [`docs/ARCHITECTURE_CANDIDATES.md`](docs/ARCHITECTURE_CANDIDATES.md)
- [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md)
- [`RESULTS.md`](RESULTS.md)
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
