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

