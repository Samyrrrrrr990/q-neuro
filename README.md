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

The stronger-control replication (`QN-000006`) changes the interpretation of that learning curve.
A validation-tuned GRU is the most sample-efficient in-domain model at 250 cases (0.920 top-1), so
operators do not hold a general in-domain sample-efficiency lead. The GRU collapses under changed
generator seeds, while the complex operator reaches 0.896 on the nuisance shift and 0.660 on the
noisy/sparse shift at 1,000 cases. The matched two-channel real control reaches 0.828 and 0.597.
This is preliminary evidence of a robustness difference on declared synthetic shifts—not evidence
that complex arithmetic, Q-Neuro broadly, or a medical system is superior.

The confirmatory sweep (`QN-000008`) evaluates five preregistered unseen world seeds at four shift
severities and uses world seed—not individual cases—as the statistical unit. At 1,000 training
cases, complex top-1 is 0.909/0.806/0.645/0.468 from nuisance through severe shift. The two-channel
real control reaches 0.846/0.745/0.585/0.414. The complex-minus-two-channel world-level effect is
positive at every severity, with 95% intervals excluding zero. This confirms a simulator robustness
phenomenon, while in-domain temperature calibration fails to transfer and often worsens shifted
calibration.

![Experiment Zero learning curves](research/figures/generated/experiment_zero_learning_curves.png)

![Generator-shift replication](research/figures/generated/generator_shift_replication.png)

![Multi-world robustness sweep](research/figures/generated/robustness_world_sweep.png)

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
