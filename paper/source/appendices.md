# Extended methods and supplementary results

## A. Repository and registry design

The repository separates simulator code, model implementations, experiment runners, configurations, registered artifacts, paired analyses, figure scripts, dashboard assets, paper sources, and tests. Every runner writes to a unique QN identifier. The SQLite registry is an index; the JSON artifact is the portable record. A failed QN-000024 run is retained with its status so that sequential numbering and debugging history remain auditable.

Configurations are YAML and include generator parameters, split sizes, seeds, model hyperparameters, optimizer settings, and evaluation worlds. Artifacts contain aggregate summaries and per-run metrics with sample standard deviation and Student-t intervals. Analysis files record paired contrasts separately from headline metrics. The discovery stage reads those artifacts and cannot modify experimental results.

## B. Reproduction commands

The locked Python environment is installed with `make sync`. `make test` runs the test suite and `make lint` checks repository formatting. Individual Make targets rerun Experiment Zero, sample efficiency, generator shift, robustness, task suite, active evidence, dynamics, ablations, observables, training laws, hard halting, trajectories, and discovery. `make figures` regenerates every PNG and vector PDF from result files. `make paper` rebuilds tables, modular LaTeX, DOCX, and the PDF; `make reproduce-paper` verifies cached artifacts and performs the complete publication build.

Full retraining is intentionally not part of the default paper target because it is slower and can produce hardware-dependent timing. Cached registered results are source artifacts, not manually transcribed numbers. Researchers who rerun training should use a new experiment identifier or preserve the existing directories before comparison.

## C. Full-data saturation

The full-data study is useful primarily as a leakage and order sanity check. All ordered models solve the chronology pairs. Differences among 0.995, 0.999, and 1.000 source top-1 are not a sound basis for an architectural claim. The unordered MLP’s zero pair accuracy validates the construction.

The phase-scramble result is similarly bounded. A fitted complex network can distribute information across phase and magnitude, so perturbing phase should hurt. Comparative evidence requires retrained controls, which are provided by the two-channel and magnitude-only ablations.

## D. Strong-control generator shift

At 1,000 cases under the noisy/sparse shift, both chronology markers are observed in 49.1% of twin cases. The complex model scores 0.998 when the marker evidence is complete and 0.380 when incomplete. Counterfactual pairs expose both markers and vary only order; complex solves 0.991 of pairs, compared with 0.844 for two-channel real. The model cannot infer absent causal evidence reliably, even when its ordered rule is strong.

Generator shifts are constructed without model-specific optimization, but they share causal code. The five-world hierarchy is used to avoid treating thousands of cases from one generator instance as independent evidence. A true external replication requires new causal code.

## E. Active acquisition details

Expected information gain evaluates positive and negative counterfactual outcomes for every unqueried finding, weights them by a diagnosis-conditioned outcome model, and selects the lowest expected posterior entropy. This adds substantial forward-pass cost: for the complex model, evaluating 200 cases takes approximately 5.43 s, compared with 1.20 s for fixed order. Query costs are otherwise assumed equal.

The fixed policy ranks findings by mutual information computed on training data. It is competitive for several models and much better than expected information for the Transformer. This demonstrates that a simpler policy can be safer when model probabilities under partial evidence are misspecified.

## F. Unconventional dynamics

The energy and adaptive attractors use averaged evidence force and therefore cannot represent chronology. Their failure on pair accuracy is a specification consequence rather than an optimization surprise. The adaptive model’s soft expected depth is logged but never converted into a compute claim.

The Hamiltonian-style operator uses a real diagonal plus complex low-rank Hermitian action. Pure and hybrid variants share injection and readout. The discrete update is normalized, so conservation is approximate. The dissipative-only failure suggests that elimination needs structured, hypothesis-dependent damping.

Density dynamics factorizes the diagnosis-space state. Numerical invariant tests pass across ranks. Rank two is selected for the main suite before the rank ablation; the rank sweep provides no evidence for added relational capacity.

## G. Training-law diagnostics

At 1,000 cases, diagnosis-only AdamW obtains 0.971 source top-1 and 0.620 shift top-1 in the training-law configuration. Multi-objective AdamW and PCGrad reach 0.978 source and 0.635 shift. Phase Gradient Optimization reaches 0.978 and 0.633 while taking 5.43 s versus 2.96 s for AdamW. Weakly positive task-gradient cosines explain why conflict-specific rotation has little opportunity.

Transition-local plasticity makes zero backward calls and trains in 1.25 s, reaching 0.642 source and 0.137 shift top-1. ZeroBackprop reaches 0.133 and 0.139. The local rule learns something, but not a robust diagnostic state. Hybrid local-plus-global training reaches 0.998 source and 0.419 shift, consistent with source locking.

## H. Hard-halting threshold selection

Candidate velocity thresholds are evaluated on source validation under predefined accuracy and NLL tolerances. The selected threshold causes every case to exit at the minimum step two. Compared with soft eight-state evaluation, this reduces state computations by 75% and measured latency by roughly 80%. Because the distribution is degenerate, difficulty-conditioned execution is not established.

Later attractor steps reduce confidence quality without improving accuracy. The correct baseline for future work is therefore fixed depth two, not the original eight-step or soft ACT model.

## I. Statistical cautions

Student-t intervals are reported because seed and world counts are small and variance is estimated. They are not corrected for the large number of explored comparisons. Exact sign-flip tests cannot resolve conventional significance thresholds with only three paired seeds. The analysis prioritizes repeated direction across unseen worlds and targeted ablations.

Case-level bootstrap intervals would be misleading for generator robustness because cases share one world. The hierarchy averages training seeds within world and uses world means as observations. A future study should preregister at least twenty independently generated worlds or use a hierarchical model with enough top-level units.

## J. Failure inventory

The repository’s failure ledger records why each plausible idea was tested, what happened, why it likely failed, and whether it is worth revisiting. It includes full-data saturation, asymmetric inputs, GRU source dominance, calibration transport, easy composition, ambiguity collapse, model-conditioned acquisition failure, generic dissipation, soft-depth accounting, graph oversmoothing, density-rank non-benefit, non-unique probes, phase-gradient non-benefit, local source locking, and hard-halting degeneracy.

Preserving these outcomes changes the research program. It redirects work from adding architectural ornament toward independent generators, ambiguity-aware objectives, real-equivalence controls, and causal state interventions.

## K. Artifact integrity

Figures read the registered metrics and emit PNG plus vector PDF. Tables read the same JSON and emit machine-readable JSON plus LaTeX. The Word and PDF builders consume those outputs. The dashboard uses a generated JavaScript data bundle. Continuous integration rebuilds the dashboard and checks for an unexpected diff.

The paper source deliberately avoids external patient data, secrets, checkpoints, and large binaries. The Git history separates experiments, analyses, dashboard, literature audit, figures, and manuscript. A release artifact can therefore be inspected at the level of both claim and commit.

## L. AI-assistance disclosure

An OpenAI Codex agent assisted with code implementation, experiment orchestration, documentation, figure and document generation, prior-art retrieval, and manuscript drafting. The agent was instructed to preserve negative results and refuse unsupported claims. The human author should independently verify every scientific statement before submission and obtain any journal-specific disclosure language required at the time of submission.

## M. Checklist for external replication

- Freeze the primary and co-primary endpoints before model training.
- Implement a separately authored causal generator with new factor and observation mechanics.
- Include tuned GRU, state-space, real operator, two-channel, block-real-equivalent, and complex controls.
- Use enough independent world seeds for stable top-level uncertainty.
- Report ambiguity NLL, valid-set mass, calibration, and compute alongside accuracy.
- Select hyperparameters without shifted test labels.
- Preserve failed and superseded runs in a public registry.
- Release code, configurations, raw summaries, and generated artifacts.
- Prohibit clinical interpretation unless a separate governed validation program succeeds.
