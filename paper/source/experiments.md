# Experiments

## Registered sequence

The registry contains 26 experiment records, 25 completed and one failed run preserved for audit. Each record stores an immutable identifier, configuration, result directory, timestamps, environment information, and status. Follow-up experiments are introduced when a previous result leaves a specific alternative explanation. The sequence progresses from task sanity checks to corrected baselines, stronger controls, unseen-world confirmation, orthogonal tasks, active acquisition, computational laws, critical ablations, observables, training laws, hard execution, trajectories, and automated discovery.

The primary confirmatory-style comparisons within this exploratory program are predefined paired effects. Training seeds are shared across models; world seeds are shared across architectures; counterfactual pairs share every variable except the intervention. The broad discovery scan is explicitly hypothesis-generating. It searches stored architecture, training, and halting summaries against declared surprise rules and Pareto objectives, then emits proposals rather than silently converting selected results into confirmation.

{{figure:experiment_evidence_map|Registered studies are ordered so that baseline validity and strong controls precede mechanism expansion. The failed QN-000024 run remains in the registry rather than being erased.|Timeline of registered experiments and their supporting, mixed, or refuting evidence status.}}

## Experiment Zero

The corrected full-data comparison trains on 14,000 cases, validates on 3,000, and tests on 3,000, with 500 independent counterfactual chronology pairs and three seeds. Four approximately 20,000-parameter models receive identical evidence and demographics. The MLP is unordered; the Transformer and operator models receive the observed sequence. This experiment establishes task correctness and phase functionality, but saturation prevents comparative conclusions.

The sample-efficiency study uses nested training sets of 250, 500, 1,000, 2,000, and 5,000 cases with shared validation/test data. A stronger-control replication adds validation-tuned GRU and two-channel real models at 250–1,000 cases, then evaluates nuisance-seed and noisy/sparse shifts. A five-world sweep at fixed 1,000 cases tests whether the apparent complex-versus-two-channel effect persists across held-out world seeds and severities.

## Orthogonal tasks and acquisition

Separate models are trained for standard/ambiguity evaluation, held-out composition, and omitted-disease evaluation. The hidden syndrome is generated independently and never used for model selection. The active-acquisition study trains on 3,000 factorial cases and evaluates 200 cases per seed across budgets one through twelve. Policy runtime is measured because expected-information queries require multiple counterfactual forward passes.

## Computational-law and ablation suites

The mechanism suite trains eighteen architectures on the same 1,000-case split with three seeds. Evaluation covers source accuracy, ambiguity, chronology pairs, and three moderate unseen worlds. The critical ablation suite then intervenes on the leading mechanisms: commutativity, phase-sensitive readout, negative evidence, Hamiltonian versus dissipative terms, and density rank. Paired effects are reported relative to the full complex operator or the relevant nested control.

## Probe, training-law, and trajectory studies

The observable study freezes trained representations and selects linear-probe regularization using validation data. It evaluates mechanism, localization, temporality, and context. For complex states, Hermitian quadratic probes are compared with linear probes. Diagnostic models are never fine-tuned by probe losses.

The training-law study evaluates nine optimizers or update laws at 250 and 1,000 cases. It records diagnosis and auxiliary metrics, gradient cosines, reverse-mode gradient-call counts, deployment parameters, time, and memory. The hard-halting study measures actual active-index computation rather than soft depth. The trajectory study stores per-step states for held-out cases, computes predeclared summaries, and evaluates chronology twins under matched steps.

## Metrics and multiplicity

Accuracy is never the sole endpoint. Every principal comparison includes NLL and ECE; chronology studies include pair accuracy; anomaly studies include AUROC and FPR at 95% true-positive rate; efficiency studies include parameters, CPU time, memory, executed states, and latency. The discovery engine uses six objectives when marking architecture Pareto status. Because the study explores many laws and metrics, isolated favorable values are not treated as confirmatory. The claim ledger records counterevidence, and the paper emphasizes effects that repeat across worlds or survive targeted ablation.

## Reproducibility

All experiments use deterministic configuration files and explicit seeds. Raw registered JSON artifacts are committed; analysis scripts regenerate paired effects; figure scripts read artifacts and emit PNG plus vector PDF; table scripts read the same artifacts; the manuscript builder emits synchronized modular LaTeX, Word, and PDF. Cached results permit paper regeneration without retraining, while individual Make targets expose the full training commands.

The test suite checks simulator invariants, models, learning rules, observables, evaluation, active acquisition, registry behavior, discovery logic, and task construction. Continuous integration installs from the locked environment, runs lint and tests, rebuilds dashboard data, and rejects a dirty dashboard artifact. A complete reproduction on different hardware should expect timing variation but not a change in the registered data or figure values unless experiments are rerun intentionally.

## Hardware and resource scope

The reported studies were designed for a consumer MacBook CPU and small memory footprint. Peak process RSS remained below approximately 0.36 GiB in the principal runs, although allocator reuse makes small cross-model deltas approximate. This resource constraint shaped the benchmark scale and the use of low-rank operators. It is a reproducibility feature, not evidence that the method is globally compute-optimal.
