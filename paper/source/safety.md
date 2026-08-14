# Safety and responsible use

## Intended use

Q-Neuro is intended for research on sequential representations, causal synthetic benchmarks, uncertainty, and reproducibility. NeuroWorld may be used to test whether an architecture respects chronology, explicit missingness, counterfactual invariance, and generator shift. The code may also serve as a compact teaching example for complex-valued recurrence and falsification-led model development.

## Prohibited clinical use

The model must not be used to diagnose, triage, recommend treatment, rank real patients, replace professional judgment, or generate medical advice. It has never seen patient data. Its output labels do not correspond to validated disease definitions, and its probability calibration fails under synthetic shift. Integrating it into a clinical workflow would create foreseeable risks of automation bias, false reassurance, delayed care, and inequitable error.

The repository’s model card, README, dashboard, manuscript, and release notes repeat this boundary. Example interfaces use synthetic terminology and do not invite entry of personal health information. No pretrained checkpoint is presented as a medical product.

## Data governance path

Any future real-data evaluation would require a separate protocol: institutional oversight where applicable, lawful data access, privacy review, data minimization, security controls, subgroup analysis, clinician-defined endpoints, prospective validation, and a plan for incident reporting. De-identification alone would not resolve governance or representativeness. Real data should not be copied into this public repository.

Model development and evaluation sets would need institutional and temporal separation. Hyperparameters and thresholds should be locked before external testing. Missingness mechanisms must be analyzed because clinical absence can reflect access, documentation, or workflow rather than biological absence. Chronology extracted from records may be timestamped by documentation rather than event onset.

## Uncertainty and abstention

A safe diagnostic research system must know when its evidence is insufficient. The ambiguity experiment shows that the current complex readout does not. OOD AUROC does not solve this problem: a model can separate an omitted syndrome while assigning too little mass to two valid known hypotheses. Future development should prioritize set-valued supervision, coverage guarantees, and explicit abstention over small gains in top-1.

Calibration must be evaluated under the deployment distribution. The source-temperature failure demonstrates that a standard post hoc correction can worsen shifted probabilities. No confidence value from the current model should be interpreted as clinical risk.

## Dual-use and communication risk

The primary misuse risk is exaggerated communication rather than autonomous capability. Terms such as “quantum,” “medical AI,” and “diagnostic reasoning” can produce unwarranted authority. Repository language therefore avoids “revolutionary,” “most powerful,” “Nobel-level,” and “clinically accurate.” The central positive claim always includes the qualifiers synthetic, tested, and within NeuroWorld.

The code is released under a permissive license to support scrutiny. Security issues should be reported privately according to SECURITY.md. Scientific disagreements and replication failures should be filed publicly when they do not expose sensitive information. Corrections should update the claim and failure ledgers, not only the paper narrative.

## Release criteria

A release is warranted when the locked environment installs, tests pass, generated source artifacts reproduce without a dirty tree, the verification render matches the audited layout, secrets and large private files are absent, and the manuscript visibly states its limitations. A release does not imply clinical readiness. Version 0.1.0 represents a reproducible synthetic research milestone.
