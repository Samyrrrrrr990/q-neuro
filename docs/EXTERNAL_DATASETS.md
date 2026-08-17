# External nonclinical dataset ledger

Status: no external dataset selected or evaluated  
Freeze date: 2026-08-14

## Decision

No real-world dataset is included in the current release. The project did not identify, license,
and preregister a dataset whose sequential order semantics, environment split, target, and
exclusions could be fixed before outcome access within this research cycle. Adding a convenient
dataset after seeing the synthetic results would create selection bias and cannot rescue the
failed provisional law or the blocked QN-GRAND-001 run.

| Dataset | Domain | License verified | Environment split frozen | Outcome accessed | Release role |
|---|---|---:|---:|---:|---|
| None | — | No | No | No | No external-validity claim |

## Required preregistration for any future dataset

Before labels or final-test outcomes are accessed, a versioned protocol must record:

- the dataset name, persistent identifier, version, license, and retrieval checksum;
- whether records are patient, human-subject, operational, or fully nonhuman;
- target definition, input modalities, time/order semantics, and missing-data treatment;
- training, validation, and final-test environments defined by site, time, device, geography, or
  another defensible source of distribution shift;
- inclusion, exclusion, deduplication, leakage, and label-quality checks;
- the complex model and complete real comparator envelope, with equal or real-favoring search;
- the primary metric, calibration safeguards, top-level statistical unit, minimum effect, sample
  size, multiplicity correction, and stopping rule;
- data-governance, privacy, ethics, and model-card obligations;
- a statement that external evidence is secondary to, and cannot retroactively alter,
  QN-GRAND-001.

Patient or clinical datasets require a separate governance and validation program described in
`docs/CLINICAL_VALIDATION_ROADMAP.md`. They must not be added as ordinary benchmark rows.
