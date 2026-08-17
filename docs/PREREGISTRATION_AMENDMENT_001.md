# Q-Neuro preregistration amendment 001

Amendment ID: `QNF-PREREG-002-A1`  
Parent: `QNF-PREREG-002` version `2.0.0`  
Trigger: failed simulator gate `QN-000027`  
Scope: simulator red-team correction only; no architecture outcome has been generated

## Observed failure

The frozen QN-000027 audit found metadata-only test accuracy 0.2095 above the 0.20 threshold and
an apparent order-only accuracy 0.926 above the 0.25 threshold. QN-GRAND-001 remains blocked.

## Diagnosis

The metadata failure is real: legacy NeuroWorld intentionally assigns label-dependent age and sex
distributions. Those covariates can act as a stable shortcut and are not necessary to test the
central computational hypothesis.

The original `order_only` audit was misnamed. It used first and last **token identities**, which
contain observed finding identity and sign, not order metadata alone. High accuracy therefore
mixes a shallow evidence shortcut with temporal position. This metric is retained as
`edge_token_identity` but is diagnostic-only. The corrected `order_only` gate uses only the
positive/negative sign sequence and sign-transition count. The numerical 0.25 threshold is not
changed.

Legacy NeuroWorld also gives non-chronology diagnoses label-specific nuisance stages. The repaired
profile uses one label-independent nuisance-stage schedule and changes only the declared chronology
markers for order twins.

## Frozen corrections

1. Set `demographic_signal_strength: 0.0`, making age and sex independent of label.
2. Set `shared_nuisance_stages: true`, sharing non-marker stages across labels.
3. Redefine `order_only` to exclude token identity; retain endpoint-token identity as a visible
   diagnostic with no confirmatory interpretation.
4. Capture source-worktree provenance before creating the result directory. QN-000027 reported a
   dirty tree because its own new output directory existed when metadata was captured; source was
   clean at invocation.
5. Preserve all QN-000027 files unchanged and rerun under a new numeric experiment ID.

All other thresholds, splits, case counts, seeds, and stop rules remain unchanged. Any failure in
the amended gate continues to block QN-GRAND-001.
