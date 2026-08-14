# QN-GRAND-001 preflight provenance note 001

Date: 2026-08-14  
Affected artifact: `experiments/results/QN-GRAND-001/environment.json`  
Outcome data accessed: no

## Observed issue

The QN-GRAND-001 runner called the clean-worktree guard before reserving the result directory, but
captured `environment.json` after creating that directory. Git therefore reported the runner's own
new `experiments/results/QN-GRAND-001/` output as an untracked dirty path. No source, configuration,
or prior evidence file was dirty when preflight began.

## Correction

The runner now captures the source environment immediately after the clean-worktree guard and
before reserving any output path. The original QN-GRAND-001 artifact remains unchanged. It is not
rerun because QNF-PREREG-002 permits QN-GRAND-001 only once.

## Scientific impact

None. The sealed benchmark was not opened, no primary effect was estimated, and the blocked
decision does not depend on this field. The six recorded gate failures remain unchanged:

1. incomplete preregistered real envelope;
2. insufficient architecture-search budgets;
3. missing compute-matching records;
4. missing full ShiftGauntlet outcome grid;
5. reduced rather than full discovery protocol;
6. missing raw-prediction artifacts.
