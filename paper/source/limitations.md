# Limitations

## Synthetic evidence and external validity

All next-phase tasks are synthetic and nonclinical. No external real-world dataset was selected, licensed, preregistered, or evaluated. The task families expose useful structural variables, but their labels, shifts, and order semantics are authored by the same research program that evaluates them. Performance does not imply usefulness for medicine, cybersecurity operations, industrial maintenance, or human reasoning despite the nonmedical task metaphors.

## Reduced comparator and search envelope

Only five model families were executed in discovery and held-out evaluation, yielding four real controls rather than the 14 required by the full preregistration. Each model received one learning-rate configuration, not the required 8 candidate and 10 real-control trials. The result therefore cannot estimate performance under comprehensive tuning. The best-real envelope is also selected within evaluation cells, making it an intentionally severe falsifier rather than a deployable model-selection procedure.

## Incomplete compute matching

Nominal evaluated parameter budgets are matched, but per-trial FLOPs and optimizer-step counts were not preserved. Wall-clock results depend on CPU implementation and are not used for the central claim. A future full run must record optimizer steps, measured or analytically estimated FLOPs, peak memory, and search totals for every trial.

## Missing raw predictions

The reduced experiments preserve aggregate metrics and paired effects but not raw per-case prediction arrays. This limits reanalysis of alternative calibration, subgroup, and dependence-aware statistics. It also constitutes a mandatory grand-test blocker. The summary can be reproduced from cached aggregate artifacts, but a full independent statistical reconstruction from logits is not possible.

## Outcome ineligibility

Discovery omits the 5,000 and 25,000 training sizes, and held-out evaluation uses 1,000 rather than the grand-test primary size of 5,000. The full ShiftGauntlet model-outcome grid was not run. Consequently, neither reduced study may support the strongest preregistered outcome category. The observed negative effect is informative about the executed scope but is not a completed universal comparison.

## Investigator and replication limits

The code, benchmark, prior-art review, analyses, and manuscript were produced by one independent investigator with coding-assistant support. The prior-art review is primary-source-led but not systematic; patents, theses, non-English work, and unpublished industrial results were not exhaustively screened. No independent team has yet rerun the pipeline or audited the generator semantics.

## Statistical limits

Only four held-out generator families contribute to the top level of the hierarchy. Worlds and training seeds are numerous but are not substitutes for diverse generator mechanisms. The hierarchical interval assumes the implemented resampling hierarchy, while the exploratory sign-flip test is not outcome-eligible. Exact zero effects partly reflect finite test-set resolution and shared mapped predictions.

## What the result does not falsify

The study does not falsify every complex-valued network, every nonlinear complex activation, phase-native input domain, alternative optimizer, or efficiency objective. It also does not establish that unconstrained real networks always dominate. It falsifies an intrinsic robustness advantage for the implemented Q-Neuro operator over its exact and strongest tested real controls under the executed synthetic profiles.
