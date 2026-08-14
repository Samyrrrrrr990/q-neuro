# ShiftGauntlet variance pilot

The variance pilot is an outcome-ineligible planning experiment. Its only decision authority is to
select the smallest preregistered confirmatory world count (`32`, `40`, `48`, or `60`) estimated to
provide 90% power for the minimum practical paired effect of `0.02`.

## Design

- statistical unit: eight independently generated evaluation worlds;
- optimization seeds: `1103`, `2203`, and `3301`;
- training sizes: 250 and 1,000 cases;
- severities: 0, 0.5, and 1;
- architectures: complex operator, exact real block equivalent, dense real recurrence, GRU, and
  diagonal state-space control;
- resource target: approximately 20,304 trainable real scalar parameters per architecture;
- hyperparameters: one common pilot learning rate, selected without shifted outcomes;
- special training profiles: spurious-correlation training and class-withholding training are
  trained separately so their test interventions are meaningful;
- endpoint used for planning: world-level complex-minus-best-real top-1 robustness AUC.

The best-real envelope is chosen within each matched task/world/seed cell. Effects are then averaged
over variants and optimization seeds within world. The larger standard deviation across the two
training sizes is passed to the frozen power simulation. This is intentionally conservative.

## Interpretation boundary

Pilot architecture effects are stored because selective hiding would be improper, but they cannot
confirm or falsify the main hypothesis. The pilot uses fewer models, less tuning, smaller datasets,
and fewer worlds than confirmation. A separate smoke profile validates execution and is explicitly
labeled as neither the frozen pilot nor an outcome experiment.
