# ShiftGauntlet protocol

`ShiftGauntlet` is Q-Neuro's deterministic distribution-shift compiler. It does not assert that a
model is robust. It creates named test interventions whose construction can be inspected before
any outcome-eligible model comparison is run.

## Design rules

1. Severity is always in `[0, 1]`; severity zero is an exact observational identity.
2. Each intervention receives an independent recorded seed.
3. Main-effect runs apply one family at a time. Interaction runs are separately labeled.
4. Source cases are copied and never mutated.
5. Labels are preserved except where prevalence selects cases, class expansion withholds training
   labels, or irreducible ambiguity intentionally pairs identical observations with different
   targets.
6. Contradiction and duplication deliberately permit repeated finding identities in a sequence.
7. Structural compilation is not an outcome experiment and cannot support an architecture claim.

## Intervention semantics

| Family | Controlled change | Important boundary |
|---|---|---|
| nuisance variable | invert synthetic age/sex surface covariates | no causal evidence change |
| prevalence | weighted selection without replacement | output size can shrink |
| conditional feature | label-conditional sign corruption | chronology retained |
| spurious inversion | label-correlated metadata is reversed across splits | explicitly split-dependent |
| missingness | MCAR, MAR, or MNAR-like token deletion | at least one token retained for batching |
| observation noise | evidence-sign corruption | token order retained |
| evidence order | canonical, random, local corruption, adversarial marker swap, reverse | evidence multiset retained |
| distractors | insert label-independent unobserved findings | sequence can lengthen |
| contradictions | append opposite signs for earlier findings | repeated finding identities intended |
| delayed decisive evidence | move declared high-information findings late | evidence multiset retained |
| unseen factor combinations | label-dependent recombination within known factor groups | synthetic structural extrapolation |
| unseen world mechanisms | shared remapping within causal factor groups | not a real-world domain shift |
| evidence deletion | random removal | at least one token retained |
| evidence duplication | repeat observations | vector view cannot encode multiplicity |
| temporal dependency | group the sign process into persistent runs | marginal evidence retained |
| class expansion | withhold up to four classes from training, retain them at test | fixed 20-class output space |
| irreducible ambiguity | copy observables across different labels | Bayes error is intentionally nonzero |

## Audits and interpretation

The compiler records input/output class frequencies, sequence lengths, changed-case counts, and
conflicting observable collisions. Token range, evidence range, nonempty sequence, unique case ID,
and zero-severity identity checks block downstream use when violated.

Several families are deliberately synthetic abstractions. Passing them cannot establish clinical
or real-world robustness. Their role is to isolate candidate failure modes and test whether any
apparent architecture advantage depends on one narrow simulator behavior.
