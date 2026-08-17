# Independent nonclinical task generators

The independent task suite tests whether any structural relationship survives outside NeuroWorld.
It contains no neurological semantics, patient data, or clinical endpoint.

## Families

- hidden causal machine;
- sequential detective;
- machine fault identification;
- network intrusion event reasoning;
- hidden relational rule;
- exchangeable Bayesian-urn control;
- analytic noncommutative composition;
- analytic commutative composition.

All families use the same 80-token transport and 20-output envelope only to avoid architecture-
specific adapters. Their label rules, causal marker pairs, seeds, and shifts are defined in
`independent_tasks/generators.py`, not in NeuroWorld.

For causal-order families, counterfactual pairs have the same evidence multiset and different
targets after the declared marker order is reversed. For commutative controls, reversal preserves
both evidence and target. Intermediate order-dependence settings replace a declared fraction of
causal-order examples with an explicit relational tag, maintaining predictability while reducing
the fraction for which order is necessary.

The initial generators are deliberately minimal controlled tasks, not realistic simulations of
industrial machinery, policing, cybersecurity operations, or Bayesian decision-making. Family
names describe abstract reasoning motifs. Generalization to real datasets requires separate,
licensed benchmark selection and preregistration.
