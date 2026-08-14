# QN-LAW-001 provisional freeze record

Status: frozen before held-out provisional confirmation  
Scope: synthetic, nonclinical, outcome-ineligible  
Machine-readable record: `research/laws/FROZEN_CANDIDATE_001.json`

## What was frozen

QN-LAW-001 predicts the cellwise top-1 difference between `complex_operator` and the strongest
evaluated real control from observed order information `I` and normalized shift severity `s`:

`Delta_hat = b0 + b1 I + b2 s + b3 I^2 + b4 s^2 + b5 I s`

with coefficients, in that order:

`[-0.0344888, 0.125439, -0.0289943, -0.210788, 0.00596384, 0.0203217]`.

The quadratic was selected by the prespecified minimum-BIC proxy over linear, logarithmic,
saturating, threshold, interaction, and quadratic candidates. QN-000040 supplied 12 aggregate law
cells. Its discovery fit was `R^2 = 0.9487` and `MAE = 0.00260`.

## What the discovery result means

This is a model of a **negative architecture gap**, not evidence of a complex advantage. Across
all 2,880 nested discovery effects, zero complex-minus-best-real differences were positive. The
mean difference was `-0.03695`. The best-real envelope contained an exact real-block
implementation of the complex computation, so a uniquely complex representational advantage was
already exposed to a direct equivalence falsifier.

The high in-sample fit is not itself confirmatory. A six-coefficient quadratic fit to 12 aggregate
cells has substantial overfitting risk, and the discovery profile omitted the larger training
sizes, the complete real envelope, and the full hyperparameter policy in QNF-PREREG-002.

## Sealed provisional confirmation

Before held-out evaluation, the following were frozen and hashed in the machine-readable record:

- four generator families absent from QN-000040: Bayesian urn, hidden-rule relational, machine
  fault diagnosis, and network-intrusion reasoning;
- 32 world seeds and five training seeds disjoint from discovery and QN-GRAND-001;
- the generator, training/evaluation runner, law evaluator, and confirmation configuration;
- confirmation thresholds: out-of-family `R^2 >= 0.50`, sign accuracy `>= 0.80`, and
  `MAE <= 0.015`.

No coefficient, functional form, family assignment, seed, or threshold may change after the
held-out run starts. Any implementation correction requires a versioned amendment and a new
experiment identifier.

## Claim boundary

Even if all provisional confirmation thresholds pass, QN-LAW-001 cannot establish Outcome E or
authorize a QN-GRAND-001 claim. QN-000040 was explicitly outcome-ineligible and compute-reduced.
The held-out result can only show whether this provisional description transfers to new synthetic
task families. It cannot establish clinical validity, universal superiority, or a quantum account
of cognition.
