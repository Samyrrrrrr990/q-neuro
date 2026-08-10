# Results

Last updated: 2026-08-09. All results use synthetic NeuroWorld cases and are exploratory unless
explicitly marked otherwise.

## Experiment Zero — corrected full-data comparison

Primary artifact: `experiments/results/QN-000003/metrics.json`

Setup: 14,000 training, 3,000 validation, and 3,000 test cases; 20 diagnosis archetypes; 40 binary
findings with explicit missingness; 500 independently generated chronology counterfactual pairs;
three seeds; approximately 20,000 trainable real scalars per model. All models receive the same
findings and demographic context. Sequence models receive the observed evidence order.

| Model | Parameters | Top-1 mean ± SD | Order-twin accuracy | Counterfactual pair accuracy | NLL | ECE | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP (unordered) | 20,002 | 0.7524 ± 0.0052 | 0.4990 | 0.0000 | 0.4373 | 0.0366 | 1.07 s |
| Tiny Transformer | 18,980 | 0.9947 ± 0.0024 | 1.0000 | 1.0000 | 0.0169 | 0.0026 | 12.69 s |
| Real operator state | 19,901 | 0.9993 ± 0.0003 | 1.0000 | 1.0000 | 0.0033 | 0.0020 | 14.65 s |
| Complex operator state | 20,304 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 0.0347 | 0.0333 | 24.47 s |

The MLP result is a successful control: chronology twins have identical aggregate evidence, so an
unordered model cannot consistently predict both directions. All ordered models learn the intended
causal order rule. The full-data task is saturated and therefore cannot establish a meaningful
complex-versus-real advantage.

Complex phase is functional, not decorative: replacing the final learned relative phases by zero
or random phases reduces mean top-1 to 0.205 and 0.216, respectively. This only shows that the
trained complex model uses phase. It does **not** show that phase is a better inductive bias because
the real operator model obtains essentially the same accuracy with much better NLL/ECE and lower
runtime.

## Experiment Zero — sample efficiency

Primary artifact: `experiments/results/QN-000004/metrics.json`. Training sets are nested; the
validation/test generator and seeds are shared across sizes.

### Mean top-1 accuracy

| Training cases | MLP | Transformer | Real operator | Complex operator |
|---:|---:|---:|---:|---:|
| 250 | 0.6627 | 0.4621 | **0.7159** | 0.6732 |
| 500 | 0.7060 | 0.6732 | 0.9403 | **0.9493** |
| 1,000 | 0.7257 | 0.9070 | 0.9846 | **0.9971** |
| 2,000 | 0.7328 | 0.9629 | 0.9948 | **0.9997** |
| 5,000 | 0.7416 | 0.9888 | 0.9992 | **1.0000** |

### Mean counterfactual pair accuracy

| Training cases | MLP | Transformer | Real operator | Complex operator |
|---:|---:|---:|---:|---:|
| 250 | 0.0000 | 0.0107 | **0.3080** | 0.0847 |
| 500 | 0.0000 | 0.3160 | 0.9233 | **0.9933** |
| 1,000 | 0.0000 | 0.9860 | 0.9980 | **0.9993** |
| 2,000 | 0.0000 | 0.9960 | 0.9993 | **1.0000** |
| 5,000 | 0.0000 | 0.9987 | 1.0000 | 1.0000 |

The operator models learn the synthetic chronology relation with fewer examples than the tested
Transformer. The real operator is better in the extreme 250-case regime; complex crosses over by
500 cases and reaches higher top-1 from 500–5,000 cases. That top-1 advantage has a clear cost:
complex NLL and ECE are worse than real at every tested size. At 1,000 cases, for example, ECE is
0.1759 complex versus 0.0284 real, and mean training time is 6.88 versus 4.29 seconds.

![Experiment Zero learning curves](research/figures/generated/experiment_zero_learning_curves.png)

## Generator-shift replication with stronger controls

Primary artifact: `experiments/results/QN-000006/metrics.json`.

This study repeats the 250/500/1,000-case range and adds two controls: a GRU whose learning rate is
selected by validation NLL, and a two-channel real operator with the complex model's paired
magnitude-squared measurement but no complex multiplication or conjugation. Evaluation includes:

- **in-domain:** original NeuroWorld seed and observation process;
- **nuisance-seed shift:** new secondary findings and nuisance evidence stages, with the core
  disease-factor map unchanged;
- **noisy/sparse shift:** a different seed, probabilities mixed 18% toward 0.5, observation rate
  reduced from 0.72 to 0.55, temporal jitter increased, and each chronology marker visible with
  probability 0.70.

### Mean top-1 accuracy at 1,000 training cases

| Model | Parameters | In-domain | Nuisance-seed shift | Noisy/sparse shift | Shift NLL | Shift ECE | Shifted counterfactual pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP | 20,002 | 0.733 | 0.615 | 0.415 | 1.929 | 0.120 | 0.000 |
| Tiny Transformer | 18,980 | 0.910 | 0.752 | 0.506 | 2.416 | 0.256 | 0.587 |
| Tuned GRU | 19,656 | 0.981 | 0.315 | 0.263 | 3.633 | 0.384 | 0.460 |
| Real operator | 19,901 | 0.984 | 0.773 | 0.531 | 1.794 | 0.118 | 0.663 |
| Two-channel real operator | 19,975 | 0.986 | 0.828 | 0.597 | 1.639 | 0.206 | 0.844 |
| Complex operator | 20,304 | **0.995** | **0.896** | **0.660** | **1.451** | 0.210 | **0.991** |

The tuned GRU invalidates the earlier implication that operator states are the most sample-efficient
in-domain mechanism: at 250 cases it reaches 0.920 top-1 versus 0.774 real operator and 0.699
complex operator. Its performance falls to 0.322 on the nuisance-seed shift and 0.272 on the
noisy/sparse shift, suggesting that it exploits simulator-specific temporal regularities.

The complex model is strongest under both shifts from 500 cases onward. At 1,000 cases its paired
top-1 difference over the two-channel real control is +0.068 on the nuisance shift (three-seed
Student-`t` 95% CI +0.006 to +0.130) and +0.064 on the noisy/sparse shift (CI +0.016 to +0.112).
The corresponding difference over the ordinary real operator is +0.123 and +0.129. These are
interesting exploratory signals, but three seeds and two hand-designed shifts are not enough for a
headline claim.

In the noisy/sparse condition, both chronology markers are observed in 49.1% of order-twin cases.
At 1,000 cases the complex model scores 0.998 when marker evidence is complete and 0.380 when it is
incomplete. Counterfactual pairs deliberately expose both markers and vary only their order; the
complex model solves 0.991 of those pairs versus 0.844 for the two-channel control.

Calibration remains a weakness. Although complex has the best shifted NLL at 1,000 cases, its ECE
is 0.210 versus 0.118 for the ordinary real operator. The complex model therefore improves shifted
ranking and likelihood without establishing a calibration Pareto frontier.

![Generator-shift replication](research/figures/generated/generator_shift_replication.png)

## What can be claimed now

- Ordered computation is necessary for the deliberately order-dependent twin task.
- The tested low-rank operator-state models are more sample-efficient than the tiny Transformer,
  but not the tuned GRU, in-domain on this simulator.
- The complex model learns phase-dependent solutions.
- Complex operators are more robust than the tested real, two-channel, Transformer, and GRU controls
  on the two declared generator shifts at 1,000 cases.
- Complex arithmetic has still not demonstrated an overall advantage: the robustness signal trades
  against worse in-domain sample efficiency than the GRU, imperfect calibration, more runtime, and
  dependence on hand-designed synthetic shifts.

## What cannot be claimed

- Novelty, clinical validity, or real neurological diagnostic ability.
- Superiority over tuned modern sequence models across distributions.
- Generalization beyond the fixed NeuroWorld generator.
- A quantum-mechanical interpretation.
- A stable complex advantage across independently designed simulators or real data.

## Validity notes

`QN-000002` is retained but superseded because the MLP alone received demographic covariates.
`QN-000003` corrected that asymmetry and replicated the qualitative result. Student-`t` confidence
intervals summarize three-seed variability and are stored with every aggregate; three seeds still
provide limited power. Peak process RSS remained below 0.36 GiB in these runs, but in-process cache
reuse makes cross-model memory deltas approximate rather than definitive.

`QN-000006` uses validation-only learning-rate selection and never selects against shifted test
metrics. The two shift definitions were designed by the same project that produced the models, so
replication over independently varied causal templates is mandatory.

## Multi-world robustness confirmation

Primary artifact: `experiments/results/QN-000008/metrics.json`.

The confirmation gate trains at 1,000 cases and evaluates five preregistered unseen world seeds at
four shift severities. Every world contains 2,000 test cases and 300 paired chronology
counterfactuals. Models are trained with three seeds; metrics are first averaged across training
seeds within each world, then confidence intervals are computed across the five world means. This
avoids treating cases from one simulator instantiation as independent replications.

| Model | In-domain | Nuisance | Mild | Moderate | Severe | Moderate counterfactual pairs |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.722 | 0.584 | 0.510 | 0.399 | 0.288 | 0.000 |
| Tiny Transformer | 0.910 | 0.747 | 0.632 | 0.491 | 0.359 | 0.574 |
| Tuned GRU | 0.982 | 0.373 | 0.337 | 0.256 | 0.181 | 0.499 |
| Real operator | 0.984 | 0.788 | 0.672 | 0.523 | 0.377 | 0.640 |
| Two-channel real operator | 0.986 | 0.846 | 0.745 | 0.585 | 0.414 | 0.869 |
| Complex operator | **0.996** | **0.909** | **0.806** | **0.645** | **0.468** | **0.987** |

The complex-minus-two-channel top-1 effect remains positive across unseen worlds:

| Severity | Mean paired difference | World-level 95% Student-`t` CI |
|---|---:|---:|
| Nuisance | +0.0627 | [+0.0552, +0.0702] |
| Mild | +0.0609 | [+0.0502, +0.0715] |
| Moderate | +0.0602 | [+0.0529, +0.0674] |
| Severe | +0.0538 | [+0.0471, +0.0606] |

This confirms that the current complex parameterization has a repeatable robustness advantage over
the current two-channel real control inside NeuroWorld. It does not prove that complex arithmetic
is uniquely responsible: the control is not an exhaustive real reparameterization, and all worlds
still share the same simulator family.

### Calibration transport failure

One scalar temperature is fitted using only in-domain validation NLL and applied unchanged under
shift. It fails to transfer. Under moderate shift, raw versus calibrated ECE is 0.122 versus 0.247
for the real operator, 0.201 versus 0.267 for two-channel real, and 0.204 versus 0.257 for complex.
Complex moderate-shift NLL worsens from 1.459 to 4.341 after scaling. In-domain calibration can make
shifted confidence substantially worse even while leaving top-1 predictions unchanged.

![Multi-world robustness sweep](research/figures/generated/robustness_world_sweep.png)

## Orthogonal NeuroWorld task suite

Primary artifact: `experiments/results/QN-000010/metrics.json`. Paired exploratory analysis:
`research/analyses/generated/neuro_task_suite_paired_effects.json`.

This experiment trains each model separately for ordinary/ambiguity evaluation, held-out evidence
composition, and a completely omitted disease class. It also evaluates an unlabeled synthetic
syndrome with a fixed cross-factor signature. All summaries use three training seeds.

| Model | Standard top-1 | Held-out composition top-1 | Ambiguous-pair NLL ↓ | Unknown-disease MSP AUROC | Hidden-syndrome representation AUROC |
|---|---:|---:|---:|---:|---:|
| MLP | 0.732 | 0.921 | 1.681 | 0.709 | 0.741 |
| Tiny Transformer | 0.979 | 0.973 | 6.171 | 0.887 | 0.941 |
| Tuned GRU | 0.993 | 0.995 | 2.450 | 0.985 | 0.983 |
| Real operator | 0.998 | 0.999 | **1.148** | 0.952 | 0.992 |
| Two-channel real operator | 1.000 | 1.000 | 1.453 | 0.997 | 0.852 |
| Complex operator | **1.000** | **1.000** | 2.581 | **0.999** | **0.999** |

### Composition: non-discriminative at this data scale

The composition split removes four declared positive-finding conjunctions from training and
requires at least one at test time. Complex and two-channel models both reach 1.000; real reaches
0.999; GRU reaches 0.995. The reference-versus-held-out generalization gap is effectively zero for
all operator models. This task therefore verifies compositional competence but does not isolate an
interference or complex-state advantage.

### Ambiguity: a clear complex-model weakness

Each ambiguity pair contains observationally identical evidence with two equally valid chronology
labels. The ideal distribution assigns all probability mass equally to the two twins, yielding
pair NLL `log(2) = 0.693`. No model reaches that target. The real operator is best at 1.148, followed
by two-channel real at 1.453 and the MLP at 1.681. Complex reaches 2.581, with only 0.212 mean
probability mass on the two valid labels. Its paired NLL disadvantage versus real is +1.434 (three-
seed Student-`t` interval +1.416 to +1.451). Complex dynamics are therefore not inherently better
at representing irreducible uncertainty; the current measurement becomes overcommitted away from
the valid twin set.

### Unknown disease and hidden syndrome

When label 19 is completely absent from training, complex maximum-softmax uncertainty reaches
0.9988 AUROC, versus 0.9974 two-channel, 0.9847 GRU, and 0.9521 real. The complex-minus-two-channel
paired difference is only +0.0015 and its interval crosses zero, so output-space rejection does not
identify a complex-specific effect. In representation space, complex centroid-distance AUROC is
0.9965; two-channel is highly seed-unstable (0.8285 ± 0.1358).

For the separately generated unlabeled syndrome, complex representation-distance AUROC is 0.9990
and real is 0.9918. Two-channel again varies strongly across seeds (0.8519 ± 0.1599). This is
evidence for robust anomaly separation by the current complex representation, not evidence that a
new disease attractor was discovered: the test measures separability from known cases and does not
fit or recover an unknown cluster count.

With only three paired seeds, the smallest attainable two-sided exact sign-flip p-value is 0.25.
Student-`t` intervals and standardized effects are descriptive, not confirmatory significance
claims.

![Orthogonal NeuroWorld task suite](research/figures/generated/neuro_task_suite.png)
