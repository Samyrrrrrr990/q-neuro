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

## What can be claimed now

- Ordered computation is necessary for the deliberately order-dependent twin task.
- The tested low-rank operator-state models are more sample-efficient than this small Transformer
  on this specific simulator and budget.
- The complex model learns phase-dependent solutions.
- Complex arithmetic has not demonstrated an overall advantage: its small medium-data top-1 gain
  trades against worse calibration, worse likelihood, more runtime, and worse performance at 250
  cases.

## What cannot be claimed

- Novelty, clinical validity, or real neurological diagnostic ability.
- Superiority over tuned modern sequence models or other recurrent baselines.
- Generalization beyond the fixed NeuroWorld generator.
- A quantum-mechanical interpretation.
- A stable complex advantage without generator-shift replication, stronger baselines, and a
  two-channel real control.

## Validity notes

`QN-000002` is retained but superseded because the MLP alone received demographic covariates.
`QN-000003` corrected that asymmetry and replicated the qualitative result. Student-`t` confidence
intervals summarize three-seed variability and are stored with every aggregate; three seeds still
provide limited power. Peak process RSS remained below 0.36 GiB in these runs, but in-process cache
reuse makes cross-model memory deltas approximate rather than definitive.

