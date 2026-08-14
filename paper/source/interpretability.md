# Interpretability and state audit

## Frozen factor probes

Linear probes recover mechanism, localization, temporality, and context from the final complex state with accuracies approximately 0.932, 0.933, 0.907, and 0.918. This confirms that simulator factors remain accessible after diagnosis training. It does not show disentanglement or causal use. A factor can be linearly decodable because it correlates with the label even if the diagnosis head ignores that direction.

Conventional states provide the necessary comparison. GRU probes are stronger on all four factors, and the diagonal state-space model is also highly decodable. Complex-minus-GRU probe effects are negative throughout. Across architectures, factor-probe strength correlates only moderately with shift robustness, around r = +0.45. The complex model’s robustness cannot therefore be explained by uniquely transparent factor coordinates.

Hermitian quadratic probes provide a complex-native observable, but they do not uniformly improve generalization and can worsen NLL. Hermiticity guarantees real scores; it does not guarantee an interpretable observable. Probe selection remains validation-only and the diagnosis model remains frozen.

{{figure:observable_probe|Mechanism, localization, temporality, and context are decodable from several frozen states. GRU and state-space controls are at least as strong as the complex operator, refuting unique hierarchical interpretability.|Heatmap and effect plot comparing linear and Hermitian probes across architectures and simulator factors.}}

## Evidence-level trajectories

For held-out cases, the final complex model reaches 0.969 top-1. Mean entropy decreases by 1.266, normalized path length is 2.956, and final state velocity is 0.175. Chronology twins end 0.841 normalized state units apart and reach a maximum separation of 0.868. These are direct measurements of the recurrent path, not a post hoc embedding.

Positive evidence increases true-label probability by 0.0467 on average. Observed-negative evidence produces a small positive mean change, 0.0109, because many negative findings rule out competing labels rather than directly contradicting the true label. The positive-minus-negative paired difference is 0.0358 with interval [0.0270, 0.0447]. A simplistic monotonic rule—positive always raises and negative always lowers the target—is therefore inappropriate for differential reasoning.

Only 4.97% of cases contain an observed-negative step that lowers true-label probability by more than the preregistered 0.05 threshold. Among those qualifying drops, 75.9% later recover to the pre-drop level. “Revival” is thus a real but conditional behavior. It should not be generalized to every case or described as human-like reconsideration.

{{table:trajectory_metrics}}

## Failure-case interpretation

The trajectory tools expose where a state changes, but not why in clinical language. An amplitude curve can show that two labels exchange probability after a token, and paired-state distance can show when order begins to matter. Neither proves that the network represents a named mechanism. The safest interpretation is computational: the model preserves and revises a distributed state in a way that can be measured at each evidence transition.

Useful future interventions include token deletion, time reversal, factor swaps, head ablation, and state-direction editing followed by diagnosis measurement. Those experiments could connect decodable factors to causal use. Until then, the repository calls these outputs state audits rather than explanations.

## Interpretability boundary

The model has an advantage over opaque one-shot classification in observability: it exposes a real recurrent trajectory, amplitudes, probabilities, entropy, phase, and velocity. This is engineering transparency. It is not clinical interpretability, because the state dimensions and operators do not map to validated medical concepts. Any user-facing explanation would require a separate grounded language layer and expert evaluation.
