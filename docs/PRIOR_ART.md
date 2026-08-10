# Prior Art and Novelty Ledger

Status: preliminary living review, focused on mechanisms needed for Experiment Zero. It is not a
systematic review and supports no claim of being first. Sources below link to the original papers
or official proceedings.

Novelty labels: **KNOWN**, **KNOWN VARIANT**, **NEW COMBINATION**, **POTENTIALLY NOVEL**, and
**UNRESOLVED**.

| Q-Neuro idea | Nearest primary work | Similarity | Difference to test | Status | Confidence |
|---|---|---|---|---|---|
| Complex-valued trainable state | [Deep Complex Networks (Trabelsi et al., ICLR 2018)](https://openreview.net/pdf?id=H1T2hmZAb) | Complex weights, activations, normalization, and real losses | Disease-indexed sequential evidence operators and phase-sensitive hypothesis measurement | KNOWN VARIANT | High |
| Norm-preserving complex recurrence | [Unitary Evolution Recurrent Neural Networks (Arjovsky et al., 2016)](https://arxiv.org/abs/1511.06464) | Complex hidden states and structured unitary evolution for stable recurrence | Evidence-specific non-commuting low-rank updates rather than one recurrent unitary map | NEW COMBINATION | Medium |
| Variable internal computation | [Adaptive Computation Time (Graves, 2016)](https://arxiv.org/abs/1603.08983) | Learned, differentiable variable recurrent steps | Halting from hypothesis-state velocity/ambiguity rather than a generic ponder unit | KNOWN VARIANT | High |
| Fixed-point inference | [Deep Equilibrium Models (Bai, Kolter, Koltun, NeurIPS 2019)](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html) | Computation as convergence to an equilibrium with tied transformations | Explicit competing diagnostic coordinates and evidence-conditioned attractor changes | NEW COMBINATION | Medium |
| Hamiltonian state evolution | [Hamiltonian Neural Networks (Greydanus et al., 2019)](https://arxiv.org/abs/1906.01563) | Learned Hamiltonian structure and conservation-inspired dynamics | Hybrid rotational/dissipative inference in a hypothesis field, not learned physical dynamics | NEW COMBINATION | Medium |
| Density-matrix text representations | [Quantum Many-body Wave Function Inspired Language Modeling (Zhang et al., 2018)](https://arxiv.org/abs/1808.09891) and later quantum-inspired language models | Amplitudes, tensor coupling, and quantum-probability-inspired composition for semantic ambiguity | Off-diagonal diagnostic relations evolved through explicit learned dissipative dynamics | UNRESOLVED | Low |
| Ordered evidence as non-commuting operators | Unitary RNNs, recurrent state-space models, and quantum-probability cognition all make composition order consequential | Sequential transforms are generally non-commutative | Explicit commutator measurement tied to counterfactual order tasks and hypothesis observables | POTENTIALLY NOVEL | Low |
| Learned stable diagnostic dynamics | [Learning Stable Deep Dynamics Models (Manek & Kolter, NeurIPS 2019)](https://papers.nips.cc/paper_files/paper/2019/hash/0a4bbceda17a6253386bc9eb45240e25-Abstract.html) | Constrains learned dynamics using stability structure | Stability, metastability, and calibrated unresolved hypotheses in diagnostic state space | NEW COMBINATION | Low |

## Conservative conclusions

1. Complex-valued networks, unitary recurrence, adaptive depth, equilibrium computation, and
   Hamiltonian inductive biases are established research directions.
2. Calling Q-Neuro "quantum" solely because it uses complex amplitudes would be inaccurate.
3. The potentially interesting object is a controlled combination: explicit competing hypothesis
   coordinates, evidence-conditioned non-commuting operators, phase-sensitive measurement, and
   metrics that connect internal commutators to order-dependent counterfactual behavior.
4. Combination novelty is not conceptual novelty. A broader literature search is mandatory before
   any paper claim, especially across quantum cognition, medical Bayesian networks, energy-based
   models, neural operators, and quantum-inspired NLP.

## Next search tranche

- Quantum probability models of question/evidence order in cognition.
- Density-matrix language and information-retrieval models using complex amplitudes.
- Differential-diagnosis systems with explicit hypotheses, factor graphs, or active test choice.
- Non-commutative neural/operator architectures outside quantum machine learning.
- Complex-valued controls that count real scalar parameters and compute fairly.

