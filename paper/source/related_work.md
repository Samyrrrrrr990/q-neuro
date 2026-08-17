# Related work

## Complex and structured recurrence

Trainable complex-valued neural networks, including complex initialization, normalization, convolution, and recurrent components, are established methods rather than a Q-Neuro invention [@trabelsi2018]. Complex and unitary recurrent networks were developed to stabilize long-range dynamics and represent rotations efficiently [@arjovsky2016; @jing2017]. Real orthogonal recurrence offers a closely related norm-preserving alternative and has shown that carefully structured real models can outperform some unitary controls with fewer parameters [@helfrich2018]. These precedents make a broad complex-versus-real comparison scientifically insufficient.

Arjovsky and colleagues explicitly describe the real two-channel representation of a complex unitary matrix [@arjovsky2016]. For the present study, that known identity is not a novelty claim; it is an adversarial control. More recent theory reinforces why arithmetic, parameterization, and optimization should be separated. Infinite-width analyses show that common real-valued backpropagation can collapse many complex-network training dynamics toward ordinary real dynamics [@tan2022]. Conversely, theoretical settings exist in which a complex neuron represents functions unavailable to a finite-width two-layer real network while learning some real targets more slowly [@wu2023]. Capacity bounds for modReLU networks also begin from the standard identification of complex coordinates with paired real coordinates [@altunoren2026]. Our negative result is therefore architecture-specific, not a theorem that complex models can never help.

## Quantum-inspired state and order effects

Quantum probability has been used to model human question-order and causal-reasoning effects through noncommuting projections and contextual state updates [@wang2014; @trueblood2012]. Hidden quantum Markov models and density-matrix-inspired language models further demonstrate that quantum-mathematical state representations can be useful without implementing physical quantum hardware [@srinivasan2018; @yan2024]. Q-Neuro uses related mathematical motifs - ordered updates, phase-sensitive state, and magnitude-squared readout - but makes no inference about physical quantum cognition. Noncommutativity is a property of the computation, not evidence about its substrate.

## Modern sequence controls and uncertainty

Structured state-space models provide efficient long-sequence dynamics and form an important real-valued comparator family [@gu2022]. Gated recurrent units, orthogonal recurrence, and unrestricted real operators address different inductive biases and optimization surfaces. Calibration and out-of-distribution detection require separate evaluation because high top-1 accuracy does not imply reliable confidence [@guo2017; @hendrycks2017]. Q-Neuro consequently records negative log-likelihood, expected calibration error, and Brier score alongside top-1 accuracy, although the present primary falsifier is top-1 performance against the cellwise best-real envelope.

## Position of the present work

The publishable object is not a new complex algebra. It is a fully traceable reversal of an architectural interpretation. A positive within-simulator result is retained, its insufficient comparator is named, an exact control is implemented, new task families are generated before confirmation, a candidate relationship is frozen, and failed readiness gates stop a sealed test. This sequence differs from an ordinary ablation study because the unfavorable result controls which claims are permitted and which experiment may legally execute. Whether that workflow is novel in the literature requires peer review; no priority claim is made here.
