# Architecture and comparator envelope

## Candidate operator

The Q-Neuro candidate is a low-rank complex recurrent operator with token-conditioned state transitions, explicit sequence order, and a real-valued class readout derived from its terminal state. The independent-task profile used operator rank 2, maximum sequence length 96, and an evaluated parameter budget of 20,304 real-valued degrees of freedom. The architecture was designed for sequential evidence integration, not for physical quantum computation.

## Exact real block

The exact-real-block operator replaces every complex state with concatenated real and imaginary coordinates and every complex multiplication with the tied real block from the preceding section. It preserves the candidate's computation, parameter count, data order, initialization mapping, and training policy. This control is stronger than the historical two-channel operator, whose channels were real but were not constrained to implement the candidate's exact complex coupling.

Across the four held-out task families, mapped complex and exact-real models produced identical top-1 values in all 1,920 nested evaluation cells. The maximum absolute difference was 3.58 x 10^-7 for NLL and 1.19 x 10^-7 for ECE. Earlier mechanism mapping found a maximum recorded metric difference of approximately 4.8 x 10^-7. These values are consistent with floating-point accumulation, not a material functional difference.

## Additional real controls

The reduced discovery and held-out profiles also evaluated a structured state-space model, a gated recurrent unit, and a real polar operator. Together with the exact block, these form the executed best-real envelope. The complete preregistered envelope contained 14 real controls, including causal and conventional Transformers, LSTM and vanilla recurrence, orthogonal and residual-gated recurrence, unrestricted paired-real operators, and several additional operator parameterizations. Ten required real controls were not trained in the reduced profile.

This distinction controls the manuscript's claims. The results strongly reject an advantage over the **executed** best-real envelope and establish exact equivalence for the implemented mapping. They do not certify that the five-model reduced set is a complete survey of real sequence learning. The incompleteness is one reason QN-GRAND-001 was blocked.

## Matching policy

Evaluated models shared a nominal parameter budget and the same source data, train/validation splits, training seeds, batch size, maximum epochs, and early-stopping policy. The reduced studies used one learning-rate choice per model rather than the preregistered eight trials for the candidate and ten for each real control. Per-trial FLOPs and optimizer-step counts were not recorded. Parameter matching is therefore documented; complete compute and search matching is not claimed.
