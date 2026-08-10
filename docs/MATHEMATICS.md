# Mathematical Specification: Experiment Zero

This document defines the implemented minimal models. It intentionally does not formalize deferred
Hamiltonian or density-matrix branches.

## Evidence and missingness

There are `F` binary findings. A case has a partially observed vector
`x in {-1, 0, +1}^F`, where `-1` is observed absent, `0` is missing, and `+1` is observed present.
Missing is never treated as negative. The sequential representation is an ordered list of signed
tokens `e_1, ..., e_T` drawn from `2F` possible tokens. Padding has its own masked identifier and
never acts on the state. Every compared model also receives the same two demographic covariates
`c` (scaled age and binary sex) so input asymmetry cannot explain a result.

## Conventional baseline

The MLP receives `[x, m]`, where `m_j = 1[x_j != 0]`. Explicitly providing the observation mask
prevents absence and missingness from being conflated:

`h = GELU(W_1 [x;m] + b_1)`, `logits = W_2 h + b_2`.

The stronger recurrent control embeds each signed evidence token, initializes a GRU state from the
same demographic context, and uses the final recurrent state for diagnosis. Its learning rate is
selected only by validation NLL from a preregistered grid.

## Low-rank operator state

Let `h_t in R^S`, `U_e,V_e in R^(S x r)`, and `b_e in R^S`. Initialize with
`h_0 = N(theta_0 + W_c c)`. For evidence token `e_t`,

`q_t = V_e^T h_t`,

`delta_t = b_e + U_e q_t`,

`h_(t+1) = N(h_t + alpha tanh(delta_t))`.

`N(v) = sqrt(S) v / max(||v||_2, epsilon)` bounds state norm without forcing a probability
simplex. The effective linear residual operator before the nonlinearity is
`O_e = I + alpha U_e V_e^T`. For two tokens `a,b`,

`[O_a,O_b] = alpha^2(U_a V_a^T U_b V_b^T - U_b V_b^T U_a V_a^T)`,

which is nonzero in general. We log sampled Frobenius commutator norms; nonzero norms alone are not
evidence that the mechanism is useful.

## Complex operator state

Let `z_t in C^S` with complex `U_e,V_e,b_e`, initialized by a complex affine map of the same
demographic context. Evolution uses the Hermitian inner product:

`q_t = V_e^H z_t`,

`delta_t = b_e + U_e q_t`,

`z_(t+1) = N_C(z_t + alpha tanh(delta_t))`,

where `N_C(z) = sqrt(S) z / max(sqrt(sum_j |z_j|^2), epsilon)`. Relative phases change both
`V_e^H z_t` and subsequent updates.

For complex readout vectors `w_d`, the diagnostic amplitude and measurement are

`a_d = w_d^H z_T`,

`p(d | z_T) = (|a_d|^2 + epsilon) / sum_k (|a_k|^2 + epsilon)`.

Training uses cross-entropy on `log(|a_d|^2 + epsilon)`. A global phase rotation leaves the
measurement invariant, while relative phase generally does not.

## Two-channel real magnitude control

The two-channel control evolves an unconstrained flat real state using the same low-rank operator
law as the real model. Its readout emits two real amplitudes per diagnosis:

`(r_d, s_d) = W_d h_T + b_d`,

`logit_d = log(r_d^2 + s_d^2 + epsilon)`.

This matches the complex model's paired magnitude-squared measurement while removing complex
multiplication, conjugation, and an explicit phase algebra. It therefore tests whether a complex
result can be explained by two interacting real channels and the measurement rule.

## Controlled generator shifts

The replication varies only declared simulator parameters. `probability_mixing` moves every
finding probability toward 0.5, `temporal_jitter` perturbs evidence times, and
`order_marker_visibility` independently hides the two chronology markers. The evaluator separates
cases with complete marker evidence from those with one or both markers missing. Counterfactual
pairs always expose both markers so their single-factor intervention remains valid.

## Parameter accounting

Every real scalar is counted. A complex tensor with `n` entries contributes `2n` trainable scalar
parameters. Model widths are selected by searching integer widths and minimizing absolute distance
to a configured budget. Parameter matching is reported, never assumed.

## Metrics

- Top-1 accuracy: mean indicator that the highest measured hypothesis equals the target.
- Top-3 recall: mean indicator that the target is among the three highest measurements.
- NLL: mean negative log probability assigned to the target.
- ECE: weighted absolute confidence/accuracy gap over fixed confidence bins.
- Order accuracy: top-1 accuracy restricted to generator-marked order-dependent cases.
- Shuffle delta: ordered accuracy minus accuracy after independently permuting observed tokens.

Across seeds, the repository reports the arithmetic mean, sample standard deviation, and a
two-sided 95% Student-`t` confidence interval. With only three seeds these intervals are necessarily
wide; they describe seed variability under the tested setup, not population-level clinical
uncertainty.

## Invariants tested

- Generator values distinguish missing and observed-negative evidence.
- Counterfactual pairs differ in exactly the declared causal factor.
- Padding does not evolve the operator state.
- State norms remain finite and bounded.
- Complex probabilities are finite, non-negative, and sum to one.
- Swapping two operators with a nonzero commutator can change the state.
