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

## Mechanism-suite extensions

These laws were introduced only after the original complex robustness signal replicated. They are
quantum-inspired mathematical parameterizations, not physical models of diagnosis.

### Energy-attractor dynamics

Let `a_d in R^S` be learned disease attractors and let `f(x)` be the mean signed-evidence embedding
plus demographic context. With temperature `T`,

`q_t(d) = softmax_d(-||h_t - a_d||^2 / T)`,

`a_bar_t = sum_d q_t(d) a_d`,

`h_(t+1) = N(h_t + eta tanh(f(x) + a_bar_t - h_t))`.

Measurement is `logit_d = -||h_T-a_d||^2/T`. This model is deliberately order-invariant because
`f(x)` averages evidence. Its chronology failure is therefore an intended falsification check.

The adaptive variant computes velocity, entropy, and top-two separation at each step:

`v_t = ||h_t-h_(t-1)|| / sqrt(S)`,

`r_t = [v_t, H(q_t), q_t^(1)-q_t^(2)]`,

`p_t = sigmoid(w^T r_t + b)`.

ACT-style allocation weights combine step logits and define a soft expected depth. A small ponder
penalty is included. This is differentiable, but the current implementation evaluates all steps;
soft expected depth must not be described as realized compute savings.

The hard follow-up maintains an active index set `A_t`. For each active case it computes

`v_t = ||h_t-h_(t-1)|| / sqrt(S)`

and exits when `t >= t_min` and `v_t <= delta`, where `delta` is selected on source validation under
explicit accuracy/NLL tolerances. Halted cases are removed from subsequent tensor operations, so
`sum_i t_i` is an executed state count rather than a soft expectation. In QN-000023, `t_i=2` for
every case. This makes the selected law algebraically equivalent to fixed two-state truncation; it
does not exhibit case-adaptive depth.

### Hamiltonian–dissipative state

For complex state `z` and evidence token `e`, define a Hermitian low-rank action

`H_e z = diag(d_e) z + U_e U_e^H z`,

where `d_e` is real. The discrete update is

`z' = N_C(z + dt (b_e - i H_e z - softplus(g_e) odot z))`.

Pure Hamiltonian, pure dissipative, and hybrid variants remove the corresponding term before
training. All retain complex evidence injection and magnitude-squared measurement so the ablation
targets the evolution law. The normalization bounds explicit-Euler drift; it is not claimed to be
an exact unitary integrator.

### Low-rank Diagnostic Density Dynamics

Rather than optimizing a dense density matrix, D3 evolves a rank-`K` complex factor
`L in C^(D x K)` and measures

`rho = L L^H / tr(L L^H)`.

The token update uses a diagnosis-space Hermitian action, damping, and factor injection:

`L' = N_F(L + dt(-i H_e L - Gamma_e odot L + B_e))`.

This construction guarantees, up to floating-point error:

- `rho = rho^H` (Hermiticity),
- `u^H rho u >= 0` (positive semidefiniteness),
- `tr(rho) = 1`.

The measured diagnosis probability is the diagonal `rho_dd = sum_k |L_dk|^2`. Off-diagonal
Frobenius norm is logged as coherence. Nonzero coherence is only a state descriptor; predictive
usefulness requires a separate later-resolution experiment.

### Factorized and associative controls

The coupled-tensor model forms `tanh(W_L x) odot tanh(W_R x)` and concatenates a linear nonlinear
channel before readout. It controls for multiplicative feature interaction without phase.

The associative-memory baseline uses learned disease queries to retrieve a weighted sum of token
embeddings, while the factor-graph model performs shared message passing over a fixed declared
NeuroWorld adjacency. These are controls, not novelty claims.

### Critical ablation laws

The commutative complex accumulator removes operator composition:

`z = N_C(z_0 + sum_t b_(e_t) + W_c c)`.

Because addition commutes, any chronology information is exactly absent. It retains a complex state
and the same form of magnitude-squared readout.

The phase-insensitive readout retains full complex evolution but replaces the complex inner product
with a constructive magnitude sum:

`a_d = sum_j |z_j| |w_dj|`, `logit_d = log(a_d^2 + epsilon)`.

It cannot express destructive readout interference. The negative-evidence ablation prevents tokens
40–79 from acting on the state, while retaining the observation process and all positive tokens.
This tests information removal, not a fully specified contradiction operator.

### Frozen hierarchical observables

Let `s(x)` be a final frozen diagnostic state and `y_f` one simulator factor. The linear probe is

`p(y_f | x) = softmax(W_f s(x) + b_f)`.

For a complex state `z`, a `C`-class Hermitian observable probe uses one learned matrix per class:

`o_c(z) = z^H A_c z + b_c`, where `A_c = A_c^H`.

The implementation parameterizes `A_c = (B_c + B_c^H)/2`, so every score is real up to numerical
precision. Probe regularization is selected without test labels, and the diagnostic network remains
frozen. The resulting accuracy measures whether the factor can be decoded by the chosen function
class. It does not imply independence of factors, causal use by the diagnosis head, or a physical
measurement interpretation.

## Experimental learning laws

### Multi-objective diagnosis

Experiment Six uses only three controlled losses:

`L = L_diagnosis + lambda (L_mechanism + L_localization)`.

The factor losses are defined only for factorial diagnoses 8–19; chronology twins do not receive
fabricated factor labels. The auxiliary heads are discarded at deployment. This tests targeted
factor supervision, not the eleven-loss omnibus objective proposed as a possibility in the research
directive.

### Phase Gradient Optimization

Let `g_0` be the diagnosis gradient and `g_k` an auxiliary task gradient. The measured task
relationship is

`c_k = <g_0, g_k> / (||g_0|| ||g_k||)`.

PGO assigns `phi_0 = 0` and `phi_k = 1/2 arccos(c_k)`. For every explicit real/imaginary parameter
pair, form `G_k = g_(k,real) + i g_(k,imag)` and combine

`G = (1/K) sum_k exp(i phi_k) G_k`.

The real and imaginary parts of `G` are installed as the corresponding parameter gradients before
an AdamW step. Agreement (`c=1`) remains phase aligned; full opposition (`c=-1`) moves an auxiliary
update into quadrature instead of letting it directly cancel diagnosis. Unpaired real parameters
receive `sum_k cos(phi_k) g_k / K`. PGO still requires reverse-mode task gradients and is not a
local or quantum algorithm.

### Transition-local plasticity

Each diagnosis is assigned a fixed normalized complex prototype `p_y`. At evidence step `t`, the
usual complex operator produces `z_t` and a locally available error observable

`e_t = p_y - z_t`.

The token injection receives a class-conditional delta update. Low-rank factors use local pre/post
signals, schematically

`Delta L_e proportional to e_t (R_e^H z_(t-1))^H`,

`Delta R_e proportional to z_(t-1) (L_e^H e_t)^H`.

Updates are averaged per active token and norm clipped. No autograd graph or backward call is used.
This is supervised local credit, not biologically validated Hebbian learning. The ZeroBackprop
prototype freezes dynamics entirely and sets each complex readout vector to the normalized class
centroid of its frozen states. The hybrid applies local updates before ordinary global AdamW.

## Evidence-level trajectory diagnostics

For complex states `z_0, ..., z_T`, the stored diagnostic path is the actual sequence computed by
the trained operator model. Hypothesis amplitudes and probabilities at every step are

`a_d(t) = <w_d, z_t>`,

`p_d(t) = |a_d(t)|^2 / sum_j |a_j(t)|^2`.

State velocity and normalized path length are

`v_t = ||z_t-z_(t-1)||_2 / sqrt(S)`,

`ell = sum_(t=1)^T v_t`.

The displayed complex-plane curve is `(Re a_d(t), Im a_d(t))`, not a projection fitted for the
figure. For a chronology pair, distance is `||z_t^AB-z_t^BA||_2/sqrt(S)`. Operational revival is
defined only when an observed-negative token first decreases true-label probability by more than
0.05 and a later step returns to the pre-drop level. These are state diagnostics, not textual
reasoning or clinical explanations.
