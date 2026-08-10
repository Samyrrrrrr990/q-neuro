# Architecture Candidates

Status: design review before Experiment Zero. Candidate 1 and Candidate 2 are selected for the
first decisive test; the rest are deferred.

## 1. Low-rank real evidence operators — selected control

For token `e`, evolve a real state with a residual rank-`r` operator and a token injection:

`h' = normalize(h + alpha * tanh(b_e + U_e(V_e^T h)))`.

Why test it: rank-one or low-rank updates are generally non-commuting, cheap, and expose an exact
matrix-valued mechanism. It isolates ordered operator composition without complex arithmetic.

Failure modes: residual normalization may erase evidence strength; a flexible recurrent map may
learn order without any useful hypothesis geometry; token-specific parameters may overfit.

## 2. Complex phase-interference operators — selected candidate

Replace `h`, `b`, `U`, and `V` with complex quantities. Measure diagnosis `d` through a complex
projection `a_d = w_d^H h`, followed by `logit_d = log(|a_d|^2 + epsilon)`.

Why test it: phase affects subsequent inner products, operator updates, and final interference, so
complex numbers are not decorative. A randomized-phase ablation can directly test the mechanism.

Failure modes: twice as many real scalars per complex parameter; complex nonlinearities can be
unstable; a two-channel real network may be equivalent; phase may be unidentifiable under global
rotation. Parameter and compute matching are mandatory.

## 3. Hybrid Hamiltonian–dissipative dynamics — deferred

`dz/dt = -i H(e,z)z - gamma grad E(z,e) + f_e`.

Why it might matter: separates reversible exploration from irreversible elimination and may give a
meaningful diagnostic-time trajectory.

Why not first: stable integration, Hermitian parameterization, energy design, and fair recurrent
controls add multiple confounders. A positive result would not identify which mechanism mattered.

## 4. Diagnostic Density Dynamics (D3) — deferred/high risk

`d rho/dt = -i[H_e,rho] + D_e(rho)` with Hermitian, positive-semidefinite, unit-trace state.

Why it might matter: off-diagonal terms can encode unresolved hypothesis relations unavailable in a
probability vector.

Why not first: storage scales quadratically with hypotheses; completely-positive dissipators and
PSD-preserving numerical updates are costly; an unconstrained covariance state is a strong control.

## 5. Attractor/equilibrium hypothesis field — deferred

Evolve until `||h_{t+1}-h_t||` is below tolerance in an evidence-conditioned energy landscape.

Why it might matter: adaptive diagnostic time and metastability become structural rather than
auxiliary predictions.

Why not first: fixed-point solvers and implicit differentiation obscure the simpler question of
whether ordered hypothesis-state operators help at all.

## Decision

Experiment Zero begins with a matched MLP, Candidate 1, and Candidate 2. The first ablations remove
order and randomize complex phase. No Hamiltonian, density, attractor, optimizer, or UI work is
justified until this minimal comparison yields a replicated signal or a clear negative result.

