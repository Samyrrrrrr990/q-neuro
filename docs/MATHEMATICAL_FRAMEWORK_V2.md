# Mathematical framework for ordered hypothesis-state computation

This document distinguishes exact results from empirical conjectures. It does not assert that a
complex parameterization is more expressive than real arithmetic.

## Definitions

**Definition 1 (environment).** An environment `E` is a joint distribution over an evidence
sequence `(e_1, …, e_T)`, target `y`, nuisance variables, and missingness process. A shifted
environment `E'` changes one declared component of this distribution.

**Definition 2 (hypothesis state).** A sequential model maintains `psi_t` in a state space `S` and
updates it by

`psi_(t+1) = F(O(e_t), psi_t)`.

The evidence map `O` may be real or complex; `F` may include injection, nonlinearity, and
normalization.

**Definition 3 (linear order sensitivity).** For linear operators `A` and `B` at state `psi`, the
order effect is

`d_psi(A,B) = ||(AB - BA) psi||`.

**Definition 4 (normalized commutator).**

`C(A,B) = ||AB - BA||_F / (||A||_F ||B||_F)`,

with `C=0` when the denominator is zero. This is a property of the declared linear operators, not
automatically of the full nonlinear update.

**Definition 5 (behavioral order sensitivity).** For paired sequences containing the same evidence
multiset in opposite order, behavioral sensitivity is mean total-variation distance between the
two predictive distributions. Jensen–Shannon divergence is a secondary measure.

**Definition 6 (risk and shift loss).** For loss `ell`, model `m`, and environment `E`,

`R_E(m) = E_E[ell(m(e_1:T), y)]`.

Generalization loss under shift is `G_(E→E')(m) = R_E'(m) - R_E(m)`. A robustness curve evaluates a
performance metric continuously over normalized shift severity, and robustness AUC integrates that
curve over `[0,1]`.

## Exact propositions

**Proposition 1 (commutation removes two-step linear order effects).** If `AB=BA`, then for every
state `psi`, `AB psi = BA psi`.

*Proof.* Substitute the equality and apply both sides to `psi`. This proposition does not cover
token-dependent injection, nonlinear activation, or normalization.

**Proposition 2 (commutator bound).** For every state `psi`,

`||(AB-BA) psi|| <= ||AB-BA||_2 ||psi||`.

*Proof.* This is the defining submultiplicative property of the induced operator norm.

**Proposition 3 (exact real representation of a complex linear map).** For `W=X+iY` and
`z=u+iv`, complex multiplication is represented over real coordinates by

`[Re(Wz); Im(Wz)] = [[X,-Y],[Y,X]] [u;v]`.

*Proof.* Expand `(X+iY)(u+iv)` and collect real and imaginary terms. Therefore Q-Neuro cannot claim
that real arithmetic is incapable of representing its computation. The empirical question concerns
inductive bias, optimization, conditioning, and efficiency at fixed resources.

## Analytic toy family

The law suite defines `A(alpha)` as a two-dimensional rotation through `pi alpha / 2` and `B` as
anisotropic scaling `diag(1.6, 0.625)`. At `alpha=0`, `A` is the identity and commutes with `B`.
As `alpha` increases, the normalized commutator increases monotonically over the sampled grid. The
closed-form matrix products allow exact comparison of `AB psi` and `BA psi` without fitting a
neural network.

This toy family establishes controllability of order dependence; it does **not** establish an
architecture advantage or sample-complexity bound.

## Empirical conjecture to be discovered, then frozen

**Conjecture C1.** If any structured-state advantage exists, its magnitude depends on an interaction
between target-relevant order dependence and environmental shift, rather than on complex arithmetic
alone.

The discovery suite compares linear, logarithmic, saturating, threshold, interaction, and quadratic
forms. A candidate is selected on discovery data and serialized before confirmation data are
opened. Confirmation requires out-of-family `R^2 >= 0.50`, effect-sign accuracy at least `0.80`, and
mean absolute error at most `0.015`. Failure of any threshold rejects a general-law claim.

## Open questions

1. Does a state-conditioned commutator predict robustness better than the global Frobenius norm?
2. Is phase information causal, or merely one convenient coordinate system for a real relational
   memory mechanism?
3. Can structured real rotation, polar, or state-space models match every observed effect?
4. Does any candidate relationship survive independent non-NeuroWorld generators?
5. Are trajectory curvature or preserved alternatives mediators rather than correlates?
