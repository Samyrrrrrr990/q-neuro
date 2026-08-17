# DISCOVERY-002 — Canalized Quotient Dynamics

Preregistration ID: `DISCOVERY-002`

Lane: **B (discovery)** — exploratory. Governed by `docs/LANE_POLICY.md`.

Date: 2026-08-15

Internal program name: Developmental Machine Learning. Neither name may be presented as an
established field.

Scope: synthetic, analytic, nonclinical computational research only.

---

## 0. DISCOVERY-001 remains falsified

`DISCOVERY-001` is **not rescued by this document.** It was frozen twice, opened twice, and failed:

- `DISCOVERY-001-P1` (`sha256 50e438e5…`) — **VACUOUS.** Its grid never reached ρ ≥ 1.1, giving 197
  chances to false-alarm and zero chances to miss.
- `DISCOVERY-001-P2` (`sha256 7d77e159…`) — **FAILED.** 96 of 96 cells at ρ ≥ 1.1 converged;
  divergence rate 0.0000 even at ρ = 3.0.

Recorded as `FAIL-007`. It stands as a statement about **quadratic objectives only**. The surviving
fragment is the `ρ < 1 ⟹ stable` direction, with zero false alarms across 269 scored cells.

Nothing in DISCOVERY-002 may cite DISCOVERY-001 as support.

---

## 1. The clue the failure gave

P2's frozen `anticipated_failure_modes` named the cause before the run: ρ is computed from the
Hessian at initialization, and a nonlinear learner **changes the geometry it occupies**.

The lesson generalizes past that one estimator:

> A nonlinear learner cannot generally be understood from the geometry at initialization.

So the program stops searching for another clever scalar evaluated at `t = 0`. The scientific object
changes from a point measurement to a **trajectory-and-basin** measurement.

---

## 2. The change of object

Old question: *did two parameter vectors stay close?*

New question: **did two perturbed developmental trajectories reach the same functional attractor?**

### Phenotype map

`Π : S → F` from full training state (parameters, optimizer state, and any internal training state)
to a space of predictors. **Phenotype is never defined by raw weights.**

Declared representation for DISCOVERY-002: **logits on a frozen audit batch**, fixed before any
outcome is examined. Declared metric: **max-norm logit distance** as primary, **Jensen–Shannon
divergence** of predictive distributions as secondary. Both declared here, in advance.

### Quotient

`G` is a declared family of semantics-preserving transformations (hidden-unit permutation, scaling
orbit, and the maps already certified in `qneuro.equivalence`). Dynamics are studied on `S/∼`.

Manifold language is **not** used. These actions are non-free, have stabilizers, and produce
singular strata; quotient-space language is used throughout.

---

## 3. Biological correspondence — analogy, not equivalence

| Biology | Machine learning |
|---|---|
| genotype | training specification |
| developmental state | parameter + optimizer + internal state |
| development | training trajectory |
| phenotype | implemented predictor |
| canalization | perturbed trajectories reaching the same functional phenotype |
| attractor / basin | stable predictor-level region and its pre-image |
| homeostasis | function recovery after perturbation |
| plasticity | leaving a basin when the task genuinely changes |
| degeneracy | different mechanisms, same function |

**These are mathematical analogies motivating experiments.** No biological claim is made anywhere in
this program, and no result may be described using biological vocabulary without the analogy being
marked.

---

## 4. Primary hypothesis

> A representation-invariant, trajectory-aware basin quantity, measured **before training finishes**,
> predicts whether independently perturbed learning systems converge to the same functional
> phenotype — better than initialization curvature, sharpness, learning rate, parameter distance, or
> simple loss statistics.

Conjectural. The program's job is to destroy it.

---

## 5. Candidate measurements

Explored in Lane B; exactly one is frozen at Stage 6 before any held-out evaluation.

- **basin coherence** — fraction of perturbed realizations landing in the same functional region
- **pairwise coherence** — `Pr[ D_F(Π(S_T^i), Π(S_T^j)) < ε ]`, with ε sensitivity-analysed and the
  confirmatory value frozen
- **short-horizon functional contraction** — whether functional spread shrinks over a window
- **quotient Lyapunov exponent** `λ_Q` — growth rate of functional separation
- **committor** `q(s) = Pr_s[τ_B < τ_A]` — fate probability, on systems small enough to estimate it
- **quasipotential barrier** — Freidlin–Wentzell action along minimum-action paths, tiny systems only

## 6. Baselines every candidate must beat

`ρ₀` (the falsified DISCOVERY-001 quantity, retained as a baseline), early loss, loss decrease,
final-minus-early loss, learning rate, parameter distance, Hessian trace, sharpness, gradient norm,
and **cross-seed ensembling** — because "canalization is just ensembling" is a live null.

---

## 7. Falsifiers

DISCOVERY-002 dies if any of these hold:

1. basin coherence adds no predictive value beyond early or final loss;
2. the effect appears in only one toy family;
3. phenotype clustering depends strongly on arbitrary metric choice;
4. committor estimates are too unstable to reproduce;
5. quasipotential estimates are vacuous;
6. quotient metrics give no benefit over raw parameter metrics;
7. canalization reproduces ordinary ensembling;
8. developmental controllers are merely expensive optimizers;
9. regeneration disappears under matched compute;
10. a biology-inspired controller loses to ordinary gating;
11. finite-agent behaviour does not approach the claimed mean-field limit;
12. criticality boundaries do not generalize;
13. prior work already contains the whole result.

Every failure is preserved under its original identifier.

---

## 8. Prior-art position, stated up front

Every individual ingredient is assumed old: Waddington landscapes, canalization, Boolean gene
regulatory networks, cell-fate attractor models, neural cellular automata, developmental encodings,
hypernetworks, meta-learning, mean-field and McKean–Vlasov neural networks, contraction analysis,
transition-path theory, metastability, Freidlin–Wentzell theory, quotient geometry,
symmetry-compatible optimizers, persistent homology of loss landscapes.

Symmetry-compatible optimizer design is already an active area, so **"respect symmetry" is not a
contribution**. Canalization and attractor robustness are established biology, so **inspiration is
not novelty**. The only admissible novelty is a new machine-learning *result*.

---

## 9. Execution order

1. This document. ✅
2. **Explain DISCOVERY-001's failure**: track `ρ(t)` through training on P2's stable cells. If
   trajectories did *not* leave the initially unstable region, **abandon this explanation.**
3. Tiny nonlinear systems with computable basin structure.
4. Estimate attractors, basin coherence, contraction, `λ_Q`; compare against `ρ₀`.
5. Find whether one dynamic quantity predicts final fate.
6. Internal replication.
7. Freeze one predictor, hashed.
8. Untouched nonlinear family. Fail ⟹ stop and preserve.
9. Only on success: a qualitatively different equivalence family.
10. Developmental controller; damage/regeneration; plasticity; finite agents; mean-field limit.
11. Larger architectures last.

Stages 9 onward are **not authorized** by this document and require a separate amendment.

---

## 10. Standing constraints

- No quantum mathematics in DISCOVERY-002. That branch is deferred and would need its own record.
- No claim may cite a Lane B artifact.
- The simplest boring explanation is tested first, every time.
- Compute is matched before any comparison is reported.
