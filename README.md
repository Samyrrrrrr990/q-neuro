# Q-Neuro

[![CI](https://github.com/Samyrrrrrr990/q-neuro/actions/workflows/ci.yml/badge.svg)](https://github.com/Samyrrrrrr990/q-neuro/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-2E74B5.svg)](LICENSE)
[![Research: synthetic only](https://img.shields.io/badge/research-synthetic%20only-148F88.svg)](docs/MODEL_CARD.md)
[![Grand benchmark: sealed](https://img.shields.io/badge/QN--GRAND--001-sealed-D8584A.svg)](experiments/results/QN-GRAND-001/preflight.json)

Q-Neuro is a falsification-first research repository for testing structured complex-valued and
real-valued recurrent models on synthetic sequential tasks. Its central result is a reversal: an
apparent complex-valued robustness advantage disappears when the comparator is replaced by an
exact real implementation of the same computation and by stronger real-model controls.

> **SYNTHETIC, NONCLINICAL RESEARCH ONLY.** Q-Neuro has not been evaluated on patients and is not a
> medical device, diagnostic system, clinical decision-support tool, or demonstration of physical
> quantum computation.

## Main result

| Evidence stage | Registered result | Interpretation |
|---|---:|---|
| Historical NeuroWorld comparison | +0.0602 top-1 vs two-channel real | Positive but comparator-sensitive |
| Reduced discovery | 0/2,880 positive effects; mean −0.03695 vs best real | No support for an intrinsic complex advantage; 52% of cells won by genuinely distinct real models |
| Untouched-family confirmation | 0/1,920 positive effects; mean −0.009158 | Non-positive result transfers to four new families; see the composition amendment below |
| Exact-real control | Top-1 matches in all 1,920 held-out cells | Implemented complex computation is reproducible in real arithmetic |
| Frozen quantitative law | R² −30.94; MAE 0.03126 | Failed magnitude transfer; retained as a negative result |
| QN-GRAND-001 | Blocked before execution; sealed | Six mandatory readiness gates remain unmet |

The held-out family/world/seed hierarchical bootstrap interval for the complex-minus-best-real
effect is [−0.01325, −0.00457]. These reduced studies are intentionally outcome-ineligible: they
do not replace the deferred full protocol and cannot support clinical, universal-superiority, or
grand-confirmatory claims.

### Amendment 001 — how the 1,920 held-out cells are composed

Every number above remains numerically correct and no result artifact has been changed. The
*interpretation* of the held-out count is amended, because the cells are not 1,920 independent
architecture comparisons:

| QN-000042 best-real winner | Cells | Mean effect | Exactly zero |
|---|---:|---:|---:|
| `exact_real_block_operator` | 1,478 (77%) | +0.00000 | 1,478 (100%) |
| `real_polar_operator` | 442 (23%) | −0.03978 | 60 (14%) |

In 77% of held-out cells the selected best-real model *is* the exact-real implementation. That
implementation is not an independent rival architecture: it evaluates the same modeled computation
and, in the current code, shares the same real parameter coordinates as the complex model (see
[`docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md`](docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md)). Those
1,478 exact zeros are **equivalence-induced**, and must not be presented as 1,478 independent wins
of a distinct real architecture.

The informative architecture-level reversal is the remaining 442 cells, where `real_polar_operator`
— a genuinely different model — beats the complex operator by −0.0398 on average.

**QN-000040 is the healthier heterogeneous-control result** and should carry more of the argument:
`state_space` (637 cells, −0.0906), `real_polar_operator` (485, −0.0469), and `gru` (367, −0.0707)
together win 52% of discovery cells, with `exact_real_block_operator` taking the remaining 1,391.

This amendment strengthens the negative result rather than weakening it. The
intrinsic-complex-arithmetic claim still fails on two independent grounds: exact realification
reproduces the implemented computation, **and** genuinely distinct real controls outperform it.
What changes is that the headline must distinguish equivalence-induced zeros from independent
architecture wins.

![Falsification phase](research/figures/generated/falsification_phase.png)

## Research program documentation

Work after the v1.0.0 release — the equivalence compiler (Q-Neuro 2.0) and the architecture program
(Q-Neuro 3.0) — is documented in three companion documents. **Fifteen frozen, hashed, prospective
predictions were opened and one passed as written**; 33 failures are preserved with mechanisms.

The one that passed, `QNEURO3-NICHE-P1`, is deliberately small and carries its own ceiling:

> On workloads with a deep worst case and heavy-tailed difficulty, halting on a supervised predicate
> attains the **optimal** per-example allocation and gives a **2.8–4.9× wall-clock inference saving
> at batch 1**, at matched accuracy and parameters — and **loses that advantage above batch ≈ 32
> under lockstep execution**, because a lockstep batch cannot exit until its slowest member does.

Reproduce it standalone, with the frozen hash verified from disk before anything is scored:

```bash
make reproduce-q3-niche
```

**The ceiling turned out to belong to the runtime, not the mechanism.** Under active-set compaction
— prior art, claimed as a baseline — the same models recover from 0.97× to **1.95×** at batch 256 on
an expensive core. Two further predictions were frozen and both failed: the cost model put the
crossover at batch 45 when it is below 16, and the recovery does not transfer to a core eight times
cheaper per step, where it reaches only 1.07×. So the second boundary is: compaction pays when
per-step cost is large relative to gather cost.

| Document | For |
|---|---|
| [**Paper**](docs/PAPER.md) | the short version — problem, principle, what was frozen, why it failed |
| [**Technical breakdown**](docs/TECHNICAL_BREAKDOWN.md) | reproduction — derivations, hyperparameters, controls, commands |
| [**Monograph**](docs/MONOGRAPH.md) | the full narrative, including every failure and the beautiful results that had to be killed |

These document ongoing research and are **not** part of the frozen v1.0.0 publication package below.

## Publication package

The complete v1.0.0 results article, **Exact Real Controls Overturn an Apparent Complex-Valued
Robustness Advantage**, is available as [Word](paper/qneuro.docx), [PDF](paper/qneuro.pdf), and
modular [LaTeX](paper/main.tex). The editable manuscript is 25 pages, accessibility-audited, and
generated from canonical Markdown plus registered artifacts.

- [Manuscript build and artifact map](paper/README.md)
- [Detailed result record](RESULTS.md)
- [Next-phase preregistration](docs/PREREGISTRATION_NEXT_PHASE.md)
- [Frozen candidate law](research/laws/FROZEN_CANDIDATE_001.json)
- [Machine-readable claims](research/claims.json)
- [Failure ledger](research/failures.json)
- [Interactive evidence dashboard](dashboard/index.html)

The manuscript has not undergone journal peer review, independent replication, or external
statistical audit. Publication files are submission-ready artifacts, not a guarantee of acceptance.

## Reproduce

```bash
uv sync --extra dev --extra paper --frozen
uv run pytest -q
make dashboard
uv run python paper/build_manuscript.py
uv run python scripts/verify_release.py
```

Cached-artifact verification checks the registered result chain without retraining thousands of
models. Full experimental reruns remain separate, compute-intensive operations and must not be
confused with independent replication. See [`REPLICATION.md`](REPLICATION.md) for the exact
boundary and commands.

## Repository map

```text
experiments/             immutable run outputs, configs, registry, and runners
q_neuro/                 models, synthetic generators, training, and evaluation code
research/                preregistrations, analyses, frozen law, claims, failures, and figures
paper/                   canonical manuscript source plus DOCX/PDF/LaTeX outputs
dashboard/               static evidence audit built from versioned artifacts
tests/                   invariant, analysis, registry, safety, and release tests
release/                 checksummed publication manifest and replication report
```

## Scientific interpretation

The exact-real mapping falsifies representational uniqueness for the implemented linear complex
operator. It does not prove that every complex network, nonlinear complex activation, optimizer,
or task is inferior. The negative result is narrower and more useful: under the executed synthetic
profiles, the claimed robustness advantage does not survive the strongest tested real controls.

Earlier positive, ablation, probe, training-law, halting, and trajectory studies remain preserved
in the immutable history. They are context for how the claim evolved, not evidence that overrides
the prospective falsification decision.

## License

Code is released under the [MIT License](LICENSE). Research artifacts are provided for
reproducibility and critical review without clinical warranty or fitness claims.
