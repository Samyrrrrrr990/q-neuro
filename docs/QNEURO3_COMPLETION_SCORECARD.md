# Q-Neuro 3.0 — Final completion scorecard

Every score links to evidence. Scores are not assigned because the programme wants to finish.

**Verdict: Q-Neuro 3.0 does NOT clear all ten gates at the required thresholds. It terminates on the
second condition — a defensible boundary is established — not the first.** Three dimensions fall
short and are itemised below rather than rounded up.

---

## The table

| Dimension | Required | Scored | Evidence |
|---|---|---|---|
| Scientific rigor | 10/10 | **9/10** | 16 frozen hashed predictions, one pass; kill conditions applied even when the failure flattered us; branch controls that killed our own branches; shortcut audits on every task; cross-policy equivalence verified, not assumed. **Missing point:** the PonderNet baseline collapsed and is probably my implementation's fault, and there is no independent reimplementation (Gate G never started). |
| Reproducibility | 10/10 | **9/10** | `make reproduce-q3` verifies all ten Q3 prediction hashes *from disk* plus five invariants; `make reproduce-q3-niche` re-derives the confirmed result end to end and passed all four clauses; environment, seeds and the dataset archive sha256 are recorded. **Missing point:** one machine, no containerised environment, ~90 min full run. |
| Reliability | ≥9/10 | **9/10** | 10/10 seeds at 1.0000 accuracy *and* 1.0000 halt accuracy; zero catastrophic runs; ECE 0.0018–0.0021; 9/12 hyperparameter configurations perfect, with all three failures at a single under-trained learning rate. **Missing point:** measured in depth on one family. |
| Engineering usefulness | ≥9/10 | **7/10 — FAILS** | Real gains exist: 3.60× batch-1, 1.86× batch-256 under compaction, 3.4× less activation memory, a planner that matches the measured optimum at 8 of 9 batch sizes. **But the mechanism loses to ACT on real data** (0.9006 at 3.61 chunks against 0.8112 at 2.39), so its usefulness is confined to tasks supplying a halt target. |
| Baseline quality | 10/10 | **8/10 — FAILS** | ACT, PonderNet, confidence exit, fixed depth, per-step selection, explicit latch, mean pooling, static-width control, exact-real control, and four execution policies. **Missing points:** PonderNet collapsed to 0.5220 and I treat that as my implementation shortcoming; no early-exit transformer, which was not task-appropriate here. |
| Claim precision | 10/10 | **10/10** | 11 claims, each with assumptions, effect size, external validity, a falsifier, novelty status, replication status and uncertainty; adversarial reviewer passes; scope corrections recorded rather than silently applied (`QNEURO3-SCOPE-CORRECTION-001`). |
| Prior-art integrity | 10/10 | **10/10** | `docs/PRIOR_ART_RUNTIME.md` and `docs/GATE4_NOVELTY_AUDIT.md`; prior art run as Gate 1 *before* implementation; no mechanism claimed as novel; every baseline that beats us reported as the headline. |
| M2 optimization | ≥9/10 | **8/10 — FAILS** | Full sweep 1→256 with median, p95, throughput, rows/example and analytic peak memory; measured CPU↔MPS crossover; hardware profile detection; five named profiles; a constraint solver that refuses rather than relaxes. **Missing points:** CPU only — MPS is not exploited at these shapes — and no kernel fusion or compilation. |
| Breadth | ≥8/10 or niche-limited | **explicitly niche-limited** | Three synthetic families and one real dataset. The mechanism wins only where a halt target exists. Stated as the headline, not a caveat. |
| Novel contribution | clearly identified | **identified** | The measured boundaries and the falsification methodology — **not** the architecture. See §4 below. |

---

## The three shortfalls, stated plainly

**Engineering usefulness (7/10 against ≥9).** The runtime characterisation is useful and the
latency numbers are real. The *architecture* is not clearly better than a 2016 method. On the one
real dataset tested, ACT reaches 0.9006 using 3.61 of 16 chunks while supervised halting reaches
0.8112 using 2.39 — and ACT needs no teacher and 54% of the training time. Improving this would
require a mechanism that beats ACT where no halt target exists, and no such mechanism survived
Gate 5.

**Baseline quality (8/10 against 10).** ACT was implemented faithfully and beat us. PonderNet
collapsed to 0.5220 with 16/16 chunks, which is almost certainly a defect in my implementation
rather than a property of the method. A correctly tuned PonderNet would only worsen our standing,
so the error does not favour us — but a 10/10 baseline suite would have it working.

**M2 optimization (8/10 against ≥9).** Everything is measured on CPU. The measured CPU↔MPS
crossover is 65,536 elements and these models never reach it, so MPS is correctly unused — but
"correctly unused" is not "optimised for", and no compilation or kernel fusion was attempted.

---

## Termination conditions

| # | Condition | Status |
|---|---|---|
| 1 | Final architecture frozen, no more search | **Met** — Gate 5 branches A, B killed by their own controls; C–F recorded as not attempted with reasons |
| 2 | Every surviving component has a successful ablation | **Met** — first-arrival vs mixture vs commit; halt-step vs final-state readout; normalisation on/off; early exit on/off; static vs routed width |
| 3 | Every major claim has a falsifier | **Met** — 11 claims, each with an explicit falsifier |
| 4 | Every major positive claim has prospective evidence | **Met** — `QNEURO3-NICHE-P1` passed all four clauses on an untouched family |
| 5 | Strong prior art explicitly acknowledged | **Met** — two audit documents; no mechanism claimed as novel |
| 6 | Strong adaptive-compute baselines beaten, or the niche stated | **Met by stating the niche** — ACT and confidence exit beat us on real data; the niche is tasks supplying a halt target |
| 7 | M2 implementation optimised and profiled | **Partially met** — profiled thoroughly, optimised only at the policy level |
| 8 | Reproduction package passes | **Met** — hashes, invariants and the end-to-end niche reproduction all pass |
| 9 | Adversarial claim audit passes | **Met** |
| 10 | No known high-value experiment could reverse the headline | **Met** — the headline is already the negative one; the branches that could have changed it were tested and killed |

---

## Why the programme is scientifically complete

The headline conclusion is negative and *robust to further work in the obvious directions*:

- It cannot be reversed by better baselines — the baselines already beat us.
- It cannot be reversed by more architecture search — the three testable branches were killed by
  their own controls, and two of them by a *simpler* model at equal cost.
- It cannot be reversed by better engineering — the runtime work already recovered what was
  recoverable, and its boundary was frozen and confirmed.

What remains open is listed in the paper and would not move the headline: a correct predictive cost
model (attempted, failed, not patched), a second hardware regime (no machine available), a properly
tuned PonderNet (would worsen our position), and independent reimplementation.

**Stopping here is the correct scientific decision, not fatigue.**
