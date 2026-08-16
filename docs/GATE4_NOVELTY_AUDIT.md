# Gate 4 — Novelty audit: what is actually ours

Separates the final contribution into five categories, at the level of update equations, execution
semantics, training objective, scheduling policy and claimed use case. Similar terminology is not
compared; mechanisms are.

**Summary up front: no new mechanism and no new architecture survives. The contribution is the
methodology and the measured boundaries.** The directive anticipated this outcome explicitly, and
it is the honest one.

---

## 1. Known mechanism — claimed by others, used here as baselines

| Mechanism | Where it comes from | How it appears here |
|---|---|---|
| Adaptive computation time | Graves (2016) | Implemented as a baseline. **It beats us on real data** (0.9006 at 3.61 chunks vs our 0.8112 at 2.39). |
| Halting distribution with per-step readouts | PonderNet, Banino et al. (2021) | Implemented as a baseline. Our first-arrival objective is the same object; the difference is that ours is *supervised* by a task-supplied halt target. |
| Confidence-based early exit | BranchyNet, DeeBERT, FastBERT and the wider early-exit literature | Implemented as a baseline. **It beats us on real data at matched compute** (0.8747 vs 0.8112). |
| Position-selection readout | Pointer Networks (Vinyals et al. 2015); span scoring in extractive QA | Our "answer read at the selected step" is this idea applied to the time axis of a recurrence. |
| Deep supervision / auxiliary per-step losses | Lee et al. (2015) | Used to give the confidence baseline calibrated per-chunk heads, so it is not a strawman. |
| Early-exit distillation from a teacher's earliest correct decision | Established in early-exit and early time-series classification | Exactly how the HAR halt target is derived. Declared in the frozen record *before* the run. |
| Layer / RMS normalisation of a recurrent state | Ba et al. (2016); Zhang & Sennrich (2019) | The fix that took seed reliability from 11/24 to 20/20. Textbook. |
| Active-set compaction | The standard early-exit loop; MoE dispatch; sequence packing | Implemented as a baseline runtime. Removes the lockstep ceiling on an expensive core. |
| Continuous / iteration-level batching | Orca-style scheduling, as deployed in LLM serving | Implemented as a baseline runtime. |
| Length bucketing | Predates transformers; standard in sequence modelling | Implemented as a baseline runtime. |
| Conditional computation over hidden groups | Mixture-of-experts and dynamic-width routing | Tested as Gate 5B. **Killed by its own control** — a statically narrow model matches it at equal cost with 43% fewer parameters. |
| Complex-valued networks | Long-established; exact realification is elementary | Tested as Gate 5A. **Killed** — identical accuracy, halting and depth at matched real parameters. |

**Nothing in this table is claimed as a contribution.** Where a baseline beats us, that is stated as
the headline rather than buried.

---

## 2. New empirical characterization — the strongest surviving category

These are measurements, not mechanisms. Each is falsifiable and several were confirmed
prospectively.

**2.1 The lockstep straggler ceiling, with its analytic form.** A lockstep batch cannot exit until
its slowest member does, so its cost tracks `E[max halt over the batch] = Σ_k k·(F(k)ⁿ − F(k−1)ⁿ)`
rather than `E[halt]`. For `P(k) ∝ 0.8^k` on 1..32 the realisable saving falls from 6.43× at batch 1
to 1.09× at 1024. The order statistic is elementary; what is not standard is stating it *alongside*
a reported adaptive-compute speedup. Predicted on one task family and **confirmed unprompted on
another** (`QNEURO3-NICHE-P1`, N4).

**2.2 The compaction boundary.** Active-set compaction removes that ceiling only when per-step cost
is large relative to gather cost: measured 1.95× at 2.66 µs/example-step and 1.07× at 0.33. Frozen
as a transfer prediction and **failed** (`QNEURO3-RUNTIME-P2`), which is how the boundary was
located.

**2.3 The scope condition on supervised halting.** It attains optimal allocation where the task
supplies a halt target, and loses to ACT and to confidence exit where the target must be distilled.
Frozen and **failed** on real data (`QNEURO3-HAR-P1`). This is the single most consequential result
of the programme and it is negative.

**2.4 The silent-failure mode of adaptive compute.** A collapsed halting run reports a plausible,
non-degenerate step count. All thirteen failed runs in cycle 1 looked like working elastic models
delivering a 1.3–1.5× saving; only the accuracy column revealed they were wrong more than half the
time. Any adaptive-compute system whose halting signal is also its answer can fail this way.

**2.5 The nominal-versus-realised gap.** A 6.5× step-count saving measured **1.0×** in wall-clock,
because the batched forward runs every step and then selects. Step counts are not a compute claim.

**2.6 `d_free = max(0, P − g_arch − n(C−1))`.** From Q-Neuro 2.0: elementary rank-nullity plus a
correct symmetry count, which converted seven consecutive navigation failures into a predicted
consequence. Confirmed exactly in 9 of 9 cells including the transition at `n = 193`.

---

## 3. New predictive principle — attempted, failed, not claimed

`QNEURO3-RUNTIME-P1` froze `T = c_step·rows + c_launch·iterations + c_compact·compactions` with all
constants measured on the target family's raw forward, and predicted the compaction crossover at
batch 45. Measured: below 16. Accurate where compute dominates (1.0% error at batch 128), wrong
where overhead does (55% at batch 16).

**Kill condition applied. No predictive runtime equation is claimed, and the equation was not
patched and re-issued.**

---

## 4. New architecture — none

Every architectural candidate either reduces to prior art or was killed by its own control:

- adaptive halting → supervised early exit (prior art), and it loses on real data;
- per-step attribution → pointer/span selection (prior art), and the separation failed to generalise
  (`QNEURO3-TRANSFER-P1`);
- adaptive width → beaten by static narrowing (Gate 5B);
- complex fields → identical to the real control at matched parameters (Gate 5A);
- depth extrapolation → frozen and falsified (`QNEURO3-EXTRAP-P1`).

**Q-Neuro 3.0's final architecture is a small assembly of known parts.** It is defensible as
engineering and it is not an invention.

---

## 5. New systems contribution — none

Compaction, continuous batching and bucketing are all prior art and are implemented as baselines.
The planner that *selects among* them from a measured step cost is ordinary engineering, and its one
non-obvious input — the crossover location — comes from measurement rather than from a model,
because the model failed.

---

## 6. What a hostile expert should be told

> The mechanisms are known. On synthetic tasks that supply a ground-truth halt target, supervised
> halting attains the optimal per-example allocation and a large batch-1 wall-clock saving at matched
> accuracy, reliably (10/10 seeds). On a real dataset that does not supply one, it is beaten by ACT
> and by confidence-based early exit. The durable contributions are the measured boundaries — the
> lockstep ceiling, the compaction crossover, the scope condition, and the silent-failure mode — and
> the falsification discipline that produced them: sixteen frozen predictions, one pass.

If the importance of that is disputed, the dispute is legitimate. What should not be easy to find is
a place where the programme cheated itself.
