# Prior-art audit: runtimes for heterogeneous-depth execution

Performed **before** implementing anything, per the standing rule that prior art is Gate 1.
Question: if the straggler ceiling under lockstep batching is a runtime problem, does the runtime
solution already exist?

**Answer up front: yes, substantially. Active-set compaction and continuous batching are
established, and this programme claims no novelty for either.** The audit below records what each
approach actually does at the level of execution semantics, and ends with the one question that
survives it.

---

## 1. The approaches, compared on what they do rather than what they are called

| Approach | Execution semantics | Compaction | Synchronisation | Batch regrouping | Objective |
|---|---|---|---|---|---|
| **Lockstep / padded batch** | advance all `n` for `max_i d_i` steps | none | one all-done check per step (optional) | none | simplicity |
| **Active-set compaction** | after each step, drop halted rows; continue on the survivors | gather/index_select on the state each step | per-step host sync to learn who halted | none | total work → `Σ d_i` |
| **Bucketing / length grouping** | partition inputs into groups of similar depth, run each group lockstep | none within a bucket | per-bucket | offline, before execution | reduce `max−mean` within a batch |
| **Continuous batching** | finished sequences leave, queued ones enter mid-flight | slot reuse | per-step scheduler pass | online, every step | throughput under a request stream |
| **Early-exit networks** (BranchyNet, DeeBERT, FastBERT) | per-sample exit at a classifier head | usually none at inference; often batch-1 deployment | — | — | latency |
| **ACT / PonderNet** | ponder distribution, mixture or sampled halt | none — the reference implementations run all steps | — | — | accuracy/compute trade-off in *training* |
| **Routed MoE** | per-token expert dispatch | permutation/gather into per-expert contiguous buffers | all-to-all | tokens regrouped by expert every layer | throughput |
| **Straggler-aware scheduling** (distributed) | backup tasks, work stealing | — | — | task reassignment | tail latency |
| **Dynamic control-flow runtimes** (`tf.while_loop`, TorchScript loops, XLA dynamic shapes) | data-dependent trip counts | recompilation or padding to buckets | — | — | expressing the program at all |

### 1.1 Where the obvious solution already is

**Active-set compaction is not new.** It is the standard implementation of any per-example
early-exit loop, it is what MoE dispatch does at every layer (gather tokens into contiguous
per-expert buffers, scatter back), and it is what sequence-packing does for variable-length
attention. `sum_i d_i` rather than `n · max_i d_i` is the well-known target.

**Continuous batching is not new.** Orca-style iteration-level scheduling and its descendants in
production LLM serving (vLLM and similar) are exactly "finished examples leave, queued examples
enter, per-iteration". That literature is explicit that the gain comes from removing the
`max`-over-batch coupling.

**Bucketing by length is not new.** Length-bucketed batching predates transformers; it is standard
in sequence modelling, and NLP dataloaders have grouped by length for a decade.

So: **no novelty is available for the runtime mechanisms themselves**, and none is claimed. Each is
implemented here as a *baseline*, not as a contribution.

---

## 2. What each approach assumes, and where those assumptions break here

The mechanisms are known. Their *applicability* to this regime is not automatic, and this is where
the audit finds something.

**Continuous batching assumes a request stream.** It recovers throughput when there is a queue of
independent requests to backfill with. The niche this programme confirmed is *single-stream,
latency-sensitive, on-device* inference — batch 1 to a few — where there is often no queue to
backfill from, and where the metric is per-request latency rather than tokens/second. Continuous
batching does not address the case it was not built for.

**Compaction assumes compaction is cheap relative to a step.** That is true when a step is a large
matmul on a GPU. It is the open question when a step is a small matmul on a fanless M2 CPU, where a
gather plus a host synchronisation can cost more than the step it saves. Section 4 of
`docs/TECHNICAL_BREAKDOWN.md` already records a measurement where an early-exit check made things
*slower* — the effect is real at this scale.

**Bucketing assumes depth is known before execution.** For sequence length it is: you can read it
off the input. **For adaptive halting it is not** — the halt depth is the output of the computation.
Bucketing therefore needs a *predictor* of depth, and its usefulness is bounded by how good that
predictor is. That is a genuinely different problem from length bucketing, and it is not answered by
the length-bucketing literature.

**MoE dispatch assumes the routing is per-layer and shallow.** Tokens are regrouped every layer at
known cost. Here the depth is per-example and unbounded up to `max_depth`, so the analogue is closer
to variable trip counts than to per-layer routing.

**ACT and PonderNet do not address this at all.** Their reference implementations run the full
`max_depth` in the batched forward and use the halting distribution as a *loss-side* object. The
reported compute savings in that literature are typically expected-step counts, not wall-clock —
which is exactly the nominal-versus-realised distinction this programme measured at 1.0×.

---

## 3. What is left after the audit

Three questions survive, and none of them is "did we invent a runtime".

1. **Quantitative, not conceptual.** How much of the `n·max / Σ d` straggler waste does compaction
   actually recover *on this hardware at this model scale*, where per-step compute is small and
   compaction overhead is not negligible? A crossover is expected; its location is not known and is
   not predicted by anything in the prior art.

2. **A latency model with the overhead terms in it.** The order-statistic result
   (`QNEURO3-CEILING-001`) models useful work only. A runtime model must include compaction,
   synchronisation, memory movement and launch cost, and must predict *when dynamic execution beats
   lockstep* as a function of batch size, depth distribution, model width and per-step cost. That
   equation is the candidate contribution, and it is falsifiable.

3. **Depth prediction as a scheduling input.** Bucketing by *predicted* halt depth is the one
   approach that needs something the prior art does not supply, because halt depth is not readable
   from the input. Whether a cheap predictor exists, and whether bucketing on it recovers a useful
   fraction of the waste without a dynamic scheduler, is open.

**Novelty position, stated now so it cannot drift:** the runtime mechanisms are prior art and are
implemented as baselines. Any contribution here is (a) a measured boundary on constrained Apple
Silicon and (b) a predictive crossover equation, if it survives a frozen prospective test. Neither
is a claim to have invented adaptive computation or its scheduling.
