"""Execution policies for heterogeneous-depth models, and the cost model that chooses between them.

`QNEURO3-CEILING-001` measured a straggler ceiling: under **lockstep** batching every example is
advanced until the last one halts, so executed work is `n · max_i d_i` while useful work is
`Σ_i d_i`. `QNEURO3-SCOPE-CORRECTION-001` narrows that result to the execution policy it was
measured on and opens the question this module answers: how much of the gap a better runtime
recovers, on hardware where a step is small and compaction is not free.

**None of these policies is novel and none is claimed to be.** Active-set compaction is the standard
early-exit loop and is what MoE dispatch does per layer; continuous batching is iteration-level
scheduling as deployed in LLM serving; bucketing by length predates transformers. See
`docs/PRIOR_ART_RUNTIME.md`. They are implemented here as baselines so the measured boundary has
something honest to be measured against.

The one thing bucketing needs that the prior art does not supply: halt depth is the *output* of the
computation, not a readable property of the input, so bucketing requires a depth predictor.

Every policy must produce identical answers. `verify_equivalence` is the check, and it is run in
the test suite rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class SteppableModel(Protocol):
    """What a runtime needs from a model, and nothing more.

    Keeping this minimal is deliberate: the policies below were measured on an attention-based
    graph-traversal core and an attention-free streaming core, and the runtime must not know which.
    """

    max_depth: int

    def init_state(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-example tensors, all with the batch as leading dimension."""

    def step(
        self, state: dict[str, torch.Tensor], position: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """One iteration. Returns the new state, halt probability, and answer logits."""


def _index(state: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: tensor[keep] for name, tensor in state.items()}


@dataclass
class Execution:
    """What a run cost, in units a cost model can use."""

    answers: torch.Tensor
    steps: torch.Tensor
    example_steps: int
    """Total rows advanced, summed over iterations. The quantity a runtime can actually reduce."""
    iterations: int
    """Number of kernel-launch rounds. Lockstep and compaction differ in rows per round, not rounds."""
    compactions: int


@torch.no_grad()
def lockstep(model: SteppableModel, batch: dict[str, torch.Tensor]) -> Execution:
    """Advance every row until the last one halts. The confirmed baseline."""

    state = model.init_state(batch)
    n = next(iter(state.values())).shape[0]
    done = torch.zeros(n, dtype=torch.bool)
    answers: torch.Tensor | None = None
    steps = torch.full((n,), float(model.max_depth))
    example_steps = iterations = 0

    for index in range(model.max_depth):
        position = torch.full((n,), index, dtype=torch.long)
        state, halt, logits = model.step(state, position)
        example_steps += n
        iterations += 1
        if answers is None:
            answers = torch.zeros(n, logits.shape[-1])
        fires = (halt > 0.5) & ~done
        if bool(fires.any()):
            answers[fires] = logits[fires]
            steps[fires] = index + 1
        done = done | fires
        if bool(done.all()):
            break
    if answers is not None and bool((~done).any()):
        answers[~done] = logits[~done]
    return Execution(answers, steps, example_steps, iterations, 0)


@torch.no_grad()
def compacted(model: SteppableModel, batch: dict[str, torch.Tensor], *, every: int = 1) -> Execution:
    """Drop halted rows and continue on the survivors. Useful work becomes `Σ d_i`.

    `every` controls how often compaction runs. Compacting on every iteration minimises executed
    rows and maximises gather and synchronisation cost; the optimum is not obviously 1 at small
    model width, which is the whole point of measuring it.
    """

    state = model.init_state(batch)
    n = next(iter(state.values())).shape[0]
    alive = torch.arange(n)
    # With `every > 1` a fired row keeps being advanced until the next compaction, and its halt
    # probability stays above threshold, so without this mask it would overwrite its own answer
    # with a later step's logits. The equivalence check caught exactly that.
    fired = torch.zeros(n, dtype=torch.bool)
    answers: torch.Tensor | None = None
    steps = torch.full((n,), float(model.max_depth))
    example_steps = iterations = compactions = 0

    for index in range(model.max_depth):
        live = alive.shape[0]
        if live == 0:
            break
        position = torch.full((live,), index, dtype=torch.long)
        state, halt, logits = model.step(state, position)
        example_steps += live
        iterations += 1
        if answers is None:
            answers = torch.zeros(n, logits.shape[-1])
        fires = (halt > 0.5) & ~fired[alive]
        if bool(fires.any()):
            answers[alive[fires]] = logits[fires]
            steps[alive[fires]] = index + 1
            fired[alive[fires]] = True
        if index == model.max_depth - 1:
            survivors = ~fired[alive]
            if bool(survivors.any()):
                answers[alive[survivors]] = logits[survivors]
            break
        if (index + 1) % every == 0:
            keep = ~fired[alive]
            if not bool(keep.all()):
                state = _index(state, keep)
                alive = alive[keep]
                compactions += 1
    return Execution(answers, steps, example_steps, iterations, compactions)


@torch.no_grad()
def bucketed(
    model: SteppableModel,
    batch: dict[str, torch.Tensor],
    predicted_depth: torch.Tensor,
    *,
    buckets: int = 4,
) -> Execution:
    """Group rows by PREDICTED halt depth, run each group lockstep.

    The straggler cost is `max − mean` *within a batch*. Partitioning by predicted depth shrinks
    that spread inside each group without needing a dynamic scheduler at all. It is only as good as
    the predictor, and halt depth is not readable from the input — which is what separates this from
    ordinary length bucketing.
    """

    n = predicted_depth.shape[0]
    order = torch.argsort(predicted_depth)
    answers: torch.Tensor | None = None
    steps = torch.zeros(n)
    example_steps = iterations = 0

    for chunk in torch.chunk(order, buckets):
        if chunk.numel() == 0:
            continue
        part = lockstep(model, {name: tensor[chunk] for name, tensor in batch.items()})
        if answers is None:
            answers = torch.zeros(n, part.answers.shape[-1])
        answers[chunk] = part.answers
        steps[chunk] = part.steps
        example_steps += part.example_steps
        iterations += part.iterations
    return Execution(answers, steps, example_steps, iterations, 0)


@torch.no_grad()
def continuous(
    model: SteppableModel, batch: dict[str, torch.Tensor], *, width: int
) -> Execution:
    """Iteration-level scheduling: finished rows leave, queued rows enter mid-flight.

    Holds the in-flight set at `width` rows so every iteration does constant work. This is the
    policy that recovers throughput in LLM serving; it needs a queue to backfill from, which the
    single-stream niche does not have. Implemented so the comparison is fair rather than assumed.
    """

    n = next(iter(batch.values())).shape[0]
    answers: torch.Tensor | None = None
    steps = torch.full((n,), float(model.max_depth))
    example_steps = iterations = compactions = 0

    queue = list(range(n))
    inflight = torch.tensor(queue[:width], dtype=torch.long)
    queue = queue[width:]
    state = model.init_state({k: v[inflight] for k, v in batch.items()})
    age = torch.zeros(inflight.shape[0], dtype=torch.long)

    while inflight.numel() > 0:
        state, halt, logits = model.step(state, age)
        example_steps += inflight.shape[0]
        iterations += 1
        age = age + 1
        if answers is None:
            answers = torch.zeros(n, logits.shape[-1])
        fires = (halt > 0.5) | (age >= model.max_depth)
        if bool(fires.any()):
            answers[inflight[fires]] = logits[fires]
            steps[inflight[fires]] = age[fires].float()
            keep = ~fires
            state = _index(state, keep)
            inflight = inflight[keep]
            age = age[keep]
            compactions += 1
            admit = min(width - inflight.shape[0], len(queue))
            if admit > 0:
                fresh = torch.tensor(queue[:admit], dtype=torch.long)
                queue = queue[admit:]
                new_state = model.init_state({k: v[fresh] for k, v in batch.items()})
                state = {k: torch.cat([state[k], new_state[k]], dim=0) for k in state}
                inflight = torch.cat([inflight, fresh])
                age = torch.cat([age, torch.zeros(admit, dtype=torch.long)])
    return Execution(answers, steps, example_steps, iterations, compactions)


def verify_equivalence(
    reference: Execution, other: Execution, *, tolerance: float = 1e-5
) -> None:
    """A runtime that changes the answer is not a runtime, it is a different model."""

    if not torch.equal(reference.steps, other.steps):
        disagreements = int((reference.steps != other.steps).sum())
        raise ValueError(f"halt steps differ in {disagreements} rows")
    gap = float((reference.answers - other.answers).abs().max())
    if gap > tolerance:
        raise ValueError(f"answers differ by {gap:.3e}, tolerance {tolerance:.1e}")


# --- cost model -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeCost:
    """Per-unit costs, all in microseconds, measured rather than assumed.

    The order-statistic model in `qneuro3.adaptive` counts useful work only. This adds the terms
    that decide whether a dynamic policy actually wins:

        T = c_step · (example-steps) + c_launch · (iterations) + c_compact · (compactions)
    """

    step_per_example: float
    launch_per_iteration: float
    compaction: float

    def predict(self, execution: Execution) -> float:
        return (
            self.step_per_example * execution.example_steps
            + self.launch_per_iteration * execution.iterations
            + self.compaction * execution.compactions
        )

    def crossover_batch(self, mean_depth: float, max_depth: int) -> float:
        """Batch size at which compaction starts beating lockstep, from the cost terms alone.

        Lockstep executes about `n · max_depth` example-steps for large `n`, since `E[max]` saturates
        at `max_depth`; compaction executes `n · mean_depth` plus one compaction per iteration.
        Setting the two equal and solving for `n`:

            c_step · n · max_depth  =  c_step · n · mean_depth + c_compact · max_depth

        so compaction wins above

            n* = c_compact · max_depth / (c_step · (max_depth − mean_depth)).

        Below `n*` the gathers cost more than the rows they remove.
        """

        headroom = max_depth - mean_depth
        if headroom <= 0:
            return float("inf")
        return self.compaction * max_depth / (self.step_per_example * headroom)
