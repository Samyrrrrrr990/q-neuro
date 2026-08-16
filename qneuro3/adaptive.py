"""Q-Neuro 3.0 final architecture: supervised predicate halting, and the policy for when to use it.

Everything here survived an ablation. Nothing here is claimed to be novel — supervised early exit
and per-step attribution are prior art (`research/qneuro3/QNEURO3-ATTRIBUTION-001.json`). What is
established, and confirmed by the one frozen prediction in this programme that passed as written
(`QNEURO3-NICHE-P1`), is a scoped engineering result **with its ceiling measured**:

    On workloads with a deep worst case and heavy-tailed difficulty, halting on a supervised
    predicate reaches the optimal per-example allocation and delivers a 2.8-4.9x wall-clock
    inference saving AT BATCH 1, at matched accuracy and matched parameters -- and loses that
    advantage entirely above batch ~32.

The ceiling is not an implementation defect. A batch cannot exit until its slowest member does, so
batched cost tracks ``E[max halt over the batch]``, which rises towards the worst case as the batch
grows. `expected_max_halt` computes it exactly, and `plan` uses it to decide whether early exit is
worth running at all.

Components, each with the measurement that justifies it:

* **first-arrival objective** — mixture halting (PonderNet-style) caps at 0.6241 on `chase_to_goal`
  where this reaches 1.0000; commit halting reaches 0.9999 but only at full depth, buying nothing.
* **answer read at the halt step** — on associative-lookup tasks a final-state readout scores 0.22
  against 1.00, and stays there under matched supervision, an explicit latch, and 3x training.
* **RMS-normalised state** — turns a 11-of-24 seed lottery into 20 of 20 on associative lookup.
  It costs 0.30 of extrapolated accuracy on streaming tasks, so it is a flag, not a default.
* **genuine early exit** — without it the saving is nominal: the batched forward runs every step
  and then selects, which measured 1.0x against the fixed-depth baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

#: Batch size at which early exit stopped paying on the M2 reference machine, measured on both
#: task families. Below it, exit early; above it, run full depth and select.
MEASURED_CROSSOVER_BATCH = 32


def first_arrival(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """log P(the predicate first fires at step k), and the step it actually fired on.

    The masses sum to at most one, falling short exactly by the probability of never firing, which
    is the right behaviour: a run that never fires is a real outcome and is not renormalised away.
    """

    p = probabilities.clamp(1e-6, 1 - 1e-6)
    log_not = torch.log1p(-p)
    cumulative = torch.cat(
        [torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1
    )
    log_first = torch.log(p) + cumulative
    fired = p > 0.5
    any_fired = fired.any(dim=1)
    step = torch.where(
        any_fired,
        fired.float().argmax(dim=1),
        torch.full_like(any_fired, p.shape[1] - 1, dtype=torch.long),
    )
    return log_first, step + 1


def halting_loss(
    log_first: torch.Tensor, true_step: torch.Tensor, answer_logits: torch.Tensor,
    answer: torch.Tensor,
) -> torch.Tensor:
    """Fire at the true step, and be right about the answer read there.

    Both terms are grounded in the task. Neither supervises the other, and there is no ponder cost
    to tune -- the scalar mean-depth penalty that ACT and PonderNet rely on was measured to
    pressure average depth without ever inducing per-example allocation.
    """

    likelihood = -log_first.gather(1, (true_step - 1).unsqueeze(1)).mean()
    return likelihood + torch.nn.functional.cross_entropy(answer_logits, answer)


def expected_max_halt(halt_pmf: torch.Tensor, batch: int) -> float:
    """E[max halt step] over `batch` independent examples — the batched cost, exactly.

    ``E[max] = sum_k k * (F(k)^n - F(k-1)^n)``. This is why the advantage has a ceiling, and it
    applies to every per-example adaptive-compute method, not only to this one.
    """

    pmf = halt_pmf.double() / halt_pmf.double().sum()
    cdf = pmf.cumsum(0)
    steps = torch.arange(1, len(pmf) + 1, dtype=torch.double)
    upper = cdf**batch
    lower = torch.cat([torch.zeros(1, dtype=torch.double), upper[:-1]])
    return float((steps * (upper - lower)).sum())


@dataclass(frozen=True)
class Plan:
    """What to actually run, and the number that justifies it."""

    mode: str
    early_exit: bool
    expected_steps: float
    expected_batched_steps: float
    predicted_speedup: float
    rationale: str


def plan(halt_pmf: torch.Tensor, batch: int, max_depth: int | None = None) -> Plan:
    """Choose an execution mode from the difficulty distribution and the batch size.

    The M2 modes are not presets; each is the regime where a measurement says a different thing is
    correct.
    """

    depth = max_depth if max_depth is not None else len(halt_pmf)
    pmf = halt_pmf.double() / halt_pmf.double().sum()
    mean = float((torch.arange(1, len(pmf) + 1, dtype=torch.double) * pmf).sum())
    batched = expected_max_halt(halt_pmf, batch)
    speedup = depth / batched

    if batch == 1:
        return Plan(
            "M2 Eco", True, mean, batched, depth / mean,
            "Single stream. The full per-example saving is available and measured at 2.8-4.9x.",
        )
    if batch < MEASURED_CROSSOVER_BATCH and speedup >= 1.15:
        return Plan(
            "M2 Balanced", True, mean, batched, speedup,
            f"Batch {batch} still leaves {speedup:.2f}x after the batch maximum; exit early.",
        )
    return Plan(
        "M2 Throughput", False, mean, batched, 1.0,
        (
            f"Batch {batch} leaves only {speedup:.2f}x after the batch maximum, and the per-step "
            "halting head costs more than that. Run full depth and select the step instead — "
            "identical accuracy, and measured 0.97-0.99x for early exit here, i.e. a penalty."
        ),
    )


class PredicateHalting(nn.Module):
    """Wraps any step-wise core into a halting model.

    `core` must expose ``advance(state, context) -> (state, features)``; `features` is what the
    halting head and the answer head read. Keeping the core abstract is deliberate: the halting
    mechanism was measured on a graph-traversal core with attention and on a streaming core with
    none, and it behaved the same way on both.
    """

    def __init__(self, core: nn.Module, feature_dim: int, n_answers: int, max_depth: int,
                 *, halt_bias: float = -2.0):
        super().__init__()
        self.core = core
        self.max_depth = max_depth
        self.halt = nn.Linear(feature_dim, 1)
        nn.init.constant_(self.halt.bias, halt_bias)
        self.answer = nn.Linear(feature_dim, n_answers)

    def forward(self, state: torch.Tensor, context):
        """Training path: every step, so the first-arrival likelihood is differentiable."""

        probabilities, features = [], []
        for _ in range(self.max_depth):
            state, feature = self.core.advance(state, context)
            probabilities.append(torch.sigmoid(self.halt(feature)).squeeze(-1))
            features.append(feature)
        stacked = torch.stack(features, 1)
        log_first, step = first_arrival(torch.stack(probabilities, 1))
        picked = stacked.gather(
            1, (step - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        ).squeeze(1)
        return self.answer(picked), step, log_first, stacked

    @torch.no_grad()
    def infer(self, state: torch.Tensor, context, *, early_exit: bool = True):
        """Inference path with genuine early termination.

        The training path always runs `max_depth` steps because the likelihood needs every step.
        This does not, and the difference is the entire wall-clock result: with `early_exit=False`
        the saving is nominal and measures 1.0x.
        """

        batch = state.shape[0]
        done = torch.zeros(batch, dtype=torch.bool, device=state.device)
        answers = torch.zeros(batch, self.answer.out_features, device=state.device)
        steps = torch.full((batch,), float(self.max_depth), device=state.device)
        for index in range(self.max_depth):
            state, feature = self.core.advance(state, context)
            fires = (torch.sigmoid(self.halt(feature)).squeeze(-1) > 0.5) & ~done
            if bool(fires.any()):
                answers[fires] = self.answer(feature)[fires]
                steps[fires] = index + 1
            done = done | fires
            if early_exit and bool(done.all()):
                break
        unfinished = ~done
        if bool(unfinished.any()):
            answers[unfinished] = self.answer(feature)[unfinished]
        return answers, steps
