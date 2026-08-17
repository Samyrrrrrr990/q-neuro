"""Architectural variants of the arrival-halting model, each a single-variable intervention.

Lane B. Nothing here is a claim. See `docs/LANE_POLICY.md`.

`QNEURO3-Q3-VARIANCE-001` measured that Q3 reaches its result on roughly half of initialisation
seeds. `QNEURO3-Q4-P1` tried to repair that by supplying the missing positional representation and
failed: it removed the collapse mode and the good mode together.

Each variant below changes exactly one thing about Q3, so that a reliability difference is
attributable. The working hypothesis they test is that the failure is **depth-limited chain
following** -- the recurrent state stops carrying position after a few hops -- rather than a
halting-head failure. That hypothesis is checked directly by
`accuracy_by_distance`, which is the discriminating measurement:

* if failing runs are accurate at distance 1-3 and wrong beyond, the core is the problem;
* if they are uniformly wrong, the halting head is the problem.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qneuro3.elastic import Core, arrival_loss
from qneuro3.tasks import chase_to_goal

NODES = 24
WIDTH = 64
MAX_DEPTH = 8


def _first_arrival(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared halting arithmetic: log P(first firing at step k), and the realised halt step."""

    p = p.clamp(1e-6, 1 - 1e-6)
    log_not = torch.log1p(-p)
    cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
    log_first = torch.log(p) + cum
    fired = p > 0.5
    any_fired = fired.any(dim=1)
    first = torch.where(
        any_fired, fired.float().argmax(dim=1),
        torch.full_like(any_fired, p.shape[1] - 1, dtype=torch.long),
    )
    return log_first, first + 1


class ArrivalModel(nn.Module):
    """Q3 with three independently switchable interventions.

    * ``normalise`` -- RMS-normalise the state after each hop. Tests whether the state's scale
      drifts as depth grows, which would degrade a fixed-threshold halting decision.
    * ``goal_match`` -- let the arrival head see ``h * key(goal)`` as well as ``h``, so a linear
      head can compute an inner product against the goal's identity. Arrival is a comparison; this
      supplies the comparison rather than requiring it to be learned.
    * ``dense_halting`` -- swap the first-arrival objective for a per-step binary target
      ``[k == distance]``. Same architecture, same inference rule, denser gradient to the head.
    """

    def __init__(
        self,
        n_nodes: int = NODES,
        d: int = WIDTH,
        max_depth: int = MAX_DEPTH,
        *,
        halt_bias: float = -2.0,
        normalise: bool = False,
        goal_match: bool = False,
        dense_halting: bool = False,
        goal_node: int = 0,
    ):
        super().__init__()
        self.core = Core(n_nodes, d, 1)
        self.max_depth = max_depth
        self.normalise = normalise
        self.goal_match = goal_match
        self.dense_halting = dense_halting
        self.goal_node = goal_node
        self.arrive = nn.Linear(2 * d if goal_match else d, 1)
        nn.init.constant_(self.arrive.bias, halt_bias)

    def _halt_input(self, h: torch.Tensor) -> torch.Tensor:
        if not self.goal_match:
            return h
        goal = self.core.key.weight[self.goal_node].unsqueeze(0).expand_as(h)
        return torch.cat([h, h * goal], dim=-1)

    def forward(self, perm: torch.Tensor, start: torch.Tensor):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        probs = []
        for _ in range(self.max_depth):
            h = self.core.advance(h, ctx)
            if self.normalise:
                h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
            probs.append(torch.sigmoid(self.arrive(self._halt_input(h))).squeeze(-1))
        p = torch.stack(probs, 1)
        log_first, steps = _first_arrival(p)
        return log_first, steps, p

    def loss(self, p: torch.Tensor, log_first: torch.Tensor, distance: torch.Tensor):
        if not self.dense_halting:
            return arrival_loss(log_first, distance)
        target = torch.zeros_like(p)
        target.scatter_(1, (distance - 1).unsqueeze(1), 1.0)
        return torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), target)


VARIANTS: dict[str, dict[str, bool]] = {
    "V0_baseline": {},
    "V1_normalise": {"normalise": True},
    "V2_goal_match": {"goal_match": True},
    "V3_dense_halting": {"dense_halting": True},
    "V4_match_dense": {"goal_match": True, "dense_halting": True},
}


def train_and_evaluate(
    name: str,
    seed: int,
    *,
    epochs: int = 8,
    train_batches: int = 500,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
) -> dict[str, Any]:
    train = [
        chase_to_goal(batch_size, NODES, MAX_DEPTH, seed=1000 + i) for i in range(train_batches)
    ]
    validation = [chase_to_goal(256, NODES, MAX_DEPTH, seed=90000 + i) for i in range(25)]

    torch.manual_seed(seed)
    model = ArrivalModel(**VARIANTS[name])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            log_first, _, p = model(batch["perm"], batch["start"])
            model.loss(p, log_first, batch["target"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    correct = total = 0
    steps_sum = 0.0
    by_distance_correct = torch.zeros(MAX_DEPTH)
    by_distance_total = torch.zeros(MAX_DEPTH)
    with torch.no_grad():
        for batch in validation:
            _, predicted, _ = model(batch["perm"], batch["start"])
            target = batch["target"]
            steps_sum += float(predicted.float().mean())
            hit = (predicted == target).float()
            correct += int(hit.sum())
            total += len(target)
            by_distance_correct.index_add_(0, target - 1, hit)
            by_distance_total.index_add_(0, target - 1, torch.ones_like(hit))

    return {
        "variant": name,
        "seed": seed,
        "params": sum(x.numel() for x in model.parameters()),
        "accuracy": correct / total,
        "mean_steps": steps_sum / len(validation),
        "accuracy_by_distance": (
            by_distance_correct / by_distance_total.clamp_min(1)
        ).tolist(),
    }
