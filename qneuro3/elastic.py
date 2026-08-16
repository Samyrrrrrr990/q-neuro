"""Q0 (fixed depth) and Q1 (learned halting) over an identical recurrent core.

Both share the same Core, so the only architectural difference is Q1's single halting head. Any
Pareto difference is therefore attributable to the halting mechanism and not to capacity.
"""

from __future__ import annotations

import torch
from torch import nn


class Core(nn.Module):
    """One hop of chain following, plus a distance readout."""

    def __init__(self, n_nodes: int, d: int, n_out: int):
        super().__init__()
        self.n_nodes = n_nodes
        # Following a chain is an associative lookup: match the current node against the KEY
        # (the node's identity) and read the VALUE (its successor). Using one embedding for both
        # makes the lookup impossible, which is a real bug that silently produces chance accuracy.
        self.key = nn.Embedding(n_nodes, d)
        self.value = nn.Embedding(n_nodes, d)
        self.step = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.readout = nn.Linear(d, n_out)

    def context(self, perm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.arange(self.n_nodes, device=perm.device)
        keys = self.key(idx).unsqueeze(0).expand(perm.shape[0], -1, -1)
        values = self.value(perm)
        return keys, values

    def start_state(self, start: torch.Tensor) -> torch.Tensor:
        return self.key(start)

    def advance(self, h: torch.Tensor, ctx) -> torch.Tensor:
        keys, values = ctx
        attn = torch.softmax(keys @ h.unsqueeze(-1) / h.shape[-1] ** 0.5, dim=1)
        return h + self.step(torch.cat([h, (attn * values).sum(1)], dim=-1))


class Q0Fixed(nn.Module):
    """Always spends `depth` steps, regardless of how far the goal actually is."""

    def __init__(self, n_nodes: int, d: int, n_out: int, depth: int):
        super().__init__()
        self.core = Core(n_nodes, d, n_out)
        self.depth = depth

    def forward(self, perm, start):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        for _ in range(self.depth):
            h = self.core.advance(h, ctx)
        return self.core.readout(h), torch.full((perm.shape[0],), float(self.depth))


class Q1Elastic(nn.Module):
    """PonderNet-style learned halting over the same core."""

    def __init__(self, n_nodes: int, d: int, n_out: int, max_depth: int, halt_bias: float = -5.0):
        super().__init__()
        self.core = Core(n_nodes, d, n_out)
        self.max_depth = max_depth
        self.halt = nn.Linear(d, 1)
        # Ponder collapse is the dominant failure mode of learned halting: the halt head shuts
        # depth down before the core has learned to use it, and the model never escapes. Biasing
        # the head to start almost never halting forces deep computation early, so the core can
        # become useful before halting has anything to trade against.
        nn.init.constant_(self.halt.bias, halt_bias)

    def forward(self, perm, start):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        remaining = torch.ones(perm.shape[0], device=perm.device)
        out = 0.0
        expected = torch.zeros(perm.shape[0], device=perm.device)
        for step in range(self.max_depth):
            h = self.core.advance(h, ctx)
            p = torch.sigmoid(self.halt(h)).squeeze(-1)
            if step == self.max_depth - 1:
                p = torch.ones_like(p)
            w = remaining * p
            out = out + w.unsqueeze(-1) * torch.softmax(self.core.readout(h), dim=-1)
            expected = expected + w * (step + 1)
            remaining = remaining * (1 - p)
        return torch.log(out + 1e-9), expected


class Q2Commit(nn.Module):
    """Halting that COMMITS to the readout at the halt step instead of mixing over steps.

    Diagnosis behind this variant: PonderNet's output is a weighted average over every step's
    prediction. That is well matched to tasks where partial computation gives a partially correct
    answer, and badly matched to tasks where intermediate states are simply wrong. Following a
    chain is the second kind: at hop 3 of an 8-hop chase the model is at the wrong node, and
    mixing that in caps achievable accuracy no matter how the halting is tuned.

    Here halting is a hard decision, made differentiable with a straight-through estimator, so the
    emitted answer is the state the model actually stopped at.
    """

    def __init__(self, n_nodes: int, d: int, n_out: int, max_depth: int, halt_bias: float = -2.0):
        super().__init__()
        self.core = Core(n_nodes, d, n_out)
        self.max_depth = max_depth
        self.halt = nn.Linear(d, 1)
        nn.init.constant_(self.halt.bias, halt_bias)

    def forward(self, perm, start):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        batch = perm.shape[0]
        alive = torch.ones(batch, device=perm.device)
        final = torch.zeros(batch, self.core.readout.out_features, device=perm.device)
        steps = torch.zeros(batch, device=perm.device)
        halt_logits = []
        for step in range(self.max_depth):
            h = self.core.advance(h, ctx)
            p = torch.sigmoid(self.halt(h)).squeeze(-1)
            halt_logits.append(p)
            stop = (p > 0.5).float() if step < self.max_depth - 1 else torch.ones_like(p)
            # straight-through: hard decision forward, soft gradient backward
            stop = stop + p - p.detach()
            take = alive * stop
            final = final + take.unsqueeze(-1) * self.core.readout(h)
            steps = steps + alive * (1.0)
            alive = alive * (1 - take)
        return final, steps, torch.stack(halt_logits, 1)


class Q3Arrival(nn.Module):
    """Halt on DETECTED ARRIVAL, and let the halt step itself be the answer.

    Cycle 1 showed the obstruction precisely: a scalar mean-depth penalty pressures average depth
    and never induces per-example allocation. The fix is not a better penalty, it is a different
    relationship between computation and output.

    Here the model chases the chain and, at each step, predicts whether it has arrived. It emits no
    learned distance readout at all: the answer IS the number of steps taken. Computation and answer
    become the same object, so allocating compute per example is not an auxiliary objective the
    model must be bribed into, it is the task.

    Training maximises the likelihood of halting at exactly the true distance, which is a
    per-example grounded signal rather than a free scalar.
    """

    def __init__(self, n_nodes: int, d: int, max_depth: int, halt_bias: float = -2.0):
        super().__init__()
        self.core = Core(n_nodes, d, 1)
        self.max_depth = max_depth
        self.arrive = nn.Linear(d, 1)
        nn.init.constant_(self.arrive.bias, halt_bias)

    def forward(self, perm, start):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        probs = []
        for _ in range(self.max_depth):
            h = self.core.advance(h, ctx)
            probs.append(torch.sigmoid(self.arrive(h)).squeeze(-1))
        p = torch.stack(probs, 1).clamp(1e-6, 1 - 1e-6)
        # Likelihood of first arrival at step t: p_t * prod_{s<t} (1 - p_s).
        log_not = torch.log1p(-p)
        cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
        log_first = torch.log(p) + cum
        # Inference: stop at the first step whose arrival probability crosses one half.
        fired = p > 0.5
        any_fired = fired.any(dim=1)
        first = torch.where(
            any_fired, fired.float().argmax(dim=1), torch.full_like(any_fired, self.max_depth - 1, dtype=torch.long)
        )
        return log_first, first + 1


def arrival_loss(log_first: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of halting at exactly the true distance."""

    return -log_first.gather(1, (distance - 1).unsqueeze(1)).squeeze(1).mean()


class Q4Grounded(nn.Module):
    """Q3, plus a training-only signal that the recurrent state must name where it is.

    Frozen as `QNEURO3-Q4-P1`. Q3's outcome is bimodal across seeds (`QNEURO3-Q3-VARIANCE-001`):
    nothing in the arrival loss forces the state to track position on the chain, so a competing
    "fire at a typical depth" solution is available and roughly four runs in ten fall into it. The
    auxiliary head supplies the missing constraint directly.

    `position_logits` is never called at inference, so the halting rule, the inference FLOPs and
    the inference-path parameters are identical to `Q3Arrival`.
    """

    def __init__(self, n_nodes: int, d: int, max_depth: int, halt_bias: float = -2.0):
        super().__init__()
        self.core = Core(n_nodes, d, 1)
        self.max_depth = max_depth
        self.arrive = nn.Linear(d, 1)
        nn.init.constant_(self.arrive.bias, halt_bias)
        self.where = nn.Linear(d, n_nodes)

    def forward(self, perm, start, *, with_positions: bool = False):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        probs, states = [], []
        for _ in range(self.max_depth):
            h = self.core.advance(h, ctx)
            probs.append(torch.sigmoid(self.arrive(h)).squeeze(-1))
            if with_positions:
                states.append(h)
        p = torch.stack(probs, 1).clamp(1e-6, 1 - 1e-6)
        log_not = torch.log1p(-p)
        cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
        log_first = torch.log(p) + cum
        fired = p > 0.5
        any_fired = fired.any(dim=1)
        first = torch.where(
            any_fired, fired.float().argmax(dim=1),
            torch.full_like(any_fired, self.max_depth - 1, dtype=torch.long),
        )
        if not with_positions:
            return log_first, first + 1
        return log_first, first + 1, self.where(torch.stack(states, 1))


def occupied_nodes(perm: torch.Tensor, start: torch.Tensor, max_depth: int) -> torch.Tensor:
    """Ground truth for the auxiliary head: which node the walk sits on after each hop."""

    current = start
    out = []
    for _ in range(max_depth):
        current = perm.gather(1, current.unsqueeze(1)).squeeze(1)
        out.append(current)
    return torch.stack(out, 1)


class Q3bLabel(nn.Module):
    """Halt on detected arrival, then emit a LABEL read from the goal node.

    This decouples the answer from the step count. If Q3's compute saving survives here, the
    mechanism is halting-on-arrival. If it vanishes, Q3's result was an artifact of the answer
    happening to equal the number of steps.
    """

    def __init__(self, n_nodes: int, d: int, max_depth: int, n_labels: int, halt_bias: float = -2.0):
        super().__init__()
        self.core = Core(n_nodes, d, n_labels)
        self.max_depth = max_depth
        self.arrive = nn.Linear(d, 1)
        nn.init.constant_(self.arrive.bias, halt_bias)
        self.label_embed = nn.Embedding(n_nodes, d)

    def forward(self, perm, start, goal_label_table):
        ctx = self.core.context(perm)
        h = self.core.start_state(start)
        probs, states = [], []
        for _ in range(self.max_depth):
            h = self.core.advance(h, ctx)
            probs.append(torch.sigmoid(self.arrive(h)).squeeze(-1))
            states.append(h)
        p = torch.stack(probs, 1).clamp(1e-6, 1 - 1e-6)
        H = torch.stack(states, 1)
        log_not = torch.log1p(-p)
        cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
        log_first = torch.log(p) + cum
        fired = p > 0.5
        any_fired = fired.any(dim=1)
        first = torch.where(
            any_fired, fired.float().argmax(dim=1),
            torch.full_like(any_fired, self.max_depth - 1, dtype=torch.long),
        )
        picked = H.gather(1, first.view(-1, 1, 1).expand(-1, 1, H.shape[-1])).squeeze(1)
        # The label lives on the goal node, so it is only readable once arrival has happened.
        conditioned = picked + goal_label_table
        return log_first, self.core.readout(conditioned), first + 1
