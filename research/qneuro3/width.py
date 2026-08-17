"""Gate 5B: does letting the model choose its WIDTH as well as its depth earn its place?

Lane B. Nothing here is a claim.

The frozen core allocates compute along one axis: how many steps. This tests the second axis. The
resource a model spends on an example becomes

    C(x) = T(x) * N(x)

with `T` the halt depth and `N` the number of active hidden groups. A branch survives only if it
adds an independent capability or a Pareto improvement beyond the frozen adaptive-halting core --
not if it merely reproduces routing that already exists.

The 2x2 is the whole experiment, because either axis alone is a known method:

    fixed depth  x fixed width   -- the full-depth baseline
    fixed depth  x adaptive width -- ordinary conditional computation / MoE-style routing
    adaptive depth x fixed width  -- the frozen Q-Neuro 3.0 core
    adaptive depth x adaptive width -- the branch under test

Width is masked on the step MLP's hidden layer, split into `groups` contiguous blocks. A router
scores the groups from the current state and keeps the top `k`; inactive blocks contribute nothing
and are not computed in the accounting. Straight-through keeps the hard mask differentiable, the
same device the commit-halting variant used in cycle 1.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from research.qneuro3.decoupled import LABELS, NODES, WIDTH, query_chase

MAX_DEPTH = 24
GROUPS = 8


class GroupedCore(nn.Module):
    """The lookup core with a maskable hidden layer and an optional per-step width router."""

    def __init__(
        self, n_nodes: int = NODES, d: int = WIDTH, n_labels: int = LABELS,
        *, groups: int = GROUPS, adaptive_width: bool = False, keep: int = GROUPS,
        static_width: bool = False,
    ):
        super().__init__()
        self.n_nodes, self.groups, self.adaptive_width, self.keep = n_nodes, groups, adaptive_width, keep
        # The control that can kill this branch: the SAME number of hidden units, chosen once at
        # construction instead of per example per step. If a statically narrow model matches the
        # routed one at equal cost, the routing adds nothing and only the narrowness mattered.
        self.static_width = static_width
        self.hidden = 2 * d
        self.block = self.hidden // groups
        self.key = nn.Embedding(n_nodes, d)
        self.value = nn.Embedding(n_nodes, d)
        self.label = nn.Embedding(n_labels, d)
        self.query = nn.Embedding(n_labels, d)
        effective = self.block * keep if static_width else self.hidden
        self.up = nn.Linear(5 * d, effective)
        self.down = nn.Linear(effective, d)
        self.router = nn.Linear(d, groups)

    def context(self, perm, labels):
        idx = torch.arange(self.n_nodes, device=perm.device)
        keys = self.key(idx).unsqueeze(0).expand(perm.shape[0], -1, -1)
        return keys, self.value(perm), self.label(labels)

    def _attend(self, h, keys):
        return torch.softmax(keys @ h.unsqueeze(-1) / h.shape[-1] ** 0.5, dim=1)

    def advance(self, h, ctx, query, carried):
        """Returns the new state, the label read, and the fraction of width actually used."""

        keys, values, label_values = ctx
        attention = self._attend(h, keys)
        features = torch.cat(
            [h, (attention * values).sum(1), carried, query, carried * query], dim=-1
        )
        inner = torch.nn.functional.gelu(self.up(features))

        if self.static_width:
            used = torch.full((h.shape[0],), self.keep / self.groups, device=h.device)
        elif self.adaptive_width:
            scores = self.router(h)
            top = scores.topk(self.keep, dim=-1).indices
            hard = torch.zeros_like(scores).scatter_(1, top, 1.0)
            # straight-through: hard mask forward, soft gradient backward
            gate = hard + torch.softmax(scores, dim=-1) - torch.softmax(scores, dim=-1).detach()
            inner = inner * gate.repeat_interleave(self.block, dim=-1)
            used = torch.full((h.shape[0],), self.keep / self.groups, device=h.device)
        else:
            used = torch.ones(h.shape[0], device=h.device)

        h = h + self.down(inner)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return h, (self._attend(h, keys) * label_values).sum(1), used


class TwoAxisModel(nn.Module):
    """Depth adaptive or not, width adaptive or not. Everything else identical."""

    def __init__(
        self, *, adaptive_depth: bool, adaptive_width: bool, keep: int = GROUPS,
        n_nodes: int = NODES, d: int = WIDTH, max_depth: int = MAX_DEPTH,
        static_width: bool = False,
    ):
        super().__init__()
        self.adaptive_depth = adaptive_depth
        self.max_depth = max_depth
        self.core = GroupedCore(
            n_nodes, d, adaptive_width=adaptive_width, keep=keep, static_width=static_width
        )
        self.score = nn.Linear(4 * d, 1)
        self.name = nn.Linear(d, n_nodes)
        if adaptive_depth:
            nn.init.constant_(self.score.bias, -2.0)

    def forward(self, batch):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        scores, states, widths = [], [], []
        for _ in range(self.max_depth):
            h, carried, used = self.core.advance(h, ctx, query, carried)
            scores.append(
                self.score(torch.cat([h, carried, query, carried * query], dim=-1)).squeeze(-1)
            )
            states.append(h)
            widths.append(used)
        per_step = torch.stack(scores, 1)
        stacked = torch.stack(states, 1)
        width = torch.stack(widths, 1)

        if self.adaptive_depth:
            p = torch.sigmoid(per_step).clamp(1e-6, 1 - 1e-6)
            log_not = torch.log1p(-p)
            cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
            per_step = torch.log(p) + cum
            fired = p > 0.5
            any_fired = fired.any(dim=1)
            chosen = torch.where(
                any_fired, fired.float().argmax(dim=1),
                torch.full_like(any_fired, self.max_depth - 1, dtype=torch.long),
            )
        else:
            chosen = per_step.argmax(1)
        picked = stacked.gather(
            1, chosen.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        ).squeeze(1)
        depth = (chosen + 1).float()
        # Active-FLOPs proxy: width used, summed over the steps ACTUALLY EXECUTED. A model without
        # adaptive depth runs every step and then selects, so it pays the full depth even though the
        # answer is attributed to one step. Charging it only up to the selected step understates its
        # cost by 4x and was a real accounting bug here.
        if self.adaptive_depth:
            executed = (
                torch.arange(self.max_depth, device=depth.device).unsqueeze(0) < depth.unsqueeze(1)
            ).float()
        else:
            executed = torch.ones_like(width)
        cost = (width * executed).sum(1)
        return self.name(picked), depth, per_step, stacked, cost


def run(
    *, adaptive_depth: bool, adaptive_width: bool, keep: int = GROUPS, seed: int = 0,
    static_width: bool = False,
    epochs: int = 4, train_batches: int = 250, batch_size: int = 128, tail: float = 0.85,
    learning_rate: float = 2e-3, max_depth: int = MAX_DEPTH, n_nodes: int = 32,
    d: int = WIDTH,
) -> dict[str, Any]:
    train = [
        query_chase(batch_size, n_nodes, max_depth, seed=1000 + i, tail=tail)
        for i in range(train_batches)
    ]
    validation = [
        query_chase(256, n_nodes, max_depth, seed=90000 + i, tail=tail) for i in range(20)
    ]
    torch.manual_seed(seed)
    model = TwoAxisModel(
        adaptive_depth=adaptive_depth, adaptive_width=adaptive_width, keep=keep,
        n_nodes=n_nodes, max_depth=max_depth, static_width=static_width, d=d,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.functional.cross_entropy
    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            _, _, per_step, stacked, _ = model(batch)
            index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            at_truth = stacked.gather(1, index).squeeze(1)
            if adaptive_depth:
                loss = -per_step.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
            else:
                loss = cross_entropy(per_step, batch["target"] - 1)
            (loss + cross_entropy(model.name(at_truth), batch["answer"])).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    correct = total = 0
    cost_sum = depth_sum = 0.0
    with torch.no_grad():
        for batch in validation:
            logits, depth, _, _, cost = model(batch)
            correct += int((logits.argmax(-1) == batch["answer"]).sum())
            total += len(batch["answer"])
            cost_sum += float(cost.mean())
            depth_sum += float(depth.mean())
    return {
        "adaptive_depth": adaptive_depth,
        "adaptive_width": adaptive_width,
        "static_width": static_width,
        "keep": keep if (adaptive_width or static_width) else GROUPS,
        "seed": seed,
        "accuracy": correct / total,
        "mean_depth": depth_sum / len(validation),
        "active_width_steps": cost_sum / len(validation),
        "d": d,
        "params": sum(p.numel() for p in model.parameters()),
    }
