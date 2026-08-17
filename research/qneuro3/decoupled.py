"""Decoupled-answer task: does the compute saving survive when the answer is NOT the step count?

Lane B. Nothing here is a claim. See `docs/LANE_POLICY.md`.

This is the control that `QNEURO3-Q3-P1` specified and did not run, left open in
`QNEURO3-CYCLE-001-CLOSE` as: *"whether Q3's result depends on the answer coinciding with the step
count."* It became runnable only once the arrival model trained reliably.

**A first version of this task leaked and is preserved as `FAIL-026`.** It kept the goal at node 0
and asked for the goal's label. Because the goal's identity was fixed and known, its label was
directly addressable by attention at any step, without walking anywhere. Every configuration scored
label accuracy 1.0000 -- including runs whose halting was 30% correct, which is what exposed it. A
control that every model passes is not a control.

The task below removes the address. The goal is **content-defined**: walk until reaching the first
node whose label equals a per-example query, then report **which node that is**.

* Nothing identifies the goal in advance. Several nodes share the query label, and which one is
  reached first depends on the path, so the answer cannot be looked up without walking.
* The answer is a node identity, statistically independent of the hop count, so a model cannot
  score by reporting its own step count.
* The halting predicate -- ``label[current] == query`` -- is genuinely checkable from the state.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

NODES = 24
WIDTH = 64
MAX_DEPTH = 8
LABELS = 6


def query_chase(
    batch: int, n_nodes: int = NODES, max_hops: int = MAX_DEPTH, *, seed: int,
    n_labels: int = LABELS, tail: float | None = None
) -> dict[str, torch.Tensor]:
    """Walk from `start` to the first node whose label equals `query`; report that node's identity.

    `tail=r` makes P(distance = k) proportional to r**k instead of uniform, the heavy-tailed regime
    declared in the frozen `QNEURO3-NICHE-P1`. `tail=None` is the original uniform path and every
    earlier number was measured on it.
    """

    generator = torch.Generator().manual_seed(seed)
    weights = None if tail is None else tail ** torch.arange(1, max_hops + 1, dtype=torch.float)
    perms, starts, queries, labels, distances, answers = [], [], [], [], [], []
    for _ in range(batch):
        order = torch.randperm(n_nodes, generator=generator)
        perm = torch.empty(n_nodes, dtype=torch.long)
        perm[order] = order.roll(-1)

        start_index = int(torch.randint(0, n_nodes, (1,), generator=generator))
        start = order[start_index]
        distance = (
            int(torch.randint(1, max_hops + 1, (1,), generator=generator))
            if weights is None
            else int(torch.multinomial(weights, 1, generator=generator)) + 1
        )
        query = int(torch.randint(0, n_labels, (1,), generator=generator))

        node_labels = torch.randint(0, n_labels, (n_nodes,), generator=generator)
        # The first `distance - 1` nodes on the path must not match, and the one at `distance` must.
        walk = []
        current = start
        for _ in range(max_hops):
            current = perm[current]
            walk.append(int(current))
        for node in walk[: distance - 1]:
            if int(node_labels[node]) == query:
                node_labels[node] = (query + 1 + int(
                    torch.randint(0, n_labels - 1, (1,), generator=generator)
                )) % n_labels
        goal = walk[distance - 1]
        node_labels[goal] = query

        perms.append(perm)
        starts.append(start)
        queries.append(query)
        labels.append(node_labels)
        distances.append(distance)
        answers.append(goal)
    return {
        "perm": torch.stack(perms),
        "start": torch.stack(starts),
        "query": torch.tensor(queries),
        "labels": torch.stack(labels),
        "target": torch.tensor(distances),
        "answer": torch.tensor(answers),
    }


class QueryCore(nn.Module):
    """One hop, reading the successor and the label of wherever the walk now stands."""

    def __init__(self, n_nodes: int, d: int, n_labels: int, *, normalise: bool = False):
        super().__init__()
        self.n_nodes = n_nodes
        self.normalise = normalise
        self.key = nn.Embedding(n_nodes, d)
        self.value = nn.Embedding(n_nodes, d)
        self.label = nn.Embedding(n_labels, d)
        self.query = nn.Embedding(n_labels, d)
        self.step = nn.Sequential(nn.Linear(5 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def context(self, perm: torch.Tensor, labels: torch.Tensor):
        idx = torch.arange(self.n_nodes, device=perm.device)
        keys = self.key(idx).unsqueeze(0).expand(perm.shape[0], -1, -1)
        return keys, self.value(perm), self.label(labels)

    def _attend(self, h: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        return torch.softmax(keys @ h.unsqueeze(-1) / h.shape[-1] ** 0.5, dim=1)

    def advance(self, h: torch.Tensor, ctx, query: torch.Tensor, carried: torch.Tensor):
        """Take one hop, then read the label of where the walk now stands.

        Read ordering matters and a first version got it wrong (`FAIL-028`): reading the label
        before the hop means the halting head is asked to check a predicate about a node it has not
        reached, which is unanswerable. The label read is therefore taken *after* the update and
        returned, both for this step's halting decision and as input to the next update, so a
        fixed-depth model can latch a match it passes.
        """

        keys, values, label_values = ctx
        attention = self._attend(h, keys)
        # `carried * query` is the elementwise product whose sum is the match score. Without it a
        # linear reader cannot express "these two are equal" at all, only "these two are large" --
        # the same expressivity gap the V2 variant exposed on the halting head.
        h = h + self.step(
            torch.cat([h, (attention * values).sum(1), carried, query, carried * query], dim=-1)
        )
        if self.normalise:
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return h, (self._attend(h, keys) * label_values).sum(1)


class FixedQuery(nn.Module):
    """Always spends `depth` hops, then names the matching node from its final state."""

    def __init__(self, n_nodes=NODES, d=WIDTH, depth=MAX_DEPTH, n_labels=LABELS, *,
                 normalise=False, max_depth=None):
        depth = max_depth if max_depth is not None else depth
        super().__init__()
        self.core = QueryCore(n_nodes, d, n_labels, normalise=normalise)
        self.depth = depth
        self.name = nn.Linear(d, n_nodes)

    def forward(self, batch: dict[str, torch.Tensor]):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        for _ in range(self.depth):
            h, carried = self.core.advance(h, ctx, query, carried)
        return self.name(h), torch.full((h.shape[0],), float(self.depth))


class ArrivalQuery(nn.Module):
    """Halts on a detected match, then names the node it stopped on."""

    def __init__(
        self, n_nodes=NODES, d=WIDTH, max_depth=MAX_DEPTH, n_labels=LABELS,
        *, halt_bias=-2.0, normalise=False,
    ):
        super().__init__()
        self.core = QueryCore(n_nodes, d, n_labels, normalise=normalise)
        self.max_depth = max_depth
        self.arrive = nn.Linear(4 * d, 1)
        nn.init.constant_(self.arrive.bias, halt_bias)
        self.name = nn.Linear(d, n_nodes)

    def forward(self, batch: dict[str, torch.Tensor]):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        probs, states = [], []
        for _ in range(self.max_depth):
            h, carried = self.core.advance(h, ctx, query, carried)
            # The predicate is "the label here equals the query", so the head is given exactly
            # those three things and nothing else has to be inferred.
            probs.append(
                torch.sigmoid(
                    self.arrive(torch.cat([h, carried, query, carried * query], dim=-1))
                ).squeeze(-1)
            )
            states.append(h)
        p = torch.stack(probs, 1).clamp(1e-6, 1 - 1e-6)
        stacked = torch.stack(states, 1)
        log_not = torch.log1p(-p)
        cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
        log_first = torch.log(p) + cum
        fired = p > 0.5
        any_fired = fired.any(dim=1)
        first = torch.where(
            any_fired, fired.float().argmax(dim=1),
            torch.full_like(any_fired, self.max_depth - 1, dtype=torch.long),
        )
        picked = stacked.gather(
            1, first.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        ).squeeze(1)
        return self.name(picked), first + 1, log_first, stacked


class SelectQuery(nn.Module):
    """The decisive control: full depth, but the answer is ATTRIBUTED TO A STEP.

    `FixedQuery` fails this task, and the tempting reading is that halting is what wins. That
    reading is wrong unless this model also fails. `SelectQuery` spends all `max_depth` hops -- no
    saving whatsoever -- but emits a candidate answer and a match score per step and selects the
    highest-scoring step, so it never has to carry a match through later iterations.

    If this reaches the arrival model's accuracy, the capability comes from per-step attribution
    and halting's separate contribution is exactly the compute saving. If it does not, stopping
    itself is doing something attribution cannot.
    """

    def __init__(
        self, n_nodes=NODES, d=WIDTH, max_depth=MAX_DEPTH, n_labels=LABELS, *, normalise=False
    ):
        super().__init__()
        self.core = QueryCore(n_nodes, d, n_labels, normalise=normalise)
        self.max_depth = max_depth
        self.score = nn.Linear(4 * d, 1)
        self.name = nn.Linear(d, n_nodes)

    def forward(self, batch: dict[str, torch.Tensor]):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        scores, states = [], []
        for _ in range(self.max_depth):
            h, carried = self.core.advance(h, ctx, query, carried)
            scores.append(
                self.score(torch.cat([h, carried, query, carried * query], dim=-1)).squeeze(-1)
            )
            states.append(h)
        logits = torch.stack(scores, 1)
        stacked = torch.stack(states, 1)
        chosen = logits.argmax(1)
        picked = stacked.gather(
            1, chosen.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
        ).squeeze(1)
        steps = torch.full((h.shape[0],), float(self.max_depth))
        return self.name(picked), steps, logits, stacked


class FixedSupervisedQuery(nn.Module):
    """Matched-supervision control. Full depth, per-step match supervision, FINAL-state answer.

    Comparing `select` against `fixed` confounds two things: `select` is told which step is the
    match and `fixed` is not. This model removes the confound. It receives exactly the per-step
    supervision `select` receives, but the answer is still read from the final state rather than
    from the matched step.

    If it succeeds, the separation was about supervision and attribution adds nothing. If it fails,
    reading the answer from the matched step is the operative mechanism, with supervision matched.
    """

    def __init__(
        self, n_nodes=NODES, d=WIDTH, max_depth=MAX_DEPTH, n_labels=LABELS, *, normalise=False
    ):
        super().__init__()
        self.core = QueryCore(n_nodes, d, n_labels, normalise=normalise)
        self.max_depth = max_depth
        self.score = nn.Linear(4 * d, 1)
        self.name = nn.Linear(d, n_nodes)

    def forward(self, batch: dict[str, torch.Tensor]):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        scores = []
        for _ in range(self.max_depth):
            h, carried = self.core.advance(h, ctx, query, carried)
            scores.append(
                self.score(torch.cat([h, carried, query, carried * query], dim=-1)).squeeze(-1)
            )
        steps = torch.full((h.shape[0],), float(self.max_depth))
        return self.name(h), steps, torch.stack(scores, 1), h


class MeanPooledQuery(nn.Module):
    """Full depth, per-step supervision, answer read from the MEAN of all states.

    Isolates gradient-path length from input-dependent selection. Here the answer gradient reaches
    every step in one hop, exactly as it does for `select`, but which step contributes is fixed in
    advance rather than chosen from the input. If this succeeds, short credit assignment is the
    operative thing. If it fails, selection is.
    """

    def __init__(
        self, n_nodes=NODES, d=WIDTH, max_depth=MAX_DEPTH, n_labels=LABELS, *, normalise=False
    ):
        super().__init__()
        self.core = QueryCore(n_nodes, d, n_labels, normalise=normalise)
        self.max_depth = max_depth
        self.score = nn.Linear(4 * d, 1)
        self.name = nn.Linear(d, n_nodes)

    def forward(self, batch: dict[str, torch.Tensor]):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        scores, states = [], []
        for _ in range(self.max_depth):
            h, carried = self.core.advance(h, ctx, query, carried)
            scores.append(
                self.score(torch.cat([h, carried, query, carried * query], dim=-1)).squeeze(-1)
            )
            states.append(h)
        pooled = torch.stack(states, 1).mean(1)
        steps = torch.full((h.shape[0],), float(self.max_depth))
        return self.name(pooled), steps, torch.stack(scores, 1), pooled


class GatedQuery(nn.Module):
    """The strongest fixed-depth baseline: an explicit learned latch.

    Full depth, per-step supervision, final-state answer -- but the update is gated, so the model
    can freeze its own state the moment it detects a match and carry it unchanged to the end. If
    preservation is what the attribution models are really buying, an explicit latch should buy it
    too, without any early stopping.
    """

    def __init__(
        self, n_nodes=NODES, d=WIDTH, max_depth=MAX_DEPTH, n_labels=LABELS, *, normalise=False
    ):
        super().__init__()
        self.core = QueryCore(n_nodes, d, n_labels, normalise=normalise)
        self.max_depth = max_depth
        self.gate = nn.Linear(4 * d, 1)
        nn.init.constant_(self.gate.bias, 2.0)  # start wide open, so it learns when to close
        self.score = nn.Linear(4 * d, 1)
        self.name = nn.Linear(d, n_nodes)

    def forward(self, batch: dict[str, torch.Tensor]):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core.query(batch["query"])
        h = self.core.key(batch["start"])
        carried = torch.zeros_like(h)
        scores = []
        for _ in range(self.max_depth):
            candidate, new_carried = self.core.advance(h, ctx, query, carried)
            features = torch.cat([candidate, new_carried, query, new_carried * query], dim=-1)
            g = torch.sigmoid(self.gate(features))
            h = g * candidate + (1 - g) * h
            carried = g * new_carried + (1 - g) * carried
            scores.append(self.score(features).squeeze(-1))
        steps = torch.full((h.shape[0],), float(self.max_depth))
        return self.name(h), steps, torch.stack(scores, 1), h


def run(
    kind: str,
    seed: int,
    *,
    normalise: bool,
    epochs: int = 8,
    train_batches: int = 500,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    max_depth: int = MAX_DEPTH,
    n_nodes: int = NODES,
    tail: float | None = None,
) -> dict[str, Any]:
    train = [
        query_chase(batch_size, n_nodes, max_depth, seed=1000 + i, tail=tail)
        for i in range(train_batches)
    ]
    validation = [
        query_chase(256, n_nodes, max_depth, seed=90000 + i, tail=tail) for i in range(25)
    ]

    torch.manual_seed(seed)
    builders = {
        "arrival": ArrivalQuery,
        "fixed": FixedQuery,
        "select": SelectQuery,
        "fixed_supervised": FixedSupervisedQuery,
        "mean_pooled": MeanPooledQuery,
        "gated": GatedQuery,
    }
    model: nn.Module = builders[kind](
        n_nodes=n_nodes, normalise=normalise, max_depth=max_depth
    )
    if kind == "fixed":
        model = FixedQuery(n_nodes=n_nodes, normalise=normalise, depth=max_depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.functional.cross_entropy

    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            if kind in ("fixed_supervised", "mean_pooled", "gated"):
                logits, _, per_step, _ = model(batch)
                loss = cross_entropy(logits, batch["answer"])
                loss = loss + cross_entropy(per_step, batch["target"] - 1)
            elif kind in ("arrival", "select"):
                _, _, per_step, stacked = model(batch)
                index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
                at_truth = stacked.gather(1, index).squeeze(1)
                if kind == "arrival":
                    loss = -per_step.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
                else:
                    # Same supervision, expressed as "which step is the match" over all steps.
                    loss = cross_entropy(per_step, batch["target"] - 1)
                loss = loss + cross_entropy(model.name(at_truth), batch["answer"])
            else:
                logits, _ = model(batch)
                loss = cross_entropy(logits, batch["answer"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    correct = total = halt_correct = 0
    steps_sum = 0.0
    by_distance_correct = torch.zeros(max_depth)
    by_distance_total = torch.zeros(max_depth)
    with torch.no_grad():
        for batch in validation:
            if kind in ("arrival", "select", "fixed_supervised", "mean_pooled", "gated"):
                logits, steps, per_step, _ = model(batch)
                selected = steps if kind == "arrival" else per_step.argmax(1) + 1
                halt_correct += int((selected == batch["target"]).sum())
            else:
                logits, steps = model(batch)
            steps_sum += float(steps.float().mean())
            hit = (logits.argmax(-1) == batch["answer"]).float()
            correct += int(hit.sum())
            total += len(batch["answer"])
            by_distance_correct.index_add_(0, batch["target"] - 1, hit)
            by_distance_total.index_add_(0, batch["target"] - 1, torch.ones_like(hit))

    return {
        "kind": kind,
        "seed": seed,
        "normalise": normalise,
        "params": sum(x.numel() for x in model.parameters()),
        "answer_accuracy": correct / total,
        "halt_accuracy": halt_correct / total if kind in ("arrival", "select", "fixed_supervised", "mean_pooled", "gated")
        else None,
        "mean_steps": steps_sum / len(validation),
        "max_depth": max_depth,
        "n_nodes": n_nodes,
        "tail": tail,
        "expected_distance": float(
            (torch.arange(1, max_depth + 1).float() * by_distance_total).sum()
            / by_distance_total.sum().clamp_min(1)
        ),
        "accuracy_by_distance": (
            by_distance_correct / by_distance_total.clamp_min(1)
        ).tolist(),
    }
