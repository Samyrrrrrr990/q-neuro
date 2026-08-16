"""Instrumented Q3 training: log mechanistic observables densely through a run.

Lane B. Nothing here is a claim. See `docs/LANE_POLICY.md`.

Q-Neuro 3.0 cycle 1 established (`QNEURO3-Q3-VARIANCE-001`) that Q3's outcome is **bimodal**:
every run lands either at accuracy 0.9994-1.0000 with 4.54 average steps, or at 0.42-0.57 with
5.2-6.1 steps, and nothing lands between 0.5664 and 0.9994. Training volume is irrelevant; the
initialisation seed decides.

That makes this the first system in the whole programme with **two genuinely distinguishable
functional attractors**. `DISCOVERY-002`'s committor branch died (`FAIL-009`) precisely because no
such system existed. This module exists to find out whether the eventual mode is predictable from
observables measured long before accuracy separates.

The observables are chosen mechanistically, not by search:

* ``dec_isgoal`` -- can a linear probe read "the walk is standing on the goal" out of the recurrent
  state? The good solution has to encode this; the "fire at a typical depth" solution does not.
* ``dec_node`` / ``dec_remaining`` -- weaker versions of the same question. Q4 showed that *forcing*
  node decodability does not produce the good mode, so decodability is not sufficient. It may still
  be necessary, and therefore predictive.
* ``halt_entropy``, ``p_profile`` -- shape of the halting distribution, independent of whether it is
  correct.
* ``attn_entropy`` -- is the core doing a sharp associative lookup or smearing over all nodes?
* ``state_rank`` -- effective rank of the state trajectory.

`accuracy` is logged too, as the quantity any predictor has to beat at the time it is measured.
"""

from __future__ import annotations

from typing import Any

import torch

from qneuro3.elastic import Q3Arrival, arrival_loss, occupied_nodes
from qneuro3.tasks import chase_to_goal

NODES = 24
WIDTH = 64
MAX_DEPTH = 8


def _ridge_probe(features: torch.Tensor, targets: torch.Tensor, classes: int) -> float:
    """Closed-form ridge onto one-hot targets; returns argmax accuracy.

    A linear probe, not a trained head: the question is what the state already exposes, not what a
    nonlinear reader could extract with enough capacity.
    """

    x = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1).double()
    y = torch.nn.functional.one_hot(targets, classes).double()
    gram = x.T @ x + 1e-3 * torch.eye(x.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(gram, x.T @ y)
    return float(((x @ weights).argmax(1) == targets).float().mean())


def _balanced_binary_probe(features: torch.Tensor, targets: torch.Tensor) -> float:
    """Balanced accuracy for the goal indicator, which is ~1/8 positive and would flatter a plain
    accuracy score."""

    x = torch.cat([features, torch.ones(features.shape[0], 1)], dim=1).double()
    y = targets.double().unsqueeze(1)
    gram = x.T @ x + 1e-3 * torch.eye(x.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(gram, x.T @ y)
    scores = (x @ weights).squeeze(1)
    positive = targets.bool()
    if not bool(positive.any()) or not bool((~positive).any()):
        return 0.5
    threshold = scores.median()
    predicted = scores > threshold
    true_positive = float((predicted & positive).sum()) / float(positive.sum())
    true_negative = float((~predicted & ~positive).sum()) / float((~positive).sum())
    return 0.5 * (true_positive + true_negative)


def _effective_rank(matrix: torch.Tensor) -> float:
    """exp of the entropy of the normalised singular spectrum."""

    singular = torch.linalg.svdvals(matrix.double() - matrix.double().mean(0, keepdim=True))
    total = singular.sum()
    if float(total) <= 0:
        return 0.0
    weights = singular / total
    entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


@torch.no_grad()
def observe(model: Q3Arrival, probe: dict[str, torch.Tensor]) -> dict[str, float]:
    """Every observable, measured on one frozen probe batch."""

    perm, start, target = probe["perm"], probe["start"], probe["target"]
    ctx = model.core.context(perm)
    h = model.core.start_state(start)
    states, probs, attention_entropies = [], [], []
    keys, _values = ctx
    for _ in range(model.max_depth):
        scores = torch.softmax(keys @ h.unsqueeze(-1) / h.shape[-1] ** 0.5, dim=1).squeeze(-1)
        attention_entropies.append(
            float(-(scores * scores.clamp_min(1e-12).log()).sum(1).mean())
        )
        h = model.core.advance(h, ctx)
        states.append(h)
        probs.append(torch.sigmoid(model.arrive(h)).squeeze(-1))

    stacked = torch.stack(states, 1)
    p = torch.stack(probs, 1).clamp(1e-6, 1 - 1e-6)
    log_not = torch.log1p(-p)
    cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
    log_first = torch.log(p) + cum
    first_mass = log_first.exp()
    fired = p > 0.5
    any_fired = fired.any(dim=1)
    first = torch.where(
        any_fired, fired.float().argmax(dim=1),
        torch.full_like(any_fired, model.max_depth - 1, dtype=torch.long),
    )
    predicted = first + 1

    occupied = occupied_nodes(perm, start, model.max_depth)
    flat_states = stacked.reshape(-1, stacked.shape[-1])
    flat_nodes = occupied.reshape(-1)
    step_index = torch.arange(model.max_depth).repeat(perm.shape[0], 1)
    remaining = (target.unsqueeze(1) - step_index - 1).clamp(0, model.max_depth - 1).reshape(-1)
    is_goal = (flat_nodes == 0).long()

    normalised = first_mass / first_mass.sum(1, keepdim=True).clamp_min(1e-12)
    halt_entropy = float(
        -(normalised * normalised.clamp_min(1e-12).log()).sum(1).mean()
    )

    record: dict[str, float] = {
        "accuracy": float((predicted == target).float().mean()),
        "loss": float(arrival_loss(log_first, target)),
        "mean_steps": float(predicted.float().mean()),
        "halt_entropy": halt_entropy,
        "p_mean": float(p.mean()),
        "p_std_across_steps": float(p.mean(0).std()),
        "never_fires_frac": float((~any_fired).float().mean()),
        "attn_entropy": sum(attention_entropies) / len(attention_entropies),
        "attn_entropy_first": attention_entropies[0],
        "state_rank": _effective_rank(stacked[:, -1]),
        "dec_node": _ridge_probe(flat_states, flat_nodes, NODES),
        "dec_remaining": _ridge_probe(flat_states, remaining, model.max_depth),
        "dec_isgoal": _balanced_binary_probe(flat_states, is_goal),
    }
    for step in range(model.max_depth):
        record[f"p_step{step + 1}"] = float(p[:, step].mean())
    return record


def run_instrumented(
    seed: int,
    *,
    epochs: int = 8,
    train_batches: int = 500,
    batch_size: int = 128,
    log_every: int = 100,
    learning_rate: float = 2e-3,
) -> dict[str, Any]:
    """One Q3 run, logging every `log_every` optimizer steps."""

    train = [chase_to_goal(batch_size, NODES, MAX_DEPTH, seed=1000 + i) for i in range(train_batches)]
    probe = chase_to_goal(512, NODES, MAX_DEPTH, seed=77000)
    validation = [chase_to_goal(256, NODES, MAX_DEPTH, seed=90000 + i) for i in range(25)]

    torch.manual_seed(seed)
    model = Q3Arrival(NODES, WIDTH, MAX_DEPTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trace: list[dict[str, float]] = []
    step_count = 0
    initial = observe(model, probe)
    initial["step"] = 0
    trace.append(initial)

    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            log_first, _ = model(batch["perm"], batch["start"])
            arrival_loss(log_first, batch["target"]).backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            step_count += 1
            if step_count % log_every == 0:
                record = observe(model, probe)
                record["step"] = step_count
                record["grad_norm"] = grad_norm
                trace.append(record)

    correct = total = 0
    steps = 0.0
    with torch.no_grad():
        for batch in validation:
            _, predicted = model(batch["perm"], batch["start"])
            steps += float(predicted.float().mean())
            correct += int((predicted == batch["target"]).sum())
            total += len(batch["target"])

    return {
        "seed": seed,
        "final_accuracy": correct / total,
        "final_steps": steps / len(validation),
        "mode": "good" if correct / total >= 0.99 else "bad",
        "trace": trace,
    }
