"""Untouched confirmation family: streaming threshold crossing.

Lane B. Implements exactly the task and models specified in the frozen `QNEURO3-TRANSFER-P1`
(sha256 3ef3b2e5...). Nothing here may be changed after the prediction is opened.

A sequence of `length` tokens with values 1..9 arrives one per step. The predicate fires at the
first index whose running sum reaches a per-example threshold. The answer is the VALUE of the token
there, which is independent of the index.

Qualitatively different from `query_chase` by construction: streaming rather than graph traversal,
an arithmetic predicate rather than an identity match, and **no attention anywhere in the core**,
so the mechanism that carried the earlier family is absent.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

LENGTH = 12
VALUES = 9
WIDTH = 64


def threshold_crossing(
    batch: int, *, seed: int, length: int = LENGTH, tail: float | None = None
) -> dict[str, torch.Tensor]:
    """Sequence, threshold, crossing index, and the value at the crossing.

    The threshold is derived from a chosen crossing index rather than drawn independently, so the
    index distribution is controlled exactly. With `tail=None` the index is uniform on 1..length,
    which is the path every QNEURO3-TRANSFER-P1 number was measured on and is left untouched.
    `tail=r` instead makes P(index = k) proportional to r**k, the heavy-tailed regime declared in
    the frozen QNEURO3-PARETO-P1.
    """

    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(1, VALUES + 1, (batch, length), generator=generator)
    if tail is None:
        index = torch.randint(0, length, (batch,), generator=generator)
    else:
        weights = tail ** torch.arange(1, length + 1, dtype=torch.float)
        index = torch.multinomial(weights, batch, replacement=True, generator=generator)
    cumulative = values.cumsum(1)
    before = torch.where(
        index > 0,
        cumulative.gather(1, (index - 1).clamp_min(0).unsqueeze(1)).squeeze(1),
        torch.zeros(batch, dtype=cumulative.dtype),
    )
    at = cumulative.gather(1, index.unsqueeze(1)).squeeze(1)
    # Any threshold in (before, at] makes `index` the first crossing point.
    offset = torch.rand(batch, generator=generator)
    threshold = before + 1 + (offset * (at - before - 1).clamp_min(0)).long()
    return {
        "values": values,
        "threshold": threshold,
        "target": index + 1,
        "answer": values.gather(1, index.unsqueeze(1)).squeeze(1) - 1,
    }


class StreamCore(nn.Module):
    """One token per step. No attention: a value embedding, the threshold, and the running state."""

    def __init__(self, d: int = WIDTH, *, normalise: bool = True):
        super().__init__()
        self.normalise = normalise
        self.value = nn.Embedding(VALUES + 1, d)
        self.threshold = nn.Linear(1, d)
        self.step = nn.Sequential(nn.Linear(3 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def start(self, threshold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.threshold(threshold.float().unsqueeze(-1) / 20.0)
        return torch.zeros_like(embedded), embedded

    def advance(self, h: torch.Tensor, token: torch.Tensor, threshold: torch.Tensor):
        read = self.value(token)
        h = h + self.step(torch.cat([h, read, threshold], dim=-1))
        if self.normalise:
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return h, read


class StreamModel(nn.Module):
    """All six readout policies over one identical core, selected by `kind`."""

    KINDS = ("fixed", "fixed_supervised", "gated", "mean_pooled", "select", "arrival")

    def __init__(self, kind: str, d: int = WIDTH, length: int = LENGTH, *, normalise: bool = True):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        self.kind = kind
        self.length = length
        self.core = StreamCore(d, normalise=normalise)
        self.score = nn.Linear(2 * d, 1)
        self.name = nn.Linear(2 * d, VALUES)
        if kind == "arrival":
            nn.init.constant_(self.score.bias, -2.0)
        if kind == "gated":
            self.gate = nn.Linear(2 * d, 1)
            nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, batch: dict[str, torch.Tensor]):
        h, threshold = self.core.start(batch["threshold"])
        values = batch["values"]
        scores, states = [], []
        for position in range(self.length):
            candidate, read = self.core.advance(h, values[:, position], threshold)
            features = torch.cat([candidate, read], dim=-1)
            if self.kind == "gated":
                g = torch.sigmoid(self.gate(features))
                h = g * candidate + (1 - g) * h
            else:
                h = candidate
            scores.append(self.score(torch.cat([h, read], dim=-1)).squeeze(-1))
            states.append(torch.cat([h, read], dim=-1))
        per_step = torch.stack(scores, 1)
        stacked = torch.stack(states, 1)

        if self.kind in ("fixed", "fixed_supervised", "gated"):
            readout = stacked[:, -1]
            steps = torch.full((h.shape[0],), float(self.length))
        elif self.kind == "mean_pooled":
            readout = stacked.mean(1)
            steps = torch.full((h.shape[0],), float(self.length))
        elif self.kind == "select":
            chosen = per_step.argmax(1)
            readout = stacked.gather(
                1, chosen.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            ).squeeze(1)
            steps = torch.full((h.shape[0],), float(self.length))
        else:
            p = torch.sigmoid(per_step).clamp(1e-6, 1 - 1e-6)
            log_not = torch.log1p(-p)
            cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
            per_step = torch.log(p) + cum
            fired = p > 0.5
            any_fired = fired.any(dim=1)
            first = torch.where(
                any_fired, fired.float().argmax(dim=1),
                torch.full_like(any_fired, self.length - 1, dtype=torch.long),
            )
            readout = stacked.gather(
                1, first.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            ).squeeze(1)
            steps = (first + 1).float()
        return self.name(readout), steps, per_step, stacked


@torch.no_grad()
def _latency(model: nn.Module, batch: dict[str, torch.Tensor], repeats: int = 12) -> float:
    """Median wall-clock milliseconds per 1000 examples. Median, so one scheduling hiccup cannot
    carry the number, and warmed up first because the first call pays allocation costs."""

    import statistics
    import time

    for _ in range(3):
        model(batch)
    n = len(batch["answer"])
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(batch)
        timings.append((time.perf_counter() - start) * 1000.0 * 1000.0 / n)
    return statistics.median(timings)


def run(kind: str, seed: int, *, epochs: int = 8, train_batches: int = 500,
        batch_size: int = 128, learning_rate: float = 2e-3, length: int = LENGTH,
        normalise: bool = True, eval_length: int | None = None,
        tail: float | None = None, measure_latency: bool = False) -> dict[str, Any]:
    """`eval_length` runs the trained model for more steps than it was trained on, on data whose
    crossing index extends that far. Used by the frozen QNEURO3-EXTRAP-P1 depth-extrapolation
    test; leave it None for the standard in-distribution evaluation."""
    train = [
        threshold_crossing(batch_size, seed=1000 + i, length=length, tail=tail)
        for i in range(train_batches)
    ]
    evaluation_length = eval_length or length
    validation = [
        threshold_crossing(256, seed=90000 + i, length=evaluation_length, tail=tail)
        for i in range(25)
    ]

    torch.manual_seed(seed)
    model = StreamModel(kind, length=length, normalise=normalise)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.functional.cross_entropy

    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            logits, _, per_step, stacked = model(batch)
            index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            at_truth = stacked.gather(1, index).squeeze(1)
            if kind == "arrival":
                loss = -per_step.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
                loss = loss + cross_entropy(model.name(at_truth), batch["answer"])
            elif kind == "select":
                loss = cross_entropy(per_step, batch["target"] - 1)
                loss = loss + cross_entropy(model.name(at_truth), batch["answer"])
            elif kind == "fixed":
                loss = cross_entropy(logits, batch["answer"])
            else:
                loss = cross_entropy(logits, batch["answer"])
                loss = loss + cross_entropy(per_step, batch["target"] - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.length = evaluation_length
    correct = total = step_correct = 0
    steps_sum = 0.0
    by_index_correct = torch.zeros(evaluation_length)
    by_index_total = torch.zeros(evaluation_length)
    with torch.no_grad():
        for batch in validation:
            logits, steps, per_step, _ = model(batch)
            steps_sum += float(steps.mean())
            hit = (logits.argmax(-1) == batch["answer"]).float()
            correct += int(hit.sum())
            selected = steps.long() if kind == "arrival" else per_step.argmax(1) + 1
            step_correct += int((selected == batch["target"]).sum())
            total += len(batch["answer"])
            by_index_correct.index_add_(0, batch["target"] - 1, hit)
            by_index_total.index_add_(0, batch["target"] - 1, torch.ones_like(hit))

    return {
        "kind": kind,
        "seed": seed,
        "params": sum(x.numel() for x in model.parameters()),
        "answer_accuracy": correct / total,
        "step_id_accuracy": step_correct / total,
        "mean_steps": steps_sum / len(validation),
        "length": length,
        "eval_length": evaluation_length,
        "normalise": normalise,
        "tail": tail,
        "expected_index": float(
            (torch.arange(1, evaluation_length + 1).float() * by_index_total).sum()
            / by_index_total.sum().clamp_min(1)
        ),
        "latency_ms_per_1k": _latency(model, validation[0]) if measure_latency else None,
        "accuracy_by_index": (by_index_correct / by_index_total.clamp_min(1)).tolist(),
        "extrapolated_accuracy": (
            float(by_index_correct[length:].sum() / by_index_total[length:].sum().clamp_min(1))
            if evaluation_length > length else None
        ),
    }
