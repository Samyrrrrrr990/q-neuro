"""Gate 5 branches A (complex fields) and C (homeostatic control), on the frozen lookup family.

Lane B. Each branch must add an independent capability or Pareto improvement beyond the frozen
adaptive-halting core. Each is built so its own natural control can kill it quickly.

**A -- complex fields.** Q-Neuro 1.0/2.0 already established that any complex network has an exact
structured-real counterpart (`FAIL-005`), so expressivity cannot be the mechanism. The only
remaining route is optimisation dynamics: whether a complex parameterisation changes how reliably
or how shallowly the halting solution is found. The control is a real model at **matched real
parameter count** -- complex dimension `d/2` against real dimension `d`.

**C -- homeostatic control.** RMS normalisation is already a homeostatic mechanism and is what
turned an 11-of-24 seed lottery into 20 of 20. The question is whether a *learned* controller adds
anything on top: a per-example adaptive gain driving the state norm towards a learned set-point,
with the error fed back. The control is fixed RMS normalisation, and the axis that matters is
reliability under hyperparameter perturbation, not accuracy at the default setting.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from research.qneuro3.decoupled import LABELS, NODES, query_chase

MAX_DEPTH = 24


class ComplexCore(nn.Module):
    """A genuinely complex state, carried as `torch.complex64` leaves.

    Halting reads `|h|` and the phase-sensitive inner product with the query, so phase can matter
    to the halting decision rather than being decoration.
    """

    def __init__(self, n_nodes: int, d: int, n_labels: int):
        super().__init__()
        self.n_nodes, self.d = n_nodes, d
        self.key = nn.Embedding(n_nodes, 2 * d)
        self.value = nn.Embedding(n_nodes, 2 * d)
        self.label = nn.Embedding(n_labels, 2 * d)
        self.query = nn.Embedding(n_labels, 2 * d)
        self.up = nn.Linear(10 * d, 4 * d)
        self.down = nn.Linear(4 * d, 2 * d)

    @staticmethod
    def _c(x: torch.Tensor) -> torch.Tensor:
        real, imag = x.chunk(2, dim=-1)
        return torch.complex(real, imag)

    @staticmethod
    def _r(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=-1)

    def context(self, perm, labels):
        idx = torch.arange(self.n_nodes, device=perm.device)
        keys = self._c(self.key(idx)).unsqueeze(0).expand(perm.shape[0], -1, -1)
        return keys, self._c(self.value(perm)), self._c(self.label(labels))

    def _attend(self, h, keys):
        # Hermitian inner product: |<k, h>| is phase sensitive, which is the point of the branch.
        score = (keys * h.unsqueeze(1).conj()).sum(-1).abs() / self.d**0.5
        return torch.softmax(score, dim=1).unsqueeze(-1)

    def advance(self, h, ctx, query, carried):
        keys, values, label_values = ctx
        attention = self._attend(h, keys)
        read = (attention * values).sum(1)
        features = torch.cat(
            [self._r(h), self._r(read), self._r(carried), self._r(query), self._r(carried * h.conj())],
            dim=-1,
        )
        h = h + self._c(self.down(torch.nn.functional.gelu(self.up(features))))
        h = h / (h.abs().pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        return h, (self._attend(h, keys) * label_values).sum(1)


class ComplexArrival(nn.Module):
    def __init__(self, n_nodes=NODES, d=32, n_labels=LABELS, max_depth=MAX_DEPTH):
        super().__init__()
        self.core = ComplexCore(n_nodes, d, n_labels)
        self.max_depth = max_depth
        self.arrive = nn.Linear(8 * d, 1)
        nn.init.constant_(self.arrive.bias, -2.0)
        self.name = nn.Linear(2 * d, n_nodes)

    def forward(self, batch):
        ctx = self.core.context(batch["perm"], batch["labels"])
        query = self.core._c(self.core.query(batch["query"]))
        h = self.core._c(self.core.key(batch["start"]))
        carried = torch.zeros_like(h)
        probs, states = [], []
        for _ in range(self.max_depth):
            h, carried = self.core.advance(h, ctx, query, carried)
            feature = torch.cat(
                [self.core._r(h), self.core._r(carried),
                 self.core._r(query), self.core._r(carried * query.conj())], dim=-1
            )
            probs.append(torch.sigmoid(self.arrive(feature).squeeze(-1)))
            states.append(self.core._r(h))
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


class HomeostaticNorm(nn.Module):
    """A learned set-point with feedback, in place of a fixed RMS normalisation.

    `gain` is driven by the error between the state's current norm and a learned target, so the
    controller can adapt per example rather than dividing by a constant. If reliability under
    hyperparameter perturbation is no better than fixed normalisation, the feedback is decoration.
    """

    def __init__(self, d: int):
        super().__init__()
        self.target = nn.Parameter(torch.ones(1))
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm = h.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
        error = (self.target - norm) / self.target.abs().clamp_min(1e-3)
        return h * (1.0 + torch.tanh(self.gain) * error) / norm


def run_complex(
    seed: int = 0, *, epochs: int = 4, train_batches: int = 250, batch_size: int = 128,
    tail: float = 0.85, learning_rate: float = 2e-3, d: int = 32, n_nodes: int = 32,
) -> dict[str, Any]:
    train = [
        query_chase(batch_size, n_nodes, MAX_DEPTH, seed=1000 + i, tail=tail)
        for i in range(train_batches)
    ]
    validation = [
        query_chase(256, n_nodes, MAX_DEPTH, seed=90000 + i, tail=tail) for i in range(20)
    ]
    torch.manual_seed(seed)
    model = ComplexArrival(n_nodes, d, LABELS, MAX_DEPTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.functional.cross_entropy
    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            _, _, log_first, stacked = model(batch)
            index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            at_truth = stacked.gather(1, index).squeeze(1)
            loss = -log_first.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
            (loss + cross_entropy(model.name(at_truth), batch["answer"])).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    correct = total = halt_correct = 0
    steps = 0.0
    with torch.no_grad():
        for batch in validation:
            logits, step, _, _ = model(batch)
            correct += int((logits.argmax(-1) == batch["answer"]).sum())
            halt_correct += int((step == batch["target"]).sum())
            steps += float(step.float().mean())
            total += len(batch["answer"])
    return {
        "branch": "A_complex", "seed": seed, "d_complex": d,
        "accuracy": correct / total, "halt_accuracy": halt_correct / total,
        "mean_steps": steps / len(validation),
        "params": sum(p.numel() for p in model.parameters()),
    }
