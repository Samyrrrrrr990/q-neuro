"""Adapters exposing the cycle-2 models to `qneuro3.runtime` execution policies.

Lane B. The adapters add no parameters and change no arithmetic; they exist so the runtime can be
measured on a model that was trained under the frozen protocol, without the runtime knowing which
family it is driving.

`position` is per-row rather than a scalar because continuous scheduling admits rows mid-flight, so
different rows in the same iteration are at different depths.
"""

from __future__ import annotations

import torch
from torch import nn

from research.qneuro3.streaming import LENGTH, VALUES, StreamModel, threshold_crossing


class StreamAdapter:
    """Streaming threshold-crossing model, driven one token per iteration."""

    def __init__(self, model: StreamModel, max_depth: int | None = None):
        self.model = model
        self.max_depth = max_depth if max_depth is not None else model.length

    def init_state(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h, threshold = self.model.core.start(batch["threshold"])
        return {"h": h, "threshold": threshold, "values": batch["values"]}

    def step(self, state, position):
        token = state["values"].gather(1, position.unsqueeze(1)).squeeze(1)
        h, read = self.model.core.advance(state["h"], token, state["threshold"])
        feature = torch.cat([h, read], dim=-1)
        halt = torch.sigmoid(self.model.score(feature).squeeze(-1))
        return (
            {"h": h, "threshold": state["threshold"], "values": state["values"]},
            halt,
            self.model.name(feature),
        )


class DepthPredictor(nn.Module):
    """Predict the halt depth from the input alone, before any iteration runs.

    Bucketing needs this and the prior art does not supply it: halt depth is the output of the
    computation, not a readable property of the input the way sequence length is. Whether a cheap
    predictor is good enough to schedule on is an open question, and this is the cheapest honest
    attempt at one -- a single pass over the sequence, no recurrence over the halting dynamics.
    """

    def __init__(self, d: int = 64, length: int = LENGTH):
        super().__init__()
        self.length = length
        self.value = nn.Embedding(VALUES + 1, d)
        self.position = nn.Embedding(length, d)
        self.threshold = nn.Linear(1, d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, length))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        n, length = batch["values"].shape
        index = torch.arange(length, device=batch["values"].device).unsqueeze(0).expand(n, -1)
        tokens = self.value(batch["values"]) + self.position(index)
        pooled = tokens.mean(1)
        threshold = self.threshold(batch["threshold"].float().unsqueeze(-1) / 20.0)
        return self.head(torch.cat([pooled, threshold], dim=-1))


def train_depth_predictor(
    *, seed: int = 0, epochs: int = 3, train_batches: int = 300, batch_size: int = 128,
    length: int = LENGTH, tail: float | None = None, learning_rate: float = 2e-3,
) -> DepthPredictor:
    train = [
        threshold_crossing(batch_size, seed=1000 + i, length=length, tail=tail)
        for i in range(train_batches)
    ]
    torch.manual_seed(seed)
    model = DepthPredictor(length=length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(batch), batch["target"] - 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def trivial_depth_predictors(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Baselines any learned predictor must beat before it earns its place.

    `threshold / mean_value` is the obvious closed-form estimate of where a running sum of values
    uniform on 1..9 crosses a threshold. If that is as good as a learned model, the learned model
    is not contributing anything.
    """

    threshold = batch["threshold"].float()
    return {
        "constant_mean": torch.full_like(threshold, float(batch["values"].shape[1]) / 2.0),
        "threshold_over_mean_value": threshold / 5.0,
        "threshold_over_batch_mean": threshold / batch["values"].float().mean(),
        "oracle": batch["target"].float(),
    }


class QueryAdapter:
    """Associative-lookup model (`query_chase`), driven one hop per iteration.

    A different core entirely from `StreamAdapter`: attention over every node at every step, so the
    per-step cost is far higher while the launch overhead is the same. That is precisely why it is
    the family the runtime crossover prediction is confirmed on rather than calibrated on.
    """

    def __init__(self, model, max_depth: int):
        self.model = model
        self.max_depth = max_depth

    def init_state(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        keys, values, label_values = self.model.core.context(batch["perm"], batch["labels"])
        return {
            "h": self.model.core.key(batch["start"]),
            "carried": torch.zeros_like(self.model.core.key(batch["start"])),
            "keys": keys,
            "values": values,
            "label_values": label_values,
            "query": self.model.core.query(batch["query"]),
        }

    def step(self, state, position):
        ctx = (state["keys"], state["values"], state["label_values"])
        h, carried = self.model.core.advance(state["h"], ctx, state["query"], state["carried"])
        feature = torch.cat([h, carried, state["query"], carried * state["query"]], dim=-1)
        halt = torch.sigmoid(self.model.arrive(feature).squeeze(-1))
        new_state = dict(state)
        new_state["h"], new_state["carried"] = h, carried
        return new_state, halt, self.model.name(h)
