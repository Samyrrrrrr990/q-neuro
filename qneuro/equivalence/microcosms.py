"""Analytic microcosms used as equivalence fixtures.

These are deliberately tiny and fully deterministic. They exist so the measurement system can be
validated before it is pointed at anything complicated, per ML2-PREREG-001 section 4.
"""

from __future__ import annotations

import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    """The smallest model carrying a non-trivial hidden-unit permutation symmetry."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden = int(hidden)
        self.out_features = int(out_features)
        self.first = nn.Linear(in_features, hidden)
        self.second = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(x)))


class HomogeneousMLP(nn.Module):
    """Two layers with a positively homogeneous activation, carrying a scaling symmetry.

    ``relu`` is required: the orbit ``W1 -> sW1, b1 -> sb1, W2 -> W2/s`` is exact only because
    ``relu(sz) = s·relu(z)`` for ``s > 0``. The second bias is deliberately left unscaled.
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden = int(hidden)
        self.out_features = int(out_features)
        self.first = nn.Linear(in_features, hidden)
        self.second = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


def fixed_batch(
    seed: int, features: int, batch_size: int = 32, classes: int = 4
) -> dict[str, torch.Tensor]:
    """A deterministic batch that does not touch the global RNG."""

    generator = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(batch_size, features, generator=generator),
        "y": torch.randint(0, classes, (batch_size,), generator=generator),
    }


def batch_stream(
    seed: int, features: int, steps: int, batch_size: int = 32, classes: int = 4
) -> list[dict[str, torch.Tensor]]:
    """A fixed, replayable minibatch sequence shared by both members of a pair."""

    return [
        fixed_batch(seed * 10_000 + step, features, batch_size=batch_size, classes=classes)
        for step in range(steps)
    ]
