"""Validation-only post-hoc calibration utilities."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive temperature by validation NLL using deterministic LBFGS."""

    logits = logits.detach().float()
    labels = labels.detach().long()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.2,
        max_iter=60,
        tolerance_grad=1e-9,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature.clamp(-4.0, 4.0))
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature.detach().clamp(-4.0, 4.0)))


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return logits / float(temperature)
