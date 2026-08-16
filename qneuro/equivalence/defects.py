"""Residual and defect metrics.

Both a parameter-space and a predictive-space quantity are always reported. The parameter one is
coordinate dependent and therefore never primary; ML2-PREREG-001 section 2 makes the predictive one
authoritative.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def residual_summary(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    """Maximum absolute and relative disagreement over a set of named tensors."""

    shared = sorted(set(left) & set(right))
    if not shared:
        return {"max_absolute": 0.0, "max_relative": 0.0, "tensors": 0}

    max_absolute = 0.0
    max_relative = 0.0
    for name in shared:
        a = left[name].detach().double()
        b = right[name].detach().double()
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {name!r}: {tuple(a.shape)} vs {tuple(b.shape)}")
        difference = (a - b).abs()
        scale = torch.maximum(a.abs(), b.abs())
        max_absolute = max(max_absolute, float(difference.max()))
        relative = torch.where(scale > 0, difference / scale.clamp_min(1e-300), torch.zeros_like(a))
        max_relative = max(max_relative, float(relative.max()))
    return {"max_absolute": max_absolute, "max_relative": max_relative, "tensors": len(shared)}


def predictive_divergence(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Divergence between two logit tensors, in logit and probability space."""

    max_logit = float((left - right).abs().max())
    probabilities_left = torch.softmax(left.double(), dim=-1)
    probabilities_right = torch.softmax(right.double(), dim=-1)
    total_variation = float(
        0.5 * (probabilities_left - probabilities_right).abs().sum(dim=-1).max()
    )
    return {"max_logit": max_logit, "max_total_variation": total_variation}
