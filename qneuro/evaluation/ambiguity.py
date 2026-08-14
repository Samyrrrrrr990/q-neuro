"""Metrics for observationally identical cases with two equally valid labels."""

from __future__ import annotations

import torch


def ambiguity_pair_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if logits.shape[0] % 2:
        raise ValueError("ambiguity cases must be ordered in pairs")
    probabilities = torch.softmax(logits.float(), dim=-1).reshape(-1, 2, logits.shape[-1])
    pair_labels = labels.reshape(-1, 2)
    first_probability = probabilities[:, 0]
    label_a = pair_labels[:, 0]
    label_b = pair_labels[:, 1]
    probability_a = first_probability.gather(1, label_a[:, None]).squeeze(1)
    probability_b = first_probability.gather(1, label_b[:, None]).squeeze(1)
    twin_mass = probability_a + probability_b
    balance = 1.0 - torch.abs(probability_a - probability_b) / twin_mass.clamp_min(1e-12)
    entropy = -(first_probability * torch.log(first_probability.clamp_min(1e-12))).sum(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    pair_nll = -0.5 * (
        torch.log(probability_a.clamp_min(1e-12)) + torch.log(probability_b.clamp_min(1e-12))
    )
    return {
        "ambiguity_twin_mass": float(twin_mass.mean()),
        "ambiguity_twin_balance": float(balance.mean()),
        "ambiguity_entropy": float(entropy.mean()),
        "ambiguity_pair_nll": float(pair_nll.mean()),
        "ambiguity_identical_prediction_rate": float(
            predictions[:, 0].eq(predictions[:, 1]).float().mean()
        ),
    }
