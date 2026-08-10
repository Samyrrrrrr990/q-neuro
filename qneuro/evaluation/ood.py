"""Distribution-shift and unknown-class detection metrics."""

from __future__ import annotations

import torch


def binary_auroc(negative_scores: torch.Tensor, positive_scores: torch.Tensor) -> float:
    """Probability that a positive has a higher score, with half credit for ties."""

    negative = negative_scores.detach().float().flatten()
    positive = positive_scores.detach().float().flatten()
    if negative.numel() == 0 or positive.numel() == 0:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


def fpr_at_tpr95(negative_scores: torch.Tensor, positive_scores: torch.Tensor) -> float:
    positive = positive_scores.detach().float().flatten()
    negative = negative_scores.detach().float().flatten()
    threshold = torch.quantile(positive, 0.05)
    return float((negative >= threshold).float().mean())


def ood_metrics(id_logits: torch.Tensor, ood_logits: torch.Tensor) -> dict[str, float]:
    id_probability = torch.softmax(id_logits.float(), dim=-1)
    ood_probability = torch.softmax(ood_logits.float(), dim=-1)
    id_msp_uncertainty = 1.0 - id_probability.max(dim=-1).values
    ood_msp_uncertainty = 1.0 - ood_probability.max(dim=-1).values
    id_entropy = -(id_probability * torch.log(id_probability.clamp_min(1e-12))).sum(dim=-1)
    ood_entropy = -(ood_probability * torch.log(ood_probability.clamp_min(1e-12))).sum(dim=-1)
    id_energy = -torch.logsumexp(id_logits.float(), dim=-1)
    ood_energy = -torch.logsumexp(ood_logits.float(), dim=-1)
    return {
        "ood_auroc_msp": binary_auroc(id_msp_uncertainty, ood_msp_uncertainty),
        "ood_auroc_entropy": binary_auroc(id_entropy, ood_entropy),
        "ood_auroc_energy": binary_auroc(id_energy, ood_energy),
        "ood_fpr95_msp": fpr_at_tpr95(id_msp_uncertainty, ood_msp_uncertainty),
        "id_mean_confidence": float(id_probability.max(dim=-1).values.mean()),
        "ood_mean_confidence": float(ood_probability.max(dim=-1).values.mean()),
        "id_mean_entropy": float(id_entropy.mean()),
        "ood_mean_entropy": float(ood_entropy.mean()),
    }
