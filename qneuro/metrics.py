"""Standard predictive and calibration metrics."""

from __future__ import annotations

import torch


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_order: torch.Tensor,
    n_bins: int = 10,
) -> dict[str, float]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    predictions = probabilities.argmax(dim=-1)
    target_probability = probabilities.gather(1, labels[:, None]).squeeze(1).clamp_min(1e-12)
    top_k = probabilities.topk(k=min(3, probabilities.shape[1]), dim=-1).indices
    confidence = probabilities.max(dim=-1).values
    correct = predictions.eq(labels)

    ece = torch.zeros((), dtype=torch.float32)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            weight = in_bin.float().mean()
            ece += weight * torch.abs(correct[in_bin].float().mean() - confidence[in_bin].mean())

    if is_order.any():
        order_accuracy = correct[is_order].float().mean()
    else:
        order_accuracy = torch.tensor(float("nan"))
    return {
        "top1": float(correct.float().mean()),
        "top3": float((top_k == labels[:, None]).any(dim=1).float().mean()),
        "nll": float(-torch.log(target_probability).mean()),
        "ece": float(ece),
        "order_accuracy": float(order_accuracy),
    }


def aggregate_seed_metrics(seed_results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Calculate mean, sample standard deviation, and normal-approximation 95% CI."""

    if not seed_results:
        return {}
    keys = sorted(set.intersection(*(set(result) for result in seed_results)))
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = torch.tensor([result[key] for result in seed_results], dtype=torch.float64)
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            continue
        mean = finite.mean()
        std = finite.std(unbiased=True) if finite.numel() > 1 else torch.zeros_like(mean)
        half_width = 1.96 * std / finite.numel() ** 0.5
        summary[key] = {
            "mean": float(mean),
            "std": float(std),
            "ci95_low": float(mean - half_width),
            "ci95_high": float(mean + half_width),
            "n": int(finite.numel()),
        }
    return summary
