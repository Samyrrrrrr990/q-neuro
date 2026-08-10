"""Standard predictive and calibration metrics."""

from __future__ import annotations

import torch

_T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    is_order: torch.Tensor,
    order_complete: torch.Tensor | None = None,
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
    metrics = {
        "top1": float(correct.float().mean()),
        "top3": float((top_k == labels[:, None]).any(dim=1).float().mean()),
        "nll": float(-torch.log(target_probability).mean()),
        "ece": float(ece),
        "order_accuracy": float(order_accuracy),
    }
    if order_complete is not None:
        resolvable = is_order & order_complete
        ambiguous = is_order & ~order_complete
        metrics["complete_order_accuracy"] = (
            float(correct[resolvable].float().mean()) if resolvable.any() else float("nan")
        )
        metrics["incomplete_order_accuracy"] = (
            float(correct[ambiguous].float().mean()) if ambiguous.any() else float("nan")
        )
        metrics["order_evidence_complete_rate"] = (
            float(order_complete[is_order].float().mean()) if is_order.any() else float("nan")
        )
    return metrics


def aggregate_seed_metrics(seed_results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Calculate mean, sample standard deviation, and two-sided Student-t 95% CI."""

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
        degrees_of_freedom = max(1, int(finite.numel()) - 1)
        critical_value = _T_CRITICAL_975.get(degrees_of_freedom, 1.96)
        half_width = critical_value * std / finite.numel() ** 0.5
        summary[key] = {
            "mean": float(mean),
            "std": float(std),
            "ci95_low": float(mean - half_width),
            "ci95_high": float(mean + half_width),
            "ci_method": "student_t",
            "n": int(finite.numel()),
        }
    return summary
