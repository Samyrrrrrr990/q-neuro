"""Representation extraction and geometry metrics."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_representations(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    representations: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for raw_batch in loader:
        batch = {key: value.to(device) for key, value in raw_batch.items()}
        representation = model.encode(**batch)
        representations.append(representation.detach().float().cpu())
        labels.append(batch["label"].detach().cpu())
    return torch.cat(representations), torch.cat(labels)


def _standardize(reference: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    mean = reference.mean(dim=0, keepdim=True)
    scale = reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
    return (value - mean) / scale


def nearest_centroid_scores(
    train_representations: torch.Tensor,
    train_labels: torch.Tensor,
    query_representations: torch.Tensor,
) -> torch.Tensor:
    train_standardized = _standardize(train_representations, train_representations)
    query_standardized = _standardize(train_representations, query_representations)
    centroids = torch.stack(
        [
            train_standardized[train_labels == label].mean(dim=0)
            for label in torch.unique(train_labels, sorted=True)
        ]
    )
    return torch.cdist(query_standardized, centroids).min(dim=1).values


def silhouette_binary(first: torch.Tensor, second: torch.Tensor, max_per_group: int = 500) -> float:
    first = first[:max_per_group].float()
    second = second[:max_per_group].float()
    combined = torch.cat([first, second])
    standardized = _standardize(combined, combined)
    distances = torch.cdist(standardized, standardized)
    first_count = first.shape[0]
    labels = torch.cat(
        [torch.zeros(first_count, dtype=torch.bool), torch.ones(second.shape[0], dtype=torch.bool)]
    )
    silhouettes: list[torch.Tensor] = []
    for index in range(standardized.shape[0]):
        same = labels == labels[index]
        same[index] = False
        other = ~labels.eq(labels[index])
        a = distances[index, same].mean()
        b = distances[index, other].mean()
        silhouettes.append((b - a) / torch.maximum(a, b).clamp_min(1e-8))
    return float(torch.stack(silhouettes).mean())
