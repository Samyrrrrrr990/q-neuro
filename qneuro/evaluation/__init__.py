"""Evaluation metrics for uncertainty, OOD behavior, and representation geometry."""

from qneuro.evaluation.ambiguity import ambiguity_pair_metrics
from qneuro.evaluation.ood import binary_auroc, ood_metrics
from qneuro.evaluation.representation import (
    collect_representations,
    nearest_centroid_scores,
    silhouette_binary,
)

__all__ = [
    "ambiguity_pair_metrics",
    "binary_auroc",
    "collect_representations",
    "nearest_centroid_scores",
    "ood_metrics",
    "silhouette_binary",
]
