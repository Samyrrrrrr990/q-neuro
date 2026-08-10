"""Evaluation metrics for uncertainty, OOD behavior, and representation geometry."""

from qneuro.evaluation.active import (
    ActiveStep,
    active_trajectory,
    aggregate_active_trajectories,
    canonicalize_case,
    estimate_positive_likelihoods,
    global_information_order,
    partial_case,
)
from qneuro.evaluation.ambiguity import ambiguity_pair_metrics
from qneuro.evaluation.ood import binary_auroc, ood_metrics
from qneuro.evaluation.representation import (
    collect_representations,
    nearest_centroid_scores,
    silhouette_binary,
)

__all__ = [
    "ActiveStep",
    "active_trajectory",
    "aggregate_active_trajectories",
    "ambiguity_pair_metrics",
    "binary_auroc",
    "canonicalize_case",
    "collect_representations",
    "estimate_positive_likelihoods",
    "global_information_order",
    "nearest_centroid_scores",
    "ood_metrics",
    "partial_case",
    "silhouette_binary",
]
