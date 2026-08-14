import math

import torch

from qneuro.evaluation import (
    ambiguity_pair_metrics,
    binary_auroc,
    nearest_centroid_scores,
    ood_metrics,
    silhouette_binary,
)


def test_binary_auroc_and_ood_metrics() -> None:
    assert binary_auroc(torch.tensor([0.0, 0.1]), torch.tensor([0.9, 1.0])) == 1.0
    id_logits = torch.tensor([[8.0, 0.0], [0.0, 8.0]])
    ood_logits = torch.zeros((2, 2))
    metrics = ood_metrics(id_logits, ood_logits)
    assert metrics["ood_auroc_msp"] == 1.0
    assert metrics["ood_mean_confidence"] < metrics["id_mean_confidence"]


def test_ambiguity_metrics_reward_balanced_twin_mass() -> None:
    logits = torch.log(torch.tensor([[0.49, 0.49, 0.02], [0.49, 0.49, 0.02]]))
    metrics = ambiguity_pair_metrics(logits, torch.tensor([0, 1]))
    assert math.isclose(metrics["ambiguity_twin_mass"], 0.98, rel_tol=1e-6)
    assert math.isclose(metrics["ambiguity_twin_balance"], 1.0, rel_tol=1e-6)
    assert metrics["ambiguity_identical_prediction_rate"] == 1.0


def test_representation_geometry_metrics() -> None:
    train = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = torch.tensor([0, 0, 1, 1])
    query = torch.tensor([[0.05, 0.0], [10.0, 10.0]])
    scores = nearest_centroid_scores(train, labels, query)
    assert scores[0] < scores[1]
    assert silhouette_binary(train[:2], train[2:]) > 0.9
