import torch

from qneuro.observables import (
    HermitianObservableProbe,
    LinearObservableProbe,
    factorial_property_labels,
    fit_probe,
)


def test_factorial_labels_and_hermitian_observable() -> None:
    labels = torch.arange(8, 20)
    properties = factorial_property_labels(labels)
    assert set(properties) == {"mechanism", "localization", "temporality", "context"}
    assert properties["mechanism"].max() == 4
    probe = HermitianObservableProbe(state_dim=3, num_classes=4)
    matrices = probe.matrices()
    assert torch.allclose(matrices, matrices.conj().transpose(-2, -1), atol=1e-6)
    representation = torch.randn(7, 6)
    logits = probe(representation)
    assert logits.shape == (7, 4)
    assert bool(torch.isfinite(logits).all())


def test_linear_probe_fits_separable_frozen_state() -> None:
    generator = torch.Generator().manual_seed(61)
    targets = torch.arange(3).repeat(30)
    centers = torch.eye(3)[targets]
    representation = centers + 0.03 * torch.randn((90, 3), generator=generator)
    _, metrics = fit_probe(
        lambda: LinearObservableProbe(3, 3),
        representation[:60],
        targets[:60],
        representation[60:],
        targets[60:],
        seed=61,
        epochs=80,
    )
    assert metrics["accuracy"] > 0.95
