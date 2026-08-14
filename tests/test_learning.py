"""Tests for experimental Q-Neuro learning laws."""

from __future__ import annotations

import torch

from qneuro.learning import AuxiliaryTrainingModel, apply_phase_gradient, multi_objective_losses
from qneuro.models import ComplexOperatorState


def _batch() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([[8, 18, 26], [9, 19, 27], [10, 20, 28]]),
        "mask": torch.ones(3, 3, dtype=torch.bool),
        "vector": torch.zeros(3, 82),
        "label": torch.tensor([8, 9, 10]),
        "is_order": torch.zeros(3, dtype=torch.bool),
        "order_complete": torch.ones(3, dtype=torch.bool),
    }


def test_phase_gradient_installs_finite_gradients_and_reports_phases() -> None:
    base = ComplexOperatorState(80, 80, 8, 2, 20, 0.25)
    model = AuxiliaryTrainingModel(base)
    losses = multi_objective_losses(model, _batch(), auxiliary_weight=0.2)
    diagnostics = apply_phase_gradient(losses, model)
    assert 0.0 <= diagnostics["phase_radians_mechanism"] <= torch.pi / 2
    assert 0.0 <= diagnostics["phase_radians_localization"] <= torch.pi / 2
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_auxiliary_losses_skip_undefined_order_twin_factors() -> None:
    batch = _batch()
    batch["label"] = torch.tensor([0, 1, 2])
    base = ComplexOperatorState(80, 80, 8, 2, 20, 0.25)
    losses = multi_objective_losses(AuxiliaryTrainingModel(base), batch, 0.2)
    assert set(losses) == {"diagnosis"}
