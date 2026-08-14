from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from experiments.run_mechanism_suite import aggregate_records, smoke_config
from neuroworld import NeuroWorld
from qneuro.data import collate_cases
from qneuro.model_factory import build_model
from qneuro.models import (
    CommutatorPenaltyComplexOperator,
    CommutingComplexOperatorState,
    FixedRandomComplexOperator,
    FrozenReadoutComplexOperator,
    MagnitudeDestroyedOperator,
    NoncommutativeRealOperator,
    PhaseDestroyedTrainingOperator,
)

MECHANISM_MODELS = (
    "commuting_operator",
    "commutator_penalty_operator",
    "noncommutative_real_operator",
    "phase_destroyed_training",
    "magnitude_destroyed",
    "no_conjugation_operator",
    "fixed_random_complex_operator",
    "frozen_dynamics",
    "frozen_readout",
    "fixed_two_state_attractor",
    "ambiguity_aware_real",
    "ambiguity_aware_complex",
)


def _batch() -> dict[str, torch.Tensor]:
    return collate_cases(NeuroWorld().generate(6, seed=6103))


def test_mechanism_models_have_finite_outputs_and_gradients() -> None:
    batch = _batch()
    for name in MECHANISM_MODELS:
        model, metadata = build_model(name, 8_000, rank=2, max_length=64, step_size=0.25)
        logits = model(**batch)
        assert logits.shape == (6, NeuroWorld.num_diagnoses), name
        assert bool(torch.isfinite(logits).all()), name
        loss = torch.nn.functional.cross_entropy(logits, batch["label"])
        auxiliary = getattr(model, "auxiliary_loss", None)
        if auxiliary is not None:
            loss = loss + auxiliary()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert gradients and all(gradient is not None for gradient in gradients), name
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients), name
        assert int(metadata["trainable_parameter_count"]) <= int(metadata["parameter_count"])


def test_commuting_control_has_zero_linear_commutator() -> None:
    model = CommutingComplexOperatorState(80, 80, 12, 20)
    assert model.commutator_norm(0, 1) == 0.0


def test_commutator_penalty_is_nonnegative_and_differentiable() -> None:
    model = CommutatorPenaltyComplexOperator(80, 80, 8, 2, 20)
    penalty = model.auxiliary_loss()
    assert penalty >= 0.0
    penalty.backward()
    assert model.left_real.grad is not None


def test_real_noncommutative_control_exposes_nonzero_commutator() -> None:
    model = NoncommutativeRealOperator(80, 20, 9, 0.25, readout_dim=20)
    assert model.commutator_norm(0, 1) > 0.0
    assert model.auxiliary_loss() <= 0.0


def test_phase_and_magnitude_interventions_enforce_declared_invariants() -> None:
    batch = _batch()
    phase_destroyed = PhaseDestroyedTrainingOperator(80, 80, 8, 2, 20)
    phase_state = phase_destroyed.evolve(batch["tokens"], batch["mask"], batch["vector"])
    assert torch.allclose(phase_state.imag, torch.zeros_like(phase_state.imag))

    magnitude_destroyed = MagnitudeDestroyedOperator(80, 80, 8, 2, 20)
    magnitude_state = magnitude_destroyed.evolve(batch["tokens"], batch["mask"], batch["vector"])
    expected = torch.ones_like(torch.abs(magnitude_state))
    assert torch.allclose(torch.abs(magnitude_state), expected, atol=1e-5)
    norms = torch.linalg.vector_norm(magnitude_state, dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, math.sqrt(8)), atol=1e-5)


def test_freezing_controls_change_only_declared_trainable_parameters() -> None:
    dynamics_frozen = FixedRandomComplexOperator(80, 80, 8, 2, 20)
    assert all(
        parameter.requires_grad == name.startswith("readout_")
        for name, parameter in dynamics_frozen.named_parameters()
    )
    readout_frozen = FrozenReadoutComplexOperator(80, 80, 8, 2, 20)
    assert not readout_frozen.readout_real.requires_grad
    assert not readout_frozen.readout_imag.requires_grad
    assert readout_frozen.left_real.requires_grad


def test_mechanism_config_covers_preregistered_controls() -> None:
    config = yaml.safe_load(Path("experiments/configs/mechanism_suite.yaml").read_text())
    names = set(config["models"]["names"])
    assert {
        "commuting_operator",
        "commutator_penalty_operator",
        "noncommutative_real_operator",
        "phase_destroyed_training",
        "fixed_random_complex_operator",
        "frozen_dynamics",
        "frozen_readout",
        "fixed_two_state_attractor",
        "ambiguity_aware_real",
        "ambiguity_aware_complex",
    } <= names
    smoke = smoke_config(config)
    assert smoke["profile"] == "smoke"
    assert len(smoke["dataset"]["world_seeds"]) == 1


def test_mechanism_aggregation_preserves_cells() -> None:
    records = [
        {
            "model": "control",
            "variant": "order",
            "severity": 1.0,
            "metrics": {"top1": 0.2 + 0.1 * index},
        }
        for index in range(3)
    ]
    summary = aggregate_records(records)
    metric = summary["control"]["order"]["1.0"]["top1"]
    assert metric["n"] == 3
    assert math.isclose(metric["mean"], 0.3)
    assert metric["std"] > 0.0
