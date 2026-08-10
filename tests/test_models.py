import math

import torch

from neuroworld import NeuroWorld
from qneuro.data import collate_cases
from qneuro.metrics import aggregate_seed_metrics
from qneuro.model_factory import build_model, parameter_count
from qneuro.models import ComplexOperatorState, RealOperatorState


def sample_batch() -> dict[str, torch.Tensor]:
    return collate_cases(NeuroWorld().generate(8, seed=3))


def test_parameter_budget_matching() -> None:
    batch = sample_batch()
    for name in ("mlp", "transformer", "real_operator", "complex_operator"):
        model, metadata = build_model(name, 20_000, rank=2, max_length=40, step_size=0.35)
        assert metadata["parameter_count"] == parameter_count(model)
        assert abs(parameter_count(model) - 20_000) / 20_000 < 0.08
        logits = model(**batch)
        assert logits.shape == (8, NeuroWorld.num_diagnoses)
        assert torch.isfinite(logits).all()


def test_complex_probabilities_are_normalized_and_finite() -> None:
    batch = sample_batch()
    model = ComplexOperatorState(80, 80, state_dim=12, rank=2, num_classes=20)
    probabilities = model.probabilities(batch["tokens"], batch["mask"], batch["vector"])
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(8), atol=1e-6)


def test_padding_does_not_change_operator_state() -> None:
    batch = sample_batch()
    tokens = torch.cat(
        [batch["tokens"], torch.full((8, 3), NeuroWorld.pad_token, dtype=torch.long)], dim=1
    )
    mask = torch.cat([batch["mask"], torch.zeros((8, 3), dtype=torch.bool)], dim=1)
    for model in (
        RealOperatorState(80, 80, state_dim=12, rank=2, num_classes=20),
        ComplexOperatorState(80, 80, state_dim=8, rank=2, num_classes=20),
    ):
        original = model.evolve(batch["tokens"], batch["mask"])
        padded = model.evolve(tokens, mask)
        assert torch.allclose(original, padded, atol=1e-6)


def test_operator_order_can_change_state() -> None:
    tokens_ab = torch.tensor([[0, 1]])
    tokens_ba = torch.tensor([[1, 0]])
    mask = torch.ones_like(tokens_ab, dtype=torch.bool)
    for model in (
        RealOperatorState(80, 80, state_dim=12, rank=2, num_classes=20),
        ComplexOperatorState(80, 80, state_dim=8, rank=2, num_classes=20),
    ):
        state_ab = model.evolve(tokens_ab, mask)
        state_ba = model.evolve(tokens_ba, mask)
        assert not torch.allclose(state_ab, state_ba)
        assert model.commutator_norm(0, 1) > 0.0
        norm = torch.sqrt(torch.sum(torch.abs(state_ab).square(), dim=-1))
        assert torch.allclose(norm, torch.tensor([math.sqrt(model.state_dim)]), atol=1e-5)


def test_seed_summary_uses_student_t_interval() -> None:
    summary = aggregate_seed_metrics([{"score": 0.0}, {"score": 1.0}, {"score": 2.0}])["score"]
    expected_half_width = 4.303 / math.sqrt(3)
    assert summary["ci_method"] == "student_t"
    assert math.isclose(summary["mean"], 1.0)
    assert math.isclose(summary["std"], 1.0)
    assert math.isclose(summary["ci95_high"], 1.0 + expected_half_width, rel_tol=1e-6)
