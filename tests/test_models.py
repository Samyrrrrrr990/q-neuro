import math

import torch

from neuroworld import NeuroWorld
from qneuro.calibration import apply_temperature, fit_temperature
from qneuro.data import collate_cases
from qneuro.metrics import aggregate_seed_metrics, classification_metrics
from qneuro.model_factory import build_model, parameter_count
from qneuro.models import ComplexOperatorState, ExactRealBlockOperatorState, RealOperatorState


def sample_batch() -> dict[str, torch.Tensor]:
    return collate_cases(NeuroWorld().generate(8, seed=3))


def test_parameter_budget_matching() -> None:
    batch = sample_batch()
    parameter_budget = 20_304
    for name in (
        "mlp",
        "transformer",
        "causal_transformer",
        "gru",
        "vanilla_rnn",
        "lstm",
        "residual_gated_recurrence",
        "dense_real_recurrence",
        "orthogonal_real_recurrence",
        "real_operator",
        "two_channel_operator",
        "complex_operator",
        "exact_real_block_operator",
        "unrestricted_paired_real_operator",
        "real_polar_operator",
        "real_rotation_block_operator",
    ):
        model, metadata = build_model(name, parameter_budget, rank=2, max_length=40, step_size=0.35)
        assert metadata["parameter_count"] == parameter_count(model)
        assert abs(parameter_count(model) - parameter_budget) / parameter_budget <= 0.02
        logits = model(**batch)
        assert logits.shape == (8, NeuroWorld.num_diagnoses)
        assert torch.isfinite(logits).all()

    complex_metadata = build_model(
        "complex_operator", parameter_budget, rank=2, max_length=40, step_size=0.35
    )[1]
    exact_metadata = build_model(
        "exact_real_block_operator", parameter_budget, rank=2, max_length=40, step_size=0.35
    )[1]
    polar_metadata = build_model(
        "real_polar_operator", parameter_budget, rank=2, max_length=40, step_size=0.35
    )[1]
    assert complex_metadata["state_real_dof"] == 48
    assert exact_metadata["state_real_dof"] == polar_metadata["state_real_dof"] == 48


def test_complex_probabilities_are_normalized_and_finite() -> None:
    batch = sample_batch()
    model = ComplexOperatorState(80, 80, state_dim=12, rank=2, num_classes=20)
    probabilities = model.probabilities(batch["tokens"], batch["mask"], batch["vector"])
    assert torch.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(8), atol=1e-6)


def test_exact_real_block_matches_complex_forward_loss_and_gradients() -> None:
    torch.manual_seed(71)
    batch = sample_batch()
    complex_model = ComplexOperatorState(80, 80, state_dim=8, rank=2, num_classes=20)
    real_block = ExactRealBlockOperatorState(80, 80, state_dim=8, rank=2, num_classes=20)
    real_block.copy_from_complex(complex_model)

    complex_logits = complex_model(**batch)
    real_logits = real_block(**batch)
    assert torch.allclose(complex_logits, real_logits, atol=2e-5, rtol=2e-5)

    complex_loss = torch.nn.functional.cross_entropy(complex_logits, batch["label"])
    real_loss = torch.nn.functional.cross_entropy(real_logits, batch["label"])
    complex_loss.backward()
    real_loss.backward()
    for (complex_name, complex_parameter), (real_name, real_parameter) in zip(
        complex_model.named_parameters(), real_block.named_parameters(), strict=True
    ):
        assert complex_name == real_name
        assert complex_parameter.grad is not None
        assert real_parameter.grad is not None
        assert torch.allclose(complex_parameter.grad, real_parameter.grad, atol=5e-5, rtol=5e-4), (
            complex_name
        )


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


def test_complex_trajectory_ends_at_evolved_state() -> None:
    model = ComplexOperatorState(80, 80, 12, 2, 20)
    tokens = torch.tensor([[1, 4, 7, 80], [2, 5, 80, 80]])
    mask = tokens.ne(80)
    vector = torch.zeros(2, 82)
    trajectory = model.trajectory(tokens, mask, vector)
    assert trajectory.shape == (2, 5, 12)
    assert torch.allclose(trajectory[:, -1], model.evolve(tokens, mask, vector), atol=1e-6)


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


def test_order_metrics_separate_complete_and_incomplete_evidence() -> None:
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 1])
    is_order = torch.tensor([True, True, True])
    order_complete = torch.tensor([True, False, False])
    metrics = classification_metrics(logits, labels, is_order, order_complete)
    assert metrics["complete_order_accuracy"] == 1.0
    assert metrics["incomplete_order_accuracy"] == 0.5
    assert math.isclose(metrics["order_evidence_complete_rate"], 1.0 / 3.0, rel_tol=1e-6)


def test_temperature_scaling_preserves_predictions_and_improves_validation_nll() -> None:
    logits = torch.tensor([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0], [0.0, 8.0]])
    labels = torch.tensor([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    calibrated = apply_temperature(logits, temperature)
    assert temperature > 0.0
    assert torch.equal(logits.argmax(dim=-1), calibrated.argmax(dim=-1))
    assert torch.nn.functional.cross_entropy(
        calibrated, labels
    ) <= torch.nn.functional.cross_entropy(logits, labels)
