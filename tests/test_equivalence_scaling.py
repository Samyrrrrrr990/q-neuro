"""QE-000003 scaling-orbit control.

For a positively homogeneous hidden layer, `W2 · relu(W1 x + b1)` is unchanged by
`W1 -> sW1, b1 -> sb1, W2 -> W2/s`. The realized predictor is identical, but the coordinates are
not, so this is the first family in the program with a genuinely non-zero covariance defect.

The scientific content of these tests is the decomposition: which parts of a practical training
system respect the equivalence, and which break it.
"""

from __future__ import annotations

import pytest
import torch

from qneuro.equivalence import HomogeneousScalingMap, paired_training_divergence
from qneuro.equivalence.microcosms import HomogeneousMLP, fixed_batch


def build_pair(
    scale: float = 2.0, seed: int = 0
) -> tuple[HomogeneousMLP, HomogeneousMLP, HomogeneousScalingMap]:
    torch.manual_seed(seed)
    source = HomogeneousMLP(6, 12, 4)
    mapping = HomogeneousScalingMap(scale)
    return source, mapping.build_target(source), mapping


def test_power_of_two_scaling_preserves_the_forward_function_bitwise() -> None:
    source, target, _ = build_pair(scale=2.0)
    batch = fixed_batch(seed=3, features=6)
    with torch.no_grad():
        assert torch.equal(source(batch["x"]), target(batch["x"]))


def test_non_power_of_two_scaling_preserves_the_function_to_rounding() -> None:
    source, target, _ = build_pair(scale=3.0)
    batch = fixed_batch(seed=3, features=6)
    with torch.no_grad():
        assert torch.allclose(source(batch["x"]), target(batch["x"]), atol=1e-5, rtol=1e-5)


def test_scaling_map_is_not_transport_degenerate() -> None:
    _, _, mapping = build_pair()
    assert not mapping.is_identity
    assert not mapping.transport_degenerate


def test_gradients_transport_inversely_to_parameters() -> None:
    """The defining non-covariance: parameters scale by s, gradients by 1/s."""

    source, target, mapping = build_pair(scale=2.0)
    batch = fixed_batch(seed=4, features=6)
    torch.nn.functional.cross_entropy(source(batch["x"]), batch["y"]).backward()
    torch.nn.functional.cross_entropy(target(batch["x"]), batch["y"]).backward()

    source_gradients = {n: p.grad for n, p in source.named_parameters() if p.grad is not None}
    mapped = mapping.map_gradients(source_gradients)
    for name, parameter in target.named_parameters():
        assert torch.allclose(mapped[name], parameter.grad, atol=1e-6, rtol=1e-4), name

    # Using the parameter map on gradients would be wrong, and must be visibly wrong.
    wrong = mapping.map_parameters(source_gradients)
    assert not torch.allclose(wrong["first.weight"], target.first.weight.grad, atol=1e-6)


def test_optimizer_moments_transport_at_first_and_second_order() -> None:
    _, _, mapping = build_pair(scale=2.0)
    state = {
        "first.weight": {
            "exp_avg": torch.ones(3, 3),
            "exp_avg_sq": torch.ones(3, 3),
            "step": torch.tensor(4.0),
        }
    }
    mapped = mapping.map_optimizer_state(state)
    # parameter scale s=2 -> gradient scale 1/2 -> exp_avg 1/2, exp_avg_sq 1/4
    assert torch.allclose(mapped["first.weight"]["exp_avg"], torch.full((3, 3), 0.5))
    assert torch.allclose(mapped["first.weight"]["exp_avg_sq"], torch.full((3, 3), 0.25))
    assert torch.equal(mapped["first.weight"]["step"], torch.tensor(4.0))


def test_plain_sgd_is_not_conjugate_under_scaling_even_with_transported_state() -> None:
    """The first genuine covariance defect in the program."""

    source, target, mapping = build_pair(scale=2.0)
    result = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name="sgd",
        warmup_steps=10,
        measured_steps=20,
        transport_optimizer_state=True,
        transport_learning_rate=False,
        weight_decay=0.0,
        seed=5,
    )
    assert result["max_logit_divergence"] > 1e-3


def test_transporting_the_learning_rate_restores_sgd_conjugacy() -> None:
    """SGD becomes conjugate exactly when eta -> eta * s^2. This is a derivation, not a fit."""

    source, target, mapping = build_pair(scale=2.0)
    result = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name="sgd",
        warmup_steps=10,
        measured_steps=20,
        transport_optimizer_state=True,
        transport_learning_rate=True,
        weight_decay=0.0,
        seed=5,
    )
    tolerance = 1e3 * torch.finfo(torch.float32).eps * max(result["reference_logit_scale"], 1.0)
    assert result["max_logit_divergence"] <= tolerance, result


def test_weight_decay_and_gradient_step_cannot_both_be_covariant() -> None:
    """No single learning-rate policy transports both terms simultaneously.

    With eta transported, the gradient term is covariant and the decay term is off by s^2.
    Turning decay on must therefore reintroduce a defect that was absent at weight_decay=0.
    """

    source, target, mapping = build_pair(scale=2.0)
    common = {
        "optimizer_name": "sgd",
        "warmup_steps": 10,
        "measured_steps": 20,
        "transport_optimizer_state": True,
        "transport_learning_rate": True,
        "seed": 5,
    }
    without_decay = paired_training_divergence(source, target, mapping, weight_decay=0.0, **common)
    with_decay = paired_training_divergence(source, target, mapping, weight_decay=1e-2, **common)
    assert with_decay["max_logit_divergence"] > 1e2 * without_decay["max_logit_divergence"]


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_defect_is_reported_rather_than_silently_zero(optimizer_name: str) -> None:
    """Native training across the orbit must show a measurable, reportable defect."""

    source, target, mapping = build_pair(scale=2.0)
    native = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name=optimizer_name,
        warmup_steps=10,
        measured_steps=20,
        transport_optimizer_state=False,
        transport_learning_rate=False,
        seed=5,
    )
    assert native["max_logit_divergence"] > 1e-4
