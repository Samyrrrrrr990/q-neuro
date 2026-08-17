"""QE-000002 zero-defect positive control.

Hidden-unit permutation is an exact discrete symmetry. Under a correctly permuted optimizer state,
paired training must stay at floating-point rounding scale. Under a deliberately un-permuted
optimizer state, the defect must be detectable. A control that cannot detect the deliberate defect
is not a working instrument, so both directions are asserted here.
"""

from __future__ import annotations

import pytest
import torch

from qneuro.equivalence import HiddenUnitPermutationMap, paired_training_divergence
from qneuro.equivalence.microcosms import TwoLayerMLP, fixed_batch


def build_pair(seed: int = 0) -> tuple[TwoLayerMLP, TwoLayerMLP, HiddenUnitPermutationMap]:
    torch.manual_seed(seed)
    source = TwoLayerMLP(6, 12, 4)
    mapping = HiddenUnitPermutationMap.random(source, seed=seed + 1)
    target = mapping.build_target(source)
    return source, target, mapping


def test_permutation_preserves_the_forward_function() -> None:
    source, target, _ = build_pair()
    batch = fixed_batch(seed=3, features=6)
    with torch.no_grad():
        assert torch.allclose(source(batch["x"]), target(batch["x"]), atol=1e-6, rtol=1e-6)


def test_permutation_round_trips_parameters_exactly() -> None:
    source, _, mapping = build_pair()
    forward = mapping.map_parameters(dict(source.named_parameters()))
    back = mapping.unmap_parameters(forward)
    for name, tensor in source.named_parameters():
        assert torch.equal(back[name], tensor.detach()), name


def test_permutation_transports_gradients() -> None:
    source, target, mapping = build_pair()
    batch = fixed_batch(seed=4, features=6)
    loss_source = torch.nn.functional.cross_entropy(source(batch["x"]), batch["y"])
    loss_target = torch.nn.functional.cross_entropy(target(batch["x"]), batch["y"])
    loss_source.backward()
    loss_target.backward()

    mapped = mapping.map_parameters(
        {name: p.grad for name, p in source.named_parameters() if p.grad is not None}
    )
    for name, parameter in target.named_parameters():
        assert torch.allclose(mapped[name], parameter.grad, atol=1e-6, rtol=1e-5), name


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_transported_optimizer_state_keeps_the_pair_at_rounding_scale(optimizer_name: str) -> None:
    """The positive control: correct transport must produce a near-zero defect."""

    source, target, mapping = build_pair()
    result = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name=optimizer_name,
        warmup_steps=15,
        measured_steps=25,
        transport_optimizer_state=True,
        seed=11,
    )
    tolerance = 100.0 * torch.finfo(torch.float32).eps * max(result["reference_logit_scale"], 1.0)
    assert result["max_logit_divergence"] <= tolerance, result


def test_untransported_adamw_state_produces_a_detectable_defect() -> None:
    """The negative control: the instrument must notice a deliberately broken transport."""

    source, target, mapping = build_pair()
    transported = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name="adamw",
        warmup_steps=15,
        measured_steps=25,
        transport_optimizer_state=True,
        seed=11,
    )
    broken = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name="adamw",
        warmup_steps=15,
        measured_steps=25,
        transport_optimizer_state=False,
        seed=11,
    )
    assert broken["max_logit_divergence"] > 1e3 * transported["max_logit_divergence"]
    assert broken["max_logit_divergence"] > 1e-4


def test_permutation_certificate_declares_level_and_non_degeneracy() -> None:
    source, target, mapping = build_pair()
    certificate = mapping.certify(source, target, batch=fixed_batch(seed=7, features=6))
    assert certificate.transport_degenerate is False
    assert certificate.declared_level.is_at_least_e3()
    assert certificate.residuals["max_logit"] < 1e-5
    assert "parameter_round_trip" in certificate.residuals
