"""QE-000004: dense versus factorized linear maps.

The first family where the honest answer is that the framework cannot transport. These tests
assert the refusals as firmly as they assert the equivalences.
"""

from __future__ import annotations

import pytest
import torch

from qneuro.equivalence import FactorizedToDenseMap
from qneuro.equivalence.factorization import (
    DenseLinear,
    FactorizedLinear,
    align_factorized_to_dense,
)
from qneuro.equivalence.microcosms import fixed_batch


def build_pair(seed: int = 0, rank: int = 6) -> tuple[DenseLinear, FactorizedLinear]:
    torch.manual_seed(seed)
    dense = DenseLinear(6, 4)
    factorized = FactorizedLinear(6, 4, rank=rank)
    align_factorized_to_dense(dense, factorized)
    return dense, factorized


def test_full_rank_factorization_realizes_the_same_predictor() -> None:
    dense, factorized = build_pair(rank=4)
    batch = fixed_batch(seed=2, features=6)
    with torch.no_grad():
        assert torch.allclose(dense(batch["x"]), factorized(batch["x"]), atol=1e-5, rtol=1e-5)


def test_product_of_factors_reproduces_the_dense_weight() -> None:
    dense, factorized = build_pair(rank=4)
    mapping = FactorizedToDenseMap()
    mapped = mapping.map_parameters(dict(factorized.named_parameters()))
    assert torch.allclose(mapped["weight"], dense.weight, atol=1e-5, rtol=1e-5)


def test_map_is_not_transport_degenerate_and_declares_itself_non_invertible() -> None:
    mapping = FactorizedToDenseMap()
    assert not mapping.is_identity
    assert not mapping.transport_degenerate
    assert mapping.spec.invertible is False


def test_inverse_is_refused_because_the_map_is_non_injective() -> None:
    mapping = FactorizedToDenseMap()
    with pytest.raises(NotImplementedError, match="non-injective"):
        mapping.unmap_parameters({"weight": torch.eye(4)})


def test_gradient_transport_is_refused() -> None:
    mapping = FactorizedToDenseMap()
    with pytest.raises(NotImplementedError, match="do not transport"):
        mapping.map_gradients({"left": torch.ones(4, 4), "right": torch.ones(4, 6)})


def test_optimizer_state_transport_is_refused_rather_than_faked() -> None:
    """The framework must decline; a silent pass-through here would be the original sin."""

    mapping = FactorizedToDenseMap()
    assert mapping.supports_optimizer_transport is False
    with pytest.raises(NotImplementedError, match="optimizer"):
        mapping.map_optimizer_state({"left": {"exp_avg": torch.ones(4, 4)}})


def test_factor_descent_induces_a_preconditioned_flow_on_the_product() -> None:
    """The implicit bias is measured, not asserted: the induced direction differs from dL/dW."""

    dense, factorized = build_pair(rank=4)
    batch = fixed_batch(seed=8, features=6)

    torch.nn.functional.cross_entropy(dense(batch["x"]), batch["y"]).backward()
    dense_gradient = dense.weight.grad.detach()

    loss = torch.nn.functional.cross_entropy(factorized(batch["x"]), batch["y"])
    loss.backward()

    mapping = FactorizedToDenseMap()
    induced = mapping.induced_preconditioned_gradient(
        factorized.left.detach(), factorized.right.detach(), dense_gradient
    )

    # The induced direction is a genuine preconditioning: same subspace, different geometry.
    cosine = torch.nn.functional.cosine_similarity(
        induced.flatten(), dense_gradient.flatten(), dim=0
    )
    assert 0.0 < float(cosine) < 0.9999
    assert not torch.allclose(induced, dense_gradient, atol=1e-4)


def test_rank_deficient_factorization_is_not_certified_as_equivalent() -> None:
    """A rank too low to represent the dense map must show a visible residual."""

    dense, factorized = build_pair(rank=1)
    mapping = FactorizedToDenseMap()
    mapped = mapping.map_parameters(dict(factorized.named_parameters()))
    residual = float((mapped["weight"] - dense.weight.detach()).abs().max())
    assert residual > 1e-3
