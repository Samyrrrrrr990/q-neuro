"""QE-000006: a genuinely torch.complex-parameterized family and its realification.

Unlike the historical Q-Neuro pair, this map is non-degenerate: the source really does carry
complex64 leaf parameters. The result is that a non-degenerate complex/real map turns out to be
exactly conjugate anyway, because PyTorch optimizes complex parameters through view_as_real.
"""

from __future__ import annotations

import pytest
import torch

from qneuro.equivalence import paired_training_divergence
from qneuro.equivalence.native_complex import (
    ComplexRealificationMap,
    NativeComplexMLP,
    RealifiedComplexMLP,
    isolated_optimizer_conjugacy,
)


def build_pair(
    seed: int = 0,
) -> tuple[NativeComplexMLP, RealifiedComplexMLP, ComplexRealificationMap]:
    torch.manual_seed(seed)
    source = NativeComplexMLP(6, 12, 4)
    mapping = ComplexRealificationMap()
    return source, mapping.build_target(source), mapping


def test_source_really_holds_complex_parameters() -> None:
    """The premise of the family: unlike ComplexOperatorState, these leaves are complex."""

    source, target, _ = build_pair()
    assert all(torch.is_complex(p) for p in source.parameters())
    assert not any(torch.is_complex(p) for p in target.parameters())


def test_map_is_not_transport_degenerate() -> None:
    _, _, mapping = build_pair()
    assert not mapping.is_identity
    assert not mapping.transport_degenerate


def test_realification_preserves_the_forward_function() -> None:
    source, target, _ = build_pair()
    x = torch.randn(16, 6, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert torch.allclose(source(x), target(x), atol=1e-5, rtol=1e-5)


def test_parameters_round_trip_exactly() -> None:
    source, _, mapping = build_pair()
    parameters = dict(source.named_parameters())
    restored = mapping.unmap_parameters(mapping.map_parameters(parameters))
    for name, tensor in parameters.items():
        assert torch.equal(restored[name], tensor.detach()), name


def test_gradients_split_like_parameters_under_the_wirtinger_convention() -> None:
    """PyTorch stores dL/dRe(p) + i dL/dIm(p), so no conjugation or factor of two appears."""

    source, target, mapping = build_pair()
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(16, 6, generator=generator)
    y = torch.randint(0, 4, (16,), generator=generator)

    torch.nn.functional.cross_entropy(source(x), y).backward()
    torch.nn.functional.cross_entropy(target(x), y).backward()

    mapped = mapping.map_gradients(
        {n: p.grad for n, p in source.named_parameters() if p.grad is not None}
    )
    for name, parameter in target.named_parameters():
        assert torch.allclose(mapped[name], parameter.grad, atol=1e-5, rtol=1e-4), name


@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_complex_adamw_is_bitwise_conjugate_to_the_real_pair(weight_decay: float) -> None:
    """The decisive measurement, with the forward pass factored out entirely.

    PyTorch keeps per-component moments for complex parameters, so complex AdamW *is* real AdamW
    on the (re, im) pair. Complex parameterization buys no optimizer geometry.
    """

    assert isolated_optimizer_conjugacy("adamw", steps=25, weight_decay=weight_decay) == 0.0


def test_a_large_epsilon_does_not_break_complex_adam_conjugacy() -> None:
    """If Adam used a modulus-based second moment, epsilon would expose it. It does not."""

    assert isolated_optimizer_conjugacy("adamw", steps=25, epsilon=1e-1) == 0.0


@pytest.mark.parametrize("weight_decay", [0.0, 1e-2])
def test_complex_sgd_is_conjugate_up_to_kernel_rounding_only(weight_decay: float) -> None:
    """SGD is conjugate as an update rule but not bitwise as an implementation.

    The residual first appears several steps in, stays at one unit in the last place, and is a
    property of complex kernel arithmetic (H4), not of optimizer geometry (H2). Asserting a hard
    zero here would be asserting an idealization the measurement does not support.
    """

    worst = isolated_optimizer_conjugacy("sgd", steps=25, weight_decay=weight_decay)
    assert 0.0 <= worst <= 4.0 * torch.finfo(torch.float32).eps


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_end_to_end_paired_training_stays_at_rounding_scale(optimizer_name: str) -> None:
    source, target, mapping = build_pair()
    result = paired_training_divergence(
        source,
        target,
        mapping,
        optimizer_name=optimizer_name,
        warmup_steps=10,
        measured_steps=20,
        transport_optimizer_state=True,
        seed=13,
    )
    tolerance = 1e4 * torch.finfo(torch.float32).eps * max(result["reference_logit_scale"], 1.0)
    assert result["max_logit_divergence"] <= tolerance, result
