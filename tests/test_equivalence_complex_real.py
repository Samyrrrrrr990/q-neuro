"""QE-000001: the historical complex / exact-real pair, certified honestly."""

from __future__ import annotations

import math

import pytest
import torch

from qneuro.equivalence import EquivalenceLevel, MapSpec
from qneuro.equivalence.complex_real import (
    CRITICAL_POLE_RADIUS,
    ComplexToExactRealMap,
    distance_to_nearest_tanh_pole,
    record_complex_tanh_inputs,
)
from qneuro.models.equivalent import ExactRealBlockOperatorState
from qneuro.models.operators import ComplexOperatorState

KWARGS = {"num_tokens": 16, "pad_token": 15, "state_dim": 8, "rank": 2, "num_classes": 5}


def build_pair() -> tuple[ComplexOperatorState, ExactRealBlockOperatorState]:
    torch.manual_seed(7)
    complex_model = ComplexOperatorState(**KWARGS)
    real_model = ExactRealBlockOperatorState(**KWARGS)
    real_model.copy_from_complex(complex_model)
    return complex_model, real_model


def sample_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    mask = torch.ones(12, 10, dtype=torch.bool)
    mask[:, -2:] = torch.rand(12, 2, generator=generator) > 0.4
    return {
        "tokens": torch.randint(0, 15, (12, 10), generator=generator),
        "mask": mask,
        "vector": torch.rand(12, 6, generator=generator),
        "label": torch.randint(0, 5, (12,), generator=generator),
    }


def test_pole_distance_is_zero_at_a_pole_and_periodic() -> None:
    poles = torch.tensor([1j * (2 * k + 1) * math.pi / 2 for k in range(-2, 3)])
    assert torch.allclose(distance_to_nearest_tanh_pole(poles), torch.zeros(5), atol=1e-6)
    offset = torch.tensor([0.25 + 1j * math.pi / 2])
    assert float(distance_to_nearest_tanh_pole(offset)) == pytest.approx(0.25, abs=1e-6)


def test_map_is_declared_transport_degenerate() -> None:
    mapping = ComplexToExactRealMap()
    assert mapping.is_identity
    assert mapping.transport_degenerate


def test_the_map_may_not_declare_a_globally_exact_level() -> None:
    """The historical naming implied E1; the spec must refuse it alongside the pole domain."""

    with pytest.raises(ValueError, match="domain restriction"):
        MapSpec(
            name="complex_to_exact_real",
            family="complex_real",
            declared_level=EquivalenceLevel.E1,
            invertible=True,
            domain=ComplexToExactRealMap().spec.domain,
        )


def test_tanh_instrumentation_restores_the_original_operator() -> None:
    original = torch.tanh
    with record_complex_tanh_inputs() as observed:
        torch.tanh(torch.complex(torch.zeros(3), torch.zeros(3)))
        assert observed["complex_calls"] == 1.0
    assert torch.tanh is original


def test_certificate_declares_e2_degeneracy_and_pole_margin() -> None:
    complex_model, real_model = build_pair()
    certificate = ComplexToExactRealMap().certify(complex_model, real_model, sample_batch())

    assert certificate.declared_level is EquivalenceLevel.E2
    assert certificate.transport_degenerate is True
    assert certificate.domain is not None
    # The parameter map really is the identity, bitwise.
    assert certificate.residuals["parameter_identity"] == 0.0
    assert certificate.residuals["max_logit"] < 1e-4
    assert certificate.residuals["critical_pole_radius"] == CRITICAL_POLE_RADIUS["float32"]
    assert math.isfinite(certificate.residuals["minimum_tanh_pole_distance"])
    assert any("Transport-degenerate" in mode for mode in certificate.known_failure_modes)
    assert any("E1 fails" in mode for mode in certificate.known_failure_modes)


def test_certificate_flags_a_run_that_reaches_the_excluded_region() -> None:
    """If training ever approaches a pole, the certificate must say it does not cover the run."""

    complex_model, real_model = build_pair()
    with torch.no_grad():
        # Drive the pre-activation onto a pole through the injection term.
        complex_model.injection_imag.fill_(math.pi / 2)
        complex_model.injection_real.zero_()
        complex_model.left_real.zero_()
        complex_model.left_imag.zero_()
        real_model.copy_from_complex(complex_model)

    certificate = ComplexToExactRealMap().certify(complex_model, real_model, sample_batch())
    assert certificate.residuals["minimum_tanh_pole_distance"] <= CRITICAL_POLE_RADIUS["float32"]
    assert any("REACHED" in mode for mode in certificate.known_failure_modes)
