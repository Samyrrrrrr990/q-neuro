"""Contract tests for the equivalence specification and certificate interface.

These tests define the interface before the first QE experiment runs. Several of them assert that
the framework *refuses* an unsupported claim; those are as important as the positive cases.
"""

from __future__ import annotations

import json

import pytest
import torch

from qneuro.equivalence import (
    Certificate,
    DomainRestriction,
    EquivalenceLevel,
    HiddenUnitPermutationMap,
    IdentityMap,
    MapSpec,
    TransportLevel,
    residual_summary,
)
from qneuro.equivalence.microcosms import TwoLayerMLP


def test_equivalence_levels_are_ordered_with_e0_strongest() -> None:
    assert EquivalenceLevel.E0.is_at_least(EquivalenceLevel.E3)
    assert not EquivalenceLevel.E3.is_at_least(EquivalenceLevel.E0)
    assert EquivalenceLevel.E2.is_at_least(EquivalenceLevel.E2)
    assert not EquivalenceLevel.FAILED.is_at_least(EquivalenceLevel.E4)


def test_transport_levels_are_ordered() -> None:
    assert TransportLevel.T5 > TransportLevel.T0
    assert TransportLevel.T4 > TransportLevel.T2


def test_declaring_e0_or_e1_with_a_domain_restriction_is_refused() -> None:
    """A globally exact level cannot be claimed alongside an excluded region."""

    domain = DomainRestriction(
        description="excludes a neighbourhood of the complex tanh poles",
        excluded="min_k |delta - i(2k+1)pi/2| <= rho_c",
        radius=1.55e-3,
        dtype="float32",
    )
    for level in (EquivalenceLevel.E0, EquivalenceLevel.E1):
        with pytest.raises(ValueError, match="domain restriction"):
            MapSpec(
                name="bad",
                family="test",
                declared_level=level,
                invertible=True,
                domain=domain,
            )


def test_identity_map_is_flagged_transport_degenerate() -> None:
    spec = MapSpec(
        name="complex_to_exact_real",
        family="complex_real",
        declared_level=EquivalenceLevel.E2,
        invertible=True,
    )
    identity = IdentityMap(spec)
    assert identity.is_identity
    assert identity.transport_degenerate


def test_permutation_map_is_not_transport_degenerate() -> None:
    model = TwoLayerMLP(4, 6, 3)
    mapping = HiddenUnitPermutationMap.random(model, seed=0)
    assert not mapping.is_identity
    assert not mapping.transport_degenerate


def test_certificate_refuses_to_serialize_without_a_declared_level() -> None:
    with pytest.raises(TypeError, match="declared_level"):
        Certificate(
            source="a",
            target="b",
            map_name="m",
            declared_level=None,  # type: ignore[arg-type]
            transport_level=TransportLevel.T2,
            transport_degenerate=False,
            dtype="float32",
            device="cpu",
            residuals={},
        )


def test_certificate_round_trips_through_json_with_schema_version() -> None:
    certificate = Certificate(
        source="TwoLayerMLP",
        target="TwoLayerMLP(permuted)",
        map_name="hidden_unit_permutation",
        declared_level=EquivalenceLevel.E2,
        transport_level=TransportLevel.T4,
        transport_degenerate=False,
        dtype="float32",
        device="cpu",
        residuals={"max_logit": 0.0},
    )
    payload = json.loads(certificate.to_json())
    assert payload["schema_version"] == Certificate.SCHEMA_VERSION
    assert payload["declared_level"] == "E2"
    assert payload["transport_degenerate"] is False
    restored = Certificate.from_json(certificate.to_json())
    assert restored == certificate


def test_certificate_records_a_downgrade_rather_than_hiding_it() -> None:
    certificate = Certificate(
        source="a",
        target="b",
        map_name="m",
        declared_level=EquivalenceLevel.E1,
        transport_level=TransportLevel.T2,
        transport_degenerate=False,
        dtype="float32",
        device="cpu",
        residuals={"max_logit": 1.0},
    )
    downgraded = certificate.downgrade(
        EquivalenceLevel.E3, reason="max logit residual exceeds the E2 tolerance"
    )
    assert downgraded.declared_level is EquivalenceLevel.E3
    assert downgraded.downgrades == (("E1", "E3", "max logit residual exceeds the E2 tolerance"),)
    # A downgrade may never be used to strengthen a claim.
    with pytest.raises(ValueError, match="stronger"):
        downgraded.downgrade(EquivalenceLevel.E0, reason="wishful")


def test_optimizer_state_transport_is_not_assumed_by_default() -> None:
    """A map must implement optimizer transport explicitly; silence is not consent."""

    spec = MapSpec(
        name="unimplemented",
        family="test",
        declared_level=EquivalenceLevel.E2,
        invertible=False,
    )

    class BareMap(IdentityMap):
        pass

    bare = BareMap(spec)
    bare.supports_optimizer_transport = False
    with pytest.raises(NotImplementedError, match="optimizer"):
        bare.map_optimizer_state({})


def test_residual_summary_reports_max_absolute_difference() -> None:
    a = {"w": torch.tensor([1.0, 2.0])}
    b = {"w": torch.tensor([1.0, 2.5])}
    summary = residual_summary(a, b)
    assert summary["max_absolute"] == pytest.approx(0.5)
    assert summary["max_relative"] == pytest.approx(0.2)
