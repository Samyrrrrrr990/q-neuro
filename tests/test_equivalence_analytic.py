"""QE-000008: the analytic microcosm and the finite-horizon transport bound.

The first test here is the invariant that matters most: **a bound must never be violated.** An
earlier implementation of the target update applied the inverse scale one time too many, which
produced ratios below one. That is not a finding, it is a bug, and this test is what catches it.
"""

from __future__ import annotations

import pytest
import torch

from qneuro.equivalence.analytic import LinearRegressionMicrocosm, transport_bound


def scaling(features: int, value: float) -> torch.Tensor:
    return torch.full((features,), value, dtype=torch.float64)


@pytest.mark.parametrize("condition_number", [1.0, 10.0, 100.0])
@pytest.mark.parametrize("step_fraction", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_the_bound_is_never_violated(
    condition_number: float, step_fraction: float, scale: float
) -> None:
    microcosm = LinearRegressionMicrocosm.build(200, 8, condition_number, seed=0)
    result = transport_bound(
        microcosm,
        scaling(8, scale),
        step_fraction * microcosm.stable_step_size(),
        steps=50,
    )
    if result["observed_at_numerical_floor"]:
        pytest.skip("observed divergence is at the float64 rounding floor")
    assert result["bound_parameter_divergence"] >= result["observed_parameter_divergence"] * (
        1.0 - 1e-9
    ), result


def test_identity_reparameterization_has_exactly_zero_defect() -> None:
    """S = I must give a zero defect and a zero divergence, with no tolerance needed."""

    microcosm = LinearRegressionMicrocosm.build(200, 8, 10.0, seed=0)
    result = transport_bound(
        microcosm, scaling(8, 1.0), 0.5 * microcosm.stable_step_size(), steps=50
    )
    assert result["mean_defect"] == 0.0
    assert result["max_defect"] == 0.0
    assert result["observed_parameter_divergence"] == 0.0


def test_condition_number_is_constructed_as_requested() -> None:
    microcosm = LinearRegressionMicrocosm.build(300, 10, 100.0, seed=1)
    spectrum = torch.linalg.eigvalsh(microcosm.hessian())
    assert float(spectrum.max() / spectrum.min()) == pytest.approx(100.0, rel=1e-6)


def test_gate_c_passes_on_the_analytic_microcosm() -> None:
    """Gate C: bound within a factor of 100 of the observed divergence."""

    worst = 0.0
    for condition_number in (1.0, 10.0, 100.0, 1000.0):
        microcosm = LinearRegressionMicrocosm.build(300, 10, condition_number, seed=0)
        for step_fraction in (0.1, 0.5, 0.9):
            for scale in (2.0, 4.0):
                result = transport_bound(
                    microcosm,
                    scaling(10, scale),
                    step_fraction * microcosm.stable_step_size(),
                    steps=200,
                )
                if not result["observed_at_numerical_floor"]:
                    worst = max(worst, result["bound_ratio"])
    assert 1.0 <= worst <= 100.0, worst


def test_a_triangle_inequality_lipschitz_constant_makes_the_bound_vacuous() -> None:
    """Section 6.10's claim, measured: how L is obtained decides whether the bound is usable."""

    microcosm = LinearRegressionMicrocosm.build(300, 10, 100.0, seed=0)
    result = transport_bound(
        microcosm, scaling(10, 2.0), 0.9 * microcosm.stable_step_size(), steps=200
    )
    assert result["target_lipschitz"] < 1.0 < result["naive_lipschitz"]
    assert result["bound_ratio"] < 100.0
    assert result["naive_bound_ratio"] > 1e6
