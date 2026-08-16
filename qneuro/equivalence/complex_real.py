"""The historical Q-Neuro complex / exact-real pair, wrapped as a flagged map.

This is rung 4 of the ladder, and it is the reason the ladder exists. The pair is
**transport-degenerate**: `ComplexOperatorState` stores only real-valued parameters and builds
complex tensors inside its forward pass, so `ExactRealBlockOperatorState` shares its coordinates
exactly and the parameter map is the identity. Agreement between them is therefore close to a
self-consistency check, and it can bear on H4 (numerical implementation) and nothing else.

See `docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md` section 2.

Equivalence is also **not global**. The two implementations of complex `tanh` differ by
construction near the poles at `i(2k+1)pi/2`: the realified path clamps its denominator to
`finfo.eps` while native `torch.tanh` does not, so the native side overflows to `inf`/`nan` while
the realified side returns a finite value. The excluded radius is measured, not derived.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Mapping

import torch

from qneuro.equivalence.certificate import Certificate
from qneuro.equivalence.defects import residual_summary
from qneuro.equivalence.maps import IdentityMap
from qneuro.equivalence.spec import DomainRestriction, EquivalenceLevel, MapSpec, TransportLevel

#: Measured by bisection over 16 approach angles at 1e-3 relative tolerance. Several times larger
#: than the naive sqrt(eps/2) estimate, so the estimate must not be substituted for it.
CRITICAL_POLE_RADIUS = {"float32": 1.55e-3, "float64": 3.16e-8}

COMPLEX_TANH_DOMAIN = DomainRestriction(
    description=(
        "Excludes a neighbourhood of the complex tanh poles, where the realified implementation "
        "clamps its denominator and native torch.tanh does not."
    ),
    excluded="min_k |delta - i(2k+1)pi/2| <= rho_c",
    radius=CRITICAL_POLE_RADIUS["float32"],
    dtype="float32",
)


def distance_to_nearest_tanh_pole(values: torch.Tensor) -> torch.Tensor:
    """Distance from each complex value to the nearest pole of ``tanh`` at ``i(2k+1)pi/2``."""

    real = values.real if torch.is_complex(values) else values
    imaginary = values.imag if torch.is_complex(values) else torch.zeros_like(values)
    index = torch.round(imaginary / math.pi - 0.5)
    pole_imaginary = (2.0 * index + 1.0) * (math.pi / 2.0)
    return torch.sqrt(real.square() + (imaginary - pole_imaginary).square())


@contextlib.contextmanager
def record_complex_tanh_inputs() -> Iterator[dict[str, float]]:
    """Instrument ``torch.tanh`` to record how close its complex arguments come to a pole.

    Patching the operator rather than reimplementing the model's algebra keeps the measurement
    honest: it observes exactly the values the real model computes, with no duplicated arithmetic
    that could drift from the implementation it is meant to audit.
    """

    observed = {"minimum_pole_distance": math.inf, "calls": 0.0, "complex_calls": 0.0}
    original = torch.tanh

    def instrumented(input: torch.Tensor, *args, **kwargs):
        observed["calls"] += 1.0
        if torch.is_complex(input):
            observed["complex_calls"] += 1.0
            with torch.no_grad():
                closest = float(distance_to_nearest_tanh_pole(input.detach()).min())
            observed["minimum_pole_distance"] = min(observed["minimum_pole_distance"], closest)
        return original(input, *args, **kwargs)

    torch.tanh = instrumented  # type: ignore[assignment]
    try:
        yield observed
    finally:
        torch.tanh = original  # type: ignore[assignment]


class ComplexToExactRealMap(IdentityMap):
    """The historical complex / exact-real relationship, declared honestly.

    The map is the identity because both modules expose identically named, identically shaped
    real parameters. It is retained so the pair can be represented and audited, and so its
    degeneracy is visible in every certificate it produces.
    """

    def __init__(self) -> None:
        super().__init__(
            MapSpec(
                name="complex_to_exact_real",
                family="complex_real",
                declared_level=EquivalenceLevel.E2,
                invertible=True,
                domain=COMPLEX_TANH_DOMAIN,
                notes=(
                    "Transport-degenerate: ComplexOperatorState holds no complex leaf parameter, "
                    "so both modules share one real coordinate system and the parameter map is the "
                    "identity. Usable for H4 (numerical implementation) only."
                ),
            )
        )

    def certify(
        self,
        complex_model: torch.nn.Module,
        real_model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> Certificate:
        """Audit the pair and emit QE-000001's certificate, including pole reachability."""

        dtype = next(complex_model.parameters()).dtype
        dtype_name = str(dtype).removeprefix("torch.")
        radius = CRITICAL_POLE_RADIUS.get(dtype_name)

        complex_parameters = dict(complex_model.named_parameters())
        real_parameters = dict(real_model.named_parameters())
        parameter_residual = residual_summary(complex_parameters, real_parameters)

        with record_complex_tanh_inputs() as observed:
            complex_logits = complex_model(**batch)
        real_logits = real_model(**batch)

        complex_loss = torch.nn.functional.cross_entropy(complex_logits, batch["label"])
        real_loss = torch.nn.functional.cross_entropy(real_logits, batch["label"])
        complex_loss.backward()
        real_loss.backward()
        gradient_residual = residual_summary(
            {n: p.grad for n, p in complex_model.named_parameters() if p.grad is not None},
            {n: p.grad for n, p in real_model.named_parameters() if p.grad is not None},
        )
        complex_model.zero_grad(set_to_none=True)
        real_model.zero_grad(set_to_none=True)

        with torch.no_grad():
            probability_residual = float(
                (torch.softmax(complex_logits, dim=-1) - torch.softmax(real_logits, dim=-1))
                .abs()
                .max()
            )
            logit_residual = float((complex_logits - real_logits).abs().max())

        minimum_pole_distance = observed["minimum_pole_distance"]
        pole_margin = (
            minimum_pole_distance / radius
            if radius and math.isfinite(minimum_pole_distance)
            else None
        )

        residuals = {
            "parameter_identity": parameter_residual["max_absolute"],
            "max_logit": logit_residual,
            "max_probability": probability_residual,
            "max_gradient": gradient_residual["max_absolute"],
            "loss": abs(float(complex_loss.detach()) - float(real_loss.detach())),
            "minimum_tanh_pole_distance": minimum_pole_distance,
            "critical_pole_radius": radius if radius is not None else float("nan"),
            "pole_margin_multiples_of_radius": pole_margin
            if pole_margin is not None
            else float("nan"),
        }

        failure_modes = [
            (
                "E1 fails: native and realified complex tanh diverge inside the declared excluded "
                "region; the native side returns inf/nan where the realified side clamps."
            ),
            (
                "Transport-degenerate: no covariance defect can be non-zero, so this pair cannot "
                "test H1, H2, H3, H5, or H6."
            ),
        ]
        if radius is not None and minimum_pole_distance <= radius:
            failure_modes.append(
                f"REACHED: observed pole distance {minimum_pole_distance:.3e} is inside the "
                f"declared excluded radius {radius:.3e}; the certificate does not cover this run."
            )

        return Certificate(
            source=type(complex_model).__name__,
            target=type(real_model).__name__,
            map_name=self.name,
            declared_level=EquivalenceLevel.E2,
            transport_level=TransportLevel.T5,
            transport_degenerate=True,
            dtype=dtype_name,
            device=str(next(complex_model.parameters()).device),
            residuals=residuals,
            domain=self.spec.domain,
            known_failure_modes=tuple(failure_modes),
            notes=self.spec.notes,
        )
