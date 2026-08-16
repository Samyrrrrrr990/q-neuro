"""Homogeneous scaling orbits: the program's first non-degenerate map.

For a positively homogeneous activation, ``relu(s·z) = s·relu(z)`` for ``s > 0``, so

    W1 -> s·W1,   b1 -> s·b1,   W2 -> W2/s,   b2 -> b2

leaves the realized predictor exactly unchanged while moving the parameters along an orbit. Unlike
a permutation, this map performs arithmetic, and the derivations below are where the covariance
defects come from.

Transport algebra
-----------------
Write ``s_p`` for the scale applied to parameter ``p``. Because the loss is unchanged,

    grad' = grad / s_p                  (cotangent, inverse to the parameter)
    exp_avg' = exp_avg / s_p            (first moment accumulates gradients)
    exp_avg_sq' = exp_avg_sq / s_p^2    (second moment accumulates squared gradients)

Plain SGD is conjugate iff the learning rate is also transported as ``eta -> eta·s_p^2``::

    target of a source step : s_p(theta - eta·g)   = s_p·theta - eta·s_p·g
    actual step in target   : s_p·theta - eta'·g/s_p

which agree exactly when ``eta' = eta·s_p^2``.

Adam's update ``m_hat / (sqrt(v_hat) + eps)`` is *scale free* in the gradient: substituting the
transported moments gives ``m_hat / (sqrt(v_hat) + s_p·eps)``, which is the source update again
rather than ``s_p`` times it. Adam therefore needs ``eta -> eta·s_p`` and remains inexact through
the ``eps`` term.

Decoupled weight decay resists both. The decay term ``-eta·lambda·theta'`` is covariant under the
*untransported* learning rate and off by ``s_p^2`` under the transported one. **No single
learning-rate policy transports the gradient step and decoupled weight decay simultaneously**,
which is a structural, measurable property of this family rather than a tuning artifact.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qneuro.equivalence.maps import ParameterMap
from qneuro.equivalence.spec import EquivalenceLevel, MapSpec

#: How each optimizer-state entry scales, as a power of the parameter's gradient scale (1/s_p).
_STATE_GRADIENT_POWER = {
    "exp_avg": 1,
    "momentum_buffer": 1,
    "exp_avg_sq": 2,
    "max_exp_avg_sq": 2,
}

#: Learning-rate exponent on s_p that makes each optimizer family conjugate, where one exists.
_LEARNING_RATE_EXPONENT = {"sgd": 2.0, "sgd_momentum": 2.0, "adam": 1.0, "adamw": 1.0}


class DiagonalScalingMap(ParameterMap):
    """A per-parameter positive rescaling with derived gradient and optimizer transport."""

    def __init__(self, scales: Mapping[str, float], spec: MapSpec | None = None):
        if any(value <= 0.0 for value in scales.values()):
            raise ValueError(
                "scaling factors must be strictly positive to preserve relu homogeneity"
            )
        self.scales = dict(scales)
        super().__init__(
            spec
            or MapSpec(
                name="diagonal_scaling",
                family="scaling_orbit",
                declared_level=EquivalenceLevel.E2,
                invertible=True,
                notes="Exact continuous symmetry of a positively homogeneous network.",
            )
        )

    def _scale(self, name: str) -> float:
        return self.scales.get(name, 1.0)

    def map_parameters(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: (tensor.detach() * self._scale(name)).clone() for name, tensor in source.items()
        }

    def unmap_parameters(self, target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: (tensor.detach() / self._scale(name)).clone() for name, tensor in target.items()
        }

    def map_gradients(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: (tensor.detach() / self._scale(name)).clone() for name, tensor in source.items()
        }

    @property
    def is_identity(self) -> bool:
        return all(value == 1.0 for value in self.scales.values())

    def learning_rate_scales(self, optimizer_name: str) -> dict[str, float]:
        exponent = _LEARNING_RATE_EXPONENT.get(optimizer_name)
        if exponent is None:
            return {}
        return {name: float(scale) ** exponent for name, scale in self.scales.items()}

    def _map_optimizer_state(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        mapped: dict[str, dict[str, Any]] = {}
        for name, entries in state.items():
            gradient_scale = 1.0 / self._scale(name)
            converted: dict[str, Any] = {}
            for key, value in entries.items():
                power = _STATE_GRADIENT_POWER.get(key)
                if power is not None and isinstance(value, torch.Tensor):
                    converted[key] = (value.detach() * gradient_scale**power).clone()
                elif isinstance(value, torch.Tensor):
                    converted[key] = value.detach().clone()
                else:
                    converted[key] = value
            mapped[name] = converted
        return mapped


class HomogeneousScalingMap(DiagonalScalingMap):
    """The two-layer homogeneous scaling orbit ``(W2/s)(s·W1)``."""

    def __init__(self, scale: float, *, first: str = "first", second: str = "second"):
        self.scale = float(scale)
        super().__init__(
            {
                f"{first}.weight": self.scale,
                f"{first}.bias": self.scale,
                f"{second}.weight": 1.0 / self.scale,
            },
            spec=MapSpec(
                name="homogeneous_scaling_orbit",
                family="scaling_orbit",
                declared_level=EquivalenceLevel.E2,
                invertible=True,
                notes=(
                    "Exact for positively homogeneous activations and strictly positive scale. "
                    "The parameter map is bitwise exact when the scale is a power of two; other "
                    "scales introduce a rounding term that must not be confused with a covariance "
                    "defect."
                ),
            ),
        )

    @property
    def is_bitwise_exact(self) -> bool:
        """True when the scale is a power of two, so multiplication is exact in binary floats."""

        mantissa, _ = torch.frexp(torch.tensor(self.scale, dtype=torch.float64))
        return bool(float(mantissa) == 0.5)

    def build_target(self, source: nn.Module) -> nn.Module:
        target = copy.deepcopy(source)
        mapped = self.map_parameters(dict(source.named_parameters()))
        with torch.no_grad():
            for name, parameter in target.named_parameters():
                parameter.copy_(mapped[name])
        return target
