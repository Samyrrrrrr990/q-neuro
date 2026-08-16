"""A genuinely ``torch.complex``-parameterized model and its realification map.

This is the family the audit said was missing. Q-Neuro's historical `ComplexOperatorState` holds no
complex leaf parameter — it stores real tensors and builds complex ones inside the forward pass — so
its "exact real" control shares its coordinates and the pair is transport-degenerate. Here the
complex model really does carry ``complex64`` parameters, so the map to real coordinates is a
genuine change of parameterization: one complex tensor becomes two real tensors, and the dtype
changes.

Transport algebra
-----------------
For ``W = A + iB`` acting on ``z = x + iy``::

    W z = (Ax - By) + i(Bx + Ay)

PyTorch's complex autograd stores the *conjugate* Wirtinger derivative, so for a real-valued loss
``p.grad`` equals ``dL/dRe(p) + i dL/dIm(p)``. Gradients therefore split exactly like parameters,
and no extra factor or conjugation is required.

The optimizer is the interesting part, and the answer is measured rather than assumed: PyTorch
treats a complex parameter through ``view_as_real``, keeping **per-component** second moments
(``exp_avg_sq.real = beta2-weighted grad.real**2``) rather than modulus-based ones. Consequently a
complex parameter and its real pair take bitwise identical AdamW steps. The map is
**non-degenerate but exactly conjugate**, which is a much stronger statement than the degenerate
historical case and is what makes the whole complex/real family optimizer-inert in this framework.

The activation is a split ``tanh`` rather than analytic complex ``tanh``. That is deliberate: it
keeps this experiment free of the pole discrepancy studied in QE-000001, so the optimizer question
is not confounded with a numerical one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qneuro.equivalence.maps import ParameterMap
from qneuro.equivalence.spec import EquivalenceLevel, MapSpec

#: Optimizer-state entries that are per-parameter tensors and split like the parameter itself.
_SPLIT_STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer")

EPS = 1e-8


def split_tanh(real: torch.Tensor, imaginary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Componentwise ``tanh``: pole-free, so it cannot confound the optimizer measurement."""

    return torch.tanh(real), torch.tanh(imaginary)


class NativeComplexMLP(nn.Module):
    """Two complex layers with genuine ``complex64`` leaf parameters and a Born-style readout."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden = int(hidden)
        self.out_features = int(out_features)
        scale = hidden**-0.5
        self.first_weight = nn.Parameter(
            torch.complex(torch.randn(hidden, in_features), torch.randn(hidden, in_features))
            * scale
        )
        self.first_bias = nn.Parameter(torch.zeros(hidden, dtype=torch.complex64))
        self.second_weight = nn.Parameter(
            torch.complex(torch.randn(out_features, hidden), torch.randn(out_features, hidden))
            * scale
        )
        self.second_bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.complex(x, torch.zeros_like(x))
        pre = z @ self.first_weight.T + self.first_bias
        hidden_real, hidden_imaginary = split_tanh(pre.real, pre.imag)
        hidden = torch.complex(hidden_real, hidden_imaginary)
        amplitude = hidden @ self.second_weight.T + self.second_bias
        return torch.log(amplitude.real.square() + amplitude.imag.square() + EPS)


class RealifiedComplexMLP(nn.Module):
    """The same function in real coordinates, with the complex algebra written out."""

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden = int(hidden)
        self.out_features = int(out_features)
        self.first_weight_real = nn.Parameter(torch.zeros(hidden, in_features))
        self.first_weight_imag = nn.Parameter(torch.zeros(hidden, in_features))
        self.first_bias_real = nn.Parameter(torch.zeros(hidden))
        self.first_bias_imag = nn.Parameter(torch.zeros(hidden))
        self.second_weight_real = nn.Parameter(torch.zeros(out_features, hidden))
        self.second_weight_imag = nn.Parameter(torch.zeros(out_features, hidden))
        self.second_bias_real = nn.Parameter(torch.zeros(out_features))
        self.second_bias_imag = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (A + iB)(x + i0) = Ax + i Bx
        pre_real = x @ self.first_weight_real.T + self.first_bias_real
        pre_imaginary = x @ self.first_weight_imag.T + self.first_bias_imag
        hidden_real, hidden_imaginary = split_tanh(pre_real, pre_imaginary)
        # (A + iB)(u + iv) = (Au - Bv) + i(Bu + Av)
        amplitude_real = (
            hidden_real @ self.second_weight_real.T
            - hidden_imaginary @ self.second_weight_imag.T
            + self.second_bias_real
        )
        amplitude_imaginary = (
            hidden_real @ self.second_weight_imag.T
            + hidden_imaginary @ self.second_weight_real.T
            + self.second_bias_imag
        )
        return torch.log(amplitude_real.square() + amplitude_imaginary.square() + EPS)


class ComplexRealificationMap(ParameterMap):
    """``C^n -> R^(2n)``: one complex tensor becomes two real tensors.

    Genuinely non-degenerate — the coordinate systems and dtypes differ — yet exactly conjugate,
    because PyTorch optimizes complex parameters through ``view_as_real``.
    """

    def __init__(self, names: tuple[str, ...] | None = None):
        self.names = names or (
            "first_weight",
            "first_bias",
            "second_weight",
            "second_bias",
        )
        super().__init__(
            MapSpec(
                name="complex_realification",
                family="native_complex_real",
                declared_level=EquivalenceLevel.E2,
                invertible=True,
                notes=(
                    "Non-degenerate: the source holds complex64 leaf parameters and the target "
                    "holds real pairs. Conjugate under SGD and AdamW because PyTorch keeps "
                    "per-component moments for complex parameters rather than modulus-based ones."
                ),
            )
        )

    def _split(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.names:
            value = tensors.get(name)
            if value is None:
                continue
            detached = value.detach()
            out[f"{name}_real"] = detached.real.clone()
            out[f"{name}_imag"] = detached.imag.clone()
        return out

    def _join(self, tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.names:
            real = tensors.get(f"{name}_real")
            imaginary = tensors.get(f"{name}_imag")
            if real is None or imaginary is None:
                continue
            out[name] = torch.complex(real.detach(), imaginary.detach()).clone()
        return out

    def map_parameters(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._split(source)

    def unmap_parameters(self, target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._join(target)

    def map_gradients(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """PyTorch stores ``dL/dRe(p) + i dL/dIm(p)``, so gradients split like parameters."""

        return self._split(source)

    @property
    def is_identity(self) -> bool:
        return False

    def _map_optimizer_state(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        mapped: dict[str, dict[str, Any]] = {}
        for name, entries in state.items():
            if name not in self.names:
                continue
            real_entries: dict[str, Any] = {}
            imaginary_entries: dict[str, Any] = {}
            for key, value in entries.items():
                if key in _SPLIT_STATE_KEYS and isinstance(value, torch.Tensor):
                    real_entries[key] = value.detach().real.clone()
                    imaginary_entries[key] = value.detach().imag.clone()
                elif isinstance(value, torch.Tensor):
                    real_entries[key] = value.detach().clone()
                    imaginary_entries[key] = value.detach().clone()
                else:
                    real_entries[key] = value
                    imaginary_entries[key] = value
            mapped[f"{name}_real"] = real_entries
            mapped[f"{name}_imag"] = imaginary_entries
        return mapped

    def build_target(self, source: NativeComplexMLP) -> RealifiedComplexMLP:
        target = RealifiedComplexMLP(source.in_features, source.hidden, source.out_features)
        mapped = self.map_parameters(dict(source.named_parameters()))
        with torch.no_grad():
            for name, parameter in target.named_parameters():
                parameter.copy_(mapped[name])
        return target


def isolated_optimizer_conjugacy(
    optimizer_name: str,
    *,
    steps: int = 25,
    weight_decay: float = 0.0,
    epsilon: float = 1e-8,
    size: int = 6,
    seed: int = 0,
) -> float:
    """Compare a complex parameter with its real pair under identical supplied gradients.

    Feeding the same gradients to both sides isolates the optimizer from the forward and backward
    passes, so the number returned is a property of the update rule alone.
    """

    torch.manual_seed(seed)
    real_init, imaginary_init = torch.randn(size), torch.randn(size)
    gradients = [(torch.randn(size), torch.randn(size)) for _ in range(steps)]

    def build(parameters: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if optimizer_name == "sgd":
            return torch.optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(parameters, lr=1e-2, weight_decay=weight_decay, eps=epsilon)
        raise ValueError(f"unknown optimizer: {optimizer_name!r}")

    complex_parameter = nn.Parameter(torch.complex(real_init.clone(), imaginary_init.clone()))
    complex_optimizer = build([complex_parameter])
    real_parameter = nn.Parameter(real_init.clone())
    imaginary_parameter = nn.Parameter(imaginary_init.clone())
    real_optimizer = build([real_parameter, imaginary_parameter])

    worst = 0.0
    for real_gradient, imaginary_gradient in gradients:
        complex_optimizer.zero_grad(set_to_none=True)
        real_optimizer.zero_grad(set_to_none=True)
        complex_parameter.grad = torch.complex(real_gradient.clone(), imaginary_gradient.clone())
        real_parameter.grad = real_gradient.clone()
        imaginary_parameter.grad = imaginary_gradient.clone()
        complex_optimizer.step()
        real_optimizer.step()
        with torch.no_grad():
            worst = max(
                worst,
                float((complex_parameter.real - real_parameter).abs().max()),
                float((complex_parameter.imag - imaginary_parameter).abs().max()),
            )
    return worst
