"""Parameter maps and the transport contract they must satisfy.

Design rule from ML2-PREREG-001 section 5: optimizer-state transport is never assumed. A map that
has not implemented it raises rather than silently passing state through, because a silent
pass-through is exactly the error that made the complex/exact-real pair look informative.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from qneuro.equivalence.certificate import Certificate
from qneuro.equivalence.defects import residual_summary
from qneuro.equivalence.spec import EquivalenceLevel, MapSpec, TransportLevel

#: Optimizer state entries that are per-parameter tensors and therefore transport like parameters
#: under an index-shuffling map. `step` is a scalar counter and is copied verbatim.
_TENSOR_STATE_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer")


class ParameterMap(ABC):
    """A semantics-preserving map between two parameterizations."""

    #: Maps must opt in to optimizer transport explicitly.
    supports_optimizer_transport: bool = True

    def __init__(self, spec: MapSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def map_parameters(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map source parameters (or any per-parameter tensor) into target coordinates."""

    @abstractmethod
    def unmap_parameters(self, target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Inverse of :meth:`map_parameters`."""

    def map_gradients(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transport gradients across the map.

        Gradients are cotangent vectors and do **not** generally transform like parameters. Maps
        that perform arithmetic must override this; reusing :meth:`map_parameters` on gradients is
        a silent correctness bug. Index-shuffling maps are the special case where the two coincide.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must state how gradients transport; they do not automatically "
            "follow the parameter map."
        )

    def learning_rate_scales(self, optimizer_name: str) -> dict[str, float]:
        """Per-parameter learning-rate factors that would make the update map conjugate.

        Returns an empty mapping when no such rescaling exists or none is needed.
        """

        del optimizer_name
        return {}

    @property
    @abstractmethod
    def is_identity(self) -> bool:
        """True when the map is the identity on a shared coordinate system."""

    @property
    def transport_degenerate(self) -> bool:
        """A degenerate pair cannot carry evidence about optimizer or prior geometry.

        See EQUIVALENCE_SCIENCE_AMENDMENT_001 section 2.2.
        """

        return self.is_identity

    def map_optimizer_state(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Transport optimizer state keyed by parameter name."""

        if not self.supports_optimizer_transport:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement optimizer state transport. "
                "Implement it or declare a transport level of T3 or lower; silence is not consent."
            )
        return self._map_optimizer_state(state)

    def _map_optimizer_state(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        raise NotImplementedError(
            f"{type(self).__name__} declares optimizer transport support but does not implement it."
        )


class IdentityMap(ParameterMap):
    """The identity. Retained so degenerate pairs can be represented and flagged honestly."""

    def map_parameters(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in source.items()}

    def unmap_parameters(self, target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in target.items()}

    def map_gradients(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.map_parameters(source)

    @property
    def is_identity(self) -> bool:
        return True

    def _map_optimizer_state(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(dict(state))


class IndexShuffleMap(ParameterMap):
    """Base for maps that reorder tensor entries without arithmetic.

    Because no arithmetic is performed, the parameter map is exact in floating point and optimizer
    state transports under exactly the same rule as parameters: adaptive optimizers are coordinate
    wise, so permuting the coordinates permutes the moments.
    """

    def __init__(
        self,
        spec: MapSpec,
        forward: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
        inverse: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
    ):
        super().__init__(spec)
        self._forward = dict(forward)
        self._inverse = dict(inverse)

    def _apply(
        self,
        table: Mapping[str, Callable[[torch.Tensor], torch.Tensor]],
        tensors: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            transform = table.get(name)
            source = tensor.detach()
            out[name] = source.clone() if transform is None else transform(source).clone()
        return out

    def map_parameters(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._apply(self._forward, source)

    def unmap_parameters(self, target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._apply(self._inverse, target)

    def map_gradients(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Reordering is an orthogonal (in fact permutation) map, so gradients follow parameters."""

        return self._apply(self._forward, source)

    @property
    def is_identity(self) -> bool:
        return not self._forward

    def _map_optimizer_state(
        self, state: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        mapped: dict[str, dict[str, Any]] = {}
        for name, entries in state.items():
            transform = self._forward.get(name)
            converted: dict[str, Any] = {}
            for key, value in entries.items():
                if key in _TENSOR_STATE_KEYS and isinstance(value, torch.Tensor):
                    converted[key] = (
                        value.detach().clone()
                        if transform is None
                        else transform(value.detach()).clone()
                    )
                elif isinstance(value, torch.Tensor):
                    converted[key] = value.detach().clone()
                else:
                    converted[key] = value
            mapped[name] = converted
        return mapped


class HiddenUnitPermutationMap(IndexShuffleMap):
    """Permute hidden units of a two-layer network and compensate the adjacent weights.

    For hidden permutation ``p``::

        W1' = W1[p]      b1' = b1[p]      W2' = W2[:, p]      b2' = b2

    The realized function is unchanged, exactly. This is the program's zero-defect positive
    control: under correctly permuted optimizer state the pair must stay at rounding scale, and
    under un-permuted state the defect must be detectable.
    """

    def __init__(self, permutation: torch.Tensor, *, first: str = "first", second: str = "second"):
        permutation = permutation.long()
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel())
        self.permutation = permutation
        self.inverse_permutation = inverse

        spec = MapSpec(
            name="hidden_unit_permutation",
            family="permutation_symmetry",
            declared_level=EquivalenceLevel.E2,
            invertible=True,
            notes=(
                "Exact discrete symmetry. Reordering performs no arithmetic, so the parameter map "
                "is bitwise exact; residual divergence originates in the second layer's reduction "
                "order, which is a genuine numerical-implementation effect."
            ),
        )
        super().__init__(
            spec,
            forward={
                f"{first}.weight": lambda t: t[permutation],
                f"{first}.bias": lambda t: t[permutation],
                f"{second}.weight": lambda t: t[:, permutation],
            },
            inverse={
                f"{first}.weight": lambda t: t[inverse],
                f"{first}.bias": lambda t: t[inverse],
                f"{second}.weight": lambda t: t[:, inverse],
            },
        )

    @classmethod
    def random(cls, model: nn.Module, seed: int) -> HiddenUnitPermutationMap:
        hidden = int(model.first.out_features)
        generator = torch.Generator().manual_seed(seed)
        return cls(torch.randperm(hidden, generator=generator))

    @property
    def is_identity(self) -> bool:
        return bool(torch.equal(self.permutation, torch.arange(self.permutation.numel())))

    def build_target(self, source: nn.Module) -> nn.Module:
        """Return a fresh module holding the mapped parameters."""

        target = copy.deepcopy(source)
        mapped = self.map_parameters(dict(source.named_parameters()))
        with torch.no_grad():
            for name, parameter in target.named_parameters():
                parameter.copy_(mapped[name])
        return target

    def certify(
        self, source: nn.Module, target: nn.Module, batch: Mapping[str, torch.Tensor]
    ) -> Certificate:
        """Audit this pair and emit a certificate.

        The declared level starts at the spec's claim and is downgraded, never upgraded, if the
        measured residuals do not support it.
        """

        source_parameters = dict(source.named_parameters())
        round_trip = residual_summary(
            source_parameters, self.unmap_parameters(self.map_parameters(source_parameters))
        )

        with torch.no_grad():
            source_logits = source(batch["x"])
            target_logits = target(batch["x"])
        logit_residual = float((source_logits - target_logits).abs().max())
        scale = float(source_logits.abs().max().clamp_min(1.0))

        source_loss = torch.nn.functional.cross_entropy(source(batch["x"]), batch["y"])
        target_loss = torch.nn.functional.cross_entropy(target(batch["x"]), batch["y"])
        source_loss.backward()
        target_loss.backward()
        gradient_residual = residual_summary(
            self.map_parameters(
                {n: p.grad for n, p in source.named_parameters() if p.grad is not None}
            ),
            {n: p.grad for n, p in target.named_parameters() if p.grad is not None},
        )
        source.zero_grad(set_to_none=True)
        target.zero_grad(set_to_none=True)

        residuals = {
            "parameter_round_trip": round_trip["max_absolute"],
            "max_logit": logit_residual,
            "max_gradient": gradient_residual["max_absolute"],
            "loss": abs(float(source_loss.detach()) - float(target_loss.detach())),
        }

        certificate = Certificate(
            source=type(source).__name__,
            target=f"{type(target).__name__}(permuted)",
            map_name=self.name,
            declared_level=self.spec.declared_level,
            transport_level=TransportLevel.T4,
            transport_degenerate=self.transport_degenerate,
            dtype=str(next(source.parameters()).dtype).removeprefix("torch."),
            device=str(next(source.parameters()).device),
            residuals=residuals,
            notes=self.spec.notes,
        )

        tolerance = 100.0 * torch.finfo(next(source.parameters()).dtype).eps * scale
        if logit_residual > tolerance:
            certificate = certificate.downgrade(
                EquivalenceLevel.E3,
                reason=f"max logit residual {logit_residual:.3e} exceeds E2 tolerance "
                f"{tolerance:.3e}",
            )
        return certificate
