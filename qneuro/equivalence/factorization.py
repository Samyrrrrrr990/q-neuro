"""Dense versus factorized linear maps: an exact map with no optimizer transport.

``W = U V`` realizes the same predictor as a dense ``W`` whenever the rank is sufficient, so the
forward map ``(U, V) -> UV`` is exact and semantics preserving. Everything else about this family
is different from the earlier rungs, and the differences are the point:

* the map is **non-injective**. Infinitely many ``(U, V)`` give the same ``W``, so no inverse
  exists and ``unmap_parameters`` must refuse rather than pick an arbitrary factorization;
* gradients transport in the forward direction only. With ``W = UV``,
  ``dL/dU = (dL/dW) V^T`` and ``dL/dV = U^T (dL/dW)``, so a dense gradient determines the
  factorized ones but not conversely;
* **there is no optimizer-state transport.** Gradient descent on ``(U, V)`` induces the flow
  ``dW/dt = -(dL/dW) V^T V - U U^T (dL/dW)`` on the product, which is a state-dependent
  preconditioning of dense gradient descent. That is a different update map, not a coordinate
  change of the same one, so no conjugating transformation of optimizer state exists.

The last point is why this family belongs in the ladder. It is the first case where the honest
answer is that the framework cannot transport, and the framework must say so rather than silently
passing state through.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from qneuro.equivalence.maps import ParameterMap
from qneuro.equivalence.spec import EquivalenceLevel, MapSpec


class DenseLinear(nn.Module):
    """``y = x W^T + b`` with a single dense weight."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def effective_weight(self) -> torch.Tensor:
        return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


class FactorizedLinear(nn.Module):
    """``y = x (U V)^T + b`` with the same realized function class when ``rank`` suffices."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.left = nn.Parameter(torch.empty(out_features, rank))
        self.right = nn.Parameter(torch.empty(rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.left, a=5**0.5)
        nn.init.kaiming_uniform_(self.right, a=5**0.5)

    def effective_weight(self) -> torch.Tensor:
        return self.left @ self.right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.effective_weight(), self.bias)


class FactorizedToDenseMap(ParameterMap):
    """``(U, V) -> UV``. Exact forward, non-invertible, and explicitly non-transportable."""

    #: Gradient descent on the factors is a preconditioned flow on the product, not a
    #: reparameterization of dense descent, so no conjugating optimizer-state map exists.
    supports_optimizer_transport = False

    def __init__(self, *, left: str = "left", right: str = "right", dense: str = "weight"):
        self.left = left
        self.right = right
        self.dense = dense
        super().__init__(
            MapSpec(
                name="factorized_to_dense",
                family="factorization",
                declared_level=EquivalenceLevel.E2,
                invertible=False,
                notes=(
                    "Exact in the factorized-to-dense direction only. The map is non-injective, so "
                    "no inverse exists, and gradient descent on the factors induces a "
                    "state-dependent preconditioner on the product rather than a coordinate change."
                ),
            )
        )

    def map_parameters(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped = {
            name: tensor.detach().clone()
            for name, tensor in source.items()
            if name not in {self.left, self.right}
        }
        if self.left in source and self.right in source:
            mapped[self.dense] = (source[self.left].detach() @ source[self.right].detach()).clone()
        return mapped

    def unmap_parameters(self, target: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "factorized_to_dense is non-injective: infinitely many (U, V) realize the same W, so "
            "there is no inverse to return. Choose an explicit factorization policy and register "
            "it as a separate map if one is needed."
        )

    def map_gradients(self, source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "gradients do not transport from factorized to dense coordinates. dL/dU = (dL/dW)V^T "
            "and dL/dV = U^T(dL/dW) determine the factor gradients from the dense one, not the "
            "reverse; the factor pair carries strictly more information than the product."
        )

    @property
    def is_identity(self) -> bool:
        return False

    def induced_preconditioned_gradient(
        self, left: torch.Tensor, right: torch.Tensor, dense_gradient: torch.Tensor
    ) -> torch.Tensor:
        """The effective update direction that factor-space descent induces on ``W = UV``.

        ``dW/dt = -(dL/dW) V^T V - U U^T (dL/dW)``. Comparing this with ``dL/dW`` is the cleanest
        available statement of the family's implicit bias, and it is measured rather than assumed.
        """

        return dense_gradient @ (right.T @ right) + (left @ left.T) @ dense_gradient


def align_factorized_to_dense(dense: DenseLinear, factorized: FactorizedLinear) -> None:
    """Set the factors so both modules realize the same function, via a truncated SVD.

    Requires ``rank >= min(in_features, out_features)`` for exactness; the caller is responsible
    for that, and the residual is reported by the certificate rather than assumed to be zero.
    """

    with torch.no_grad():
        u, s, vh = torch.linalg.svd(dense.weight, full_matrices=False)
        rank = factorized.rank
        factorized.left.copy_(u[:, :rank] * s[:rank].sqrt())
        factorized.right.copy_(s[:rank].sqrt().unsqueeze(-1) * vh[:rank])
        if factorized.bias is not None and dense.bias is not None:
            factorized.bias.copy_(dense.bias)
