"""Analytic microcosm where every term of the transport bound is computable in closed form.

Gate C of `docs/ML2_PREREGISTRATION_001.md` asks whether the finite-horizon bound

    e_{k+1} <= L_k · e_k + delta_k

is *non-vacuous* — within a prespecified multiplicative range of the divergence it is supposed to
bound. That question cannot be answered on a transport-degenerate pair, and it cannot be answered
honestly on a model whose Lipschitz constants have to be estimated. So it is answered here, on
linear regression under a diagonal reparameterization, where the Lipschitz constant is a spectral
norm and the one-step defect has a closed form.

Setup
-----
Least squares ``L(theta) = 1/(2n) ||X theta - y||^2`` gives ``H = X^T X / n`` and ``b = X^T y / n``.
Gradient descent is affine::

    U(theta)  = (I - eta·H) theta + eta·b

Under a diagonal reparameterization ``theta' = S theta`` the transported objective has gradient
``S^-1 (H S^-1 theta' - b)``, so the target update is::

    U'(theta') = (I - eta·S^-1 H S^-1) theta' + eta·S^-1 b

Both update maps are affine, so their Lipschitz constants are exactly the spectral norms of their
linear parts — no estimation, no slack.

The one-step covariance defect has a closed form::

    delta_k = || T(U(theta_k)) - U'(T(theta_k)) ||
            = eta · || (S^-1 - S)(H theta_k - b) ||

which vanishes exactly when ``S = I``, as it must.

With mapped initialization ``e_0 = 0`` the recursion unrolls to::

    e_K <= sum_i delta_i · L^(K-1-i)

and a predictive bound follows through the Lipschitz constant of the readout, ``L_F = ||X S^-1||``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearRegressionMicrocosm:
    """Least squares with a prescribed condition number and a diagonal reparameterization."""

    design: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def build(
        cls, samples: int, features: int, condition_number: float, seed: int
    ) -> LinearRegressionMicrocosm:
        """Construct a design matrix whose Gram spectrum spans exactly ``condition_number``."""

        generator = torch.Generator().manual_seed(seed)
        raw = torch.randn(samples, features, generator=generator, dtype=torch.float64)
        left, _, right = torch.linalg.svd(raw, full_matrices=False)
        spectrum = torch.logspace(
            0.0,
            torch.log10(torch.tensor(condition_number, dtype=torch.float64)),
            features,
            dtype=torch.float64,
        ).sqrt()
        design = left @ torch.diag(spectrum) @ right
        truth = torch.randn(features, generator=generator, dtype=torch.float64)
        noise = 0.1 * torch.randn(samples, generator=generator, dtype=torch.float64)
        return cls(design=design, targets=design @ truth + noise)

    @property
    def samples(self) -> int:
        return int(self.design.shape[0])

    @property
    def features(self) -> int:
        return int(self.design.shape[1])

    def hessian(self) -> torch.Tensor:
        return self.design.T @ self.design / self.samples

    def linear_term(self) -> torch.Tensor:
        return self.design.T @ self.targets / self.samples

    def gradient(self, theta: torch.Tensor) -> torch.Tensor:
        return self.hessian() @ theta - self.linear_term()

    def stable_step_size(self) -> float:
        """The gradient-descent stability threshold ``2 / lambda_max``."""

        return float(2.0 / torch.linalg.eigvalsh(self.hessian()).max())


def transport_bound(
    microcosm: LinearRegressionMicrocosm,
    scale: torch.Tensor,
    step_size: float,
    steps: int,
    theta_0: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run both trajectories and compare the observed divergence with the exact bound.

    Everything here is float64 and closed form. The returned ``bound_ratio`` is the quantity Gate C
    thresholds: bound divided by what actually happened.
    """

    hessian = microcosm.hessian()
    linear_term = microcosm.linear_term()
    inverse_scale = 1.0 / scale
    theta = torch.zeros(microcosm.features, dtype=torch.float64) if theta_0 is None else theta_0

    # Exact Lipschitz constants of the two affine update maps.
    source_operator = torch.eye(microcosm.features, dtype=torch.float64) - step_size * hessian
    target_hessian = torch.diag(inverse_scale) @ hessian @ torch.diag(inverse_scale)
    target_operator = (
        torch.eye(microcosm.features, dtype=torch.float64) - step_size * target_hessian
    )
    source_lipschitz = float(torch.linalg.matrix_norm(source_operator, ord=2))
    target_lipschitz = float(torch.linalg.matrix_norm(target_operator, ord=2))

    # The same bound, with the Lipschitz constant obtained the way a practitioner would if they
    # applied the triangle inequality instead of computing the spectral norm of the whole operator:
    # ||I - eta H'|| <= ||I|| + eta ||H'||. This discards the cancellation that makes gradient
    # descent contractive at all, and section 6.10 predicts it should be far looser.
    naive_lipschitz = 1.0 + step_size * float(torch.linalg.matrix_norm(target_hessian, ord=2))

    source = theta.clone()
    target = scale * theta.clone()  # mapped initialization, so e_0 = 0 exactly

    accumulated_bound = 0.0
    naive_bound = 0.0
    observed = 0.0
    defects: list[float] = []
    for _ in range(steps):
        gradient = hessian @ source - linear_term
        # delta_k = eta * ||(S^-1 - S) (H theta - b)||
        defect = float(step_size * torch.linalg.vector_norm((inverse_scale - scale) * gradient))
        defects.append(defect)

        source = source - step_size * gradient
        # grad' (theta') = S^-1 (H S^-1 theta' - b). Applying S^-1 to the already double-scaled
        # Hessian would scale it three times and silently break the bound.
        target = target - step_size * (
            inverse_scale * (hessian @ (inverse_scale * target) - linear_term)
        )

        accumulated_bound = target_lipschitz * accumulated_bound + defect
        naive_bound = naive_lipschitz * naive_bound + defect
        observed = float(torch.linalg.vector_norm(target - scale * source))

    # Predictive-space bound passes through the readout's Lipschitz constant.
    readout = microcosm.design @ torch.diag(inverse_scale)
    readout_lipschitz = float(torch.linalg.matrix_norm(readout, ord=2))
    predictive = float(
        torch.linalg.vector_norm(readout @ target - microcosm.design @ source, ord=float("inf"))
    )

    # Below this, `observed` is the difference of two O(1) float64 quantities accumulated over the
    # horizon, so it is rounding noise and any ratio computed from it is meaningless.
    numerical_floor = (
        100.0
        * float(torch.finfo(torch.float64).eps)
        * max(float(torch.linalg.vector_norm(scale * source)), 1.0)
    )
    floor = 1e-300
    return {
        "numerical_floor": numerical_floor,
        "observed_at_numerical_floor": observed <= numerical_floor,
        "source_lipschitz": source_lipschitz,
        "target_lipschitz": target_lipschitz,
        "naive_lipschitz": naive_lipschitz,
        "contractive": target_lipschitz < 1.0,
        "naive_bound_parameter_divergence": naive_bound,
        "naive_bound_ratio": naive_bound / max(observed, 1e-300),
        "mean_defect": sum(defects) / len(defects),
        "max_defect": max(defects),
        "observed_parameter_divergence": observed,
        "bound_parameter_divergence": accumulated_bound,
        "bound_ratio": accumulated_bound / max(observed, floor),
        "observed_predictive_divergence": predictive,
        "bound_predictive_divergence": readout_lipschitz * accumulated_bound,
        "predictive_bound_ratio": (readout_lipschitz * accumulated_bound) / max(predictive, floor),
        "readout_lipschitz": readout_lipschitz,
    }
