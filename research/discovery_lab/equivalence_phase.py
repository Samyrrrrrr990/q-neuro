"""DISCOVERY-001: a sharp stability boundary in equivalence breaking.

Lane B. Nothing here is a claim; see `docs/LANE_POLICY.md`.

Observation
-----------
Two models that represent *exactly the same predictor* at initialization, trained on the same data
with the same optimizer and the same hyperparameters, can end up on opposite sides of a stability
boundary purely because of the coordinates they are written in. One converges; the other diverges
without limit.

Mechanism (derived, not fitted)
-------------------------------
For a homogeneous scaling orbit with uniform scale ``s`` and an *untransported* learning rate, the
target's gradient-descent operator is::

    I - eta * S^-1 H S^-1  =  I - (eta / s^2) H

so the target's effective step size is ``eta / s^2``. Gradient descent is stable exactly when the
effective step times the largest curvature is below 2, which gives a dimensionless control
parameter::

    rho = eta * lambda_max(H) / (2 * s^2)

with a predicted transition at ``rho = 1``, independent of conditioning, seed, and problem scale.
The source is stable whenever ``rho * s^2 < 1``, so for ``s < 1`` there is an open window in which
the source converges and its exact equivalent diverges.

The simplest boring explanation
-------------------------------
This *is* textbook gradient-descent stability. Reparameterizing changes the effective Hessian, hence
the effective step size, hence stability. Nothing about the mechanism is new, and this module says
so in its own record. What the boundary supplies to this program is different: an exactly predictable
place where equivalence breaks, and a mechanistic account of why cross-family defect calibration
failed at Gate D — the discovery families straddle a phase boundary, so they are not one population
and no single calibration can span them.

Differential prediction
-----------------------
Adam's update is scale-free in the gradient, so its effective step does not pick up the ``1/s^2``
factor. The boundary should therefore be **absent** for Adam at the same ``rho``. That is a sharp,
falsifiable prediction and it is tested here rather than assumed.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from qneuro.equivalence.analytic import LinearRegressionMicrocosm

DISCOVERY_ID = "DISCOVERY-001"

#: Predicted transition, derived analytically before any sweep was run.
PREDICTED_CRITICAL_RHO = 1.0

#: Instability is a *growth* property of a single trajectory, not a magnitude property of a pair.
#:
#: Two earlier criteria were tried and discarded, and the reasons are kept because they are the
#: interesting part. (1) An absolute threshold on the paired divergence conflated an unstable run
#: with a merely slow one: at large scale the target's effective step is tiny, so after a fixed
#: budget it sits far from the mapped source while being perfectly stable. (2) A growth ratio on the
#: *paired* divergence still mixed the target's stability with the source's own convergence.
#:
#: What the theory actually predicts is whether the **target** is stable. Measuring that directly,
#: as growth of the target's parameter norm over the final third of training, is independent of the
#: control parameter under test and of the source's trajectory.
GROWTH_RATIO_THRESHOLD = 2.0


def control_parameter(step_size: float, largest_eigenvalue: float, scale: float) -> float:
    """rho = eta * lambda_max / (2 s^2)."""

    return step_size * largest_eigenvalue / (2.0 * scale * scale)


def critical_scale(step_size: float, largest_eigenvalue: float) -> float:
    """The scale at which rho = 1."""

    return math.sqrt(step_size * largest_eigenvalue / 2.0)


def _paired_run(
    microcosm: LinearRegressionMicrocosm,
    scale: torch.Tensor,
    step_size: float,
    steps: int,
    optimizer: str,
) -> dict[str, float]:
    """Run source and target from a mapped initialization; report the final divergence."""

    hessian = microcosm.hessian()
    linear_term = microcosm.linear_term()
    inverse_scale = 1.0 / scale
    source = torch.zeros(microcosm.features, dtype=torch.float64)
    target = scale * source.clone()

    # Adam state, kept in float64 alongside the closed-form gradients.
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    moments = {
        "source": [torch.zeros_like(source), torch.zeros_like(source)],
        "target": [torch.zeros_like(target), torch.zeros_like(target)],
    }

    def update(
        parameters: torch.Tensor, gradient: torch.Tensor, key: str, step: int
    ) -> torch.Tensor:
        if optimizer == "sgd":
            return parameters - step_size * gradient
        first, second = moments[key]
        first.mul_(beta1).add_(gradient, alpha=1 - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
        corrected_first = first / (1 - beta1 ** (step + 1))
        corrected_second = second / (1 - beta2 ** (step + 1))
        return parameters - step_size * corrected_first / (corrected_second.sqrt() + epsilon)

    blew_up = False
    target_midpoint = float("nan")
    source_midpoint = float("nan")
    checkpoint = (2 * steps) // 3 - 1
    for step in range(steps):
        source_gradient = hessian @ source - linear_term
        target_gradient = inverse_scale * (hessian @ (inverse_scale * target) - linear_term)
        source = update(source, source_gradient, "source", step)
        target = update(target, target_gradient, "target", step)
        if step == checkpoint:
            target_midpoint = float(torch.linalg.vector_norm(target))
            source_midpoint = float(torch.linalg.vector_norm(source))
        if not torch.isfinite(target).all() or not torch.isfinite(source).all():
            blew_up = True
            break

    if blew_up:
        return {
            "divergence": float("inf"),
            "target_growth_ratio": float("inf"),
            "source_growth_ratio": 0.0,
            "source_stable": 1.0,
        }

    def growth(final: float, midpoint: float) -> float:
        # A trajectory can overflow the float range *before* any individual entry becomes
        # non-finite, because the norm squares its entries. That produced inf/inf = nan, and
        # `nan > threshold` is False, so runaway runs were silently scored as convergent. Any
        # non-finite norm is divergence, and is reported as such.
        if not math.isfinite(final) or not math.isfinite(midpoint):
            return float("inf")
        return final / midpoint if midpoint > 0.0 else 0.0

    target_growth = growth(float(torch.linalg.vector_norm(target)), target_midpoint)
    source_growth = growth(float(torch.linalg.vector_norm(source)), source_midpoint)
    return {
        "divergence": float(torch.linalg.vector_norm(target - scale * source)),
        "target_growth_ratio": target_growth,
        "source_growth_ratio": source_growth,
        "source_stable": float(source_growth <= GROWTH_RATIO_THRESHOLD),
    }


def sweep(
    condition_numbers: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    step_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    optimizers: tuple[str, ...] = ("sgd", "adamw"),
    scale_points: int = 41,
    steps: int = 300,
) -> list[dict[str, Any]]:
    """Sweep the scale across the predicted boundary at many conditionings and step sizes."""

    records: list[dict[str, Any]] = []
    for seed in seeds:
        for condition_number in condition_numbers:
            microcosm = LinearRegressionMicrocosm.build(300, 10, condition_number, seed=seed)
            largest = float(torch.linalg.eigvalsh(microcosm.hessian()).max())
            stable_step = 2.0 / largest
            for fraction in step_fractions:
                step_size = fraction * stable_step
                predicted = critical_scale(step_size, largest)
                # Sweep symmetrically in log-space around the prediction so the grid cannot be
                # accused of being centred on the observed answer.
                for index in range(scale_points):
                    ratio = 10 ** (-0.6 + 1.2 * index / (scale_points - 1))
                    scale_value = predicted * ratio
                    scaling = torch.full((10,), scale_value, dtype=torch.float64)
                    for optimizer in optimizers:
                        outcome = _paired_run(microcosm, scaling, step_size, steps, optimizer)
                        rho = control_parameter(step_size, largest, scale_value)
                        records.append(
                            {
                                "seed": seed,
                                "condition_number": condition_number,
                                "step_fraction": fraction,
                                "optimizer": optimizer,
                                "scale": scale_value,
                                "scale_over_predicted_critical": ratio,
                                "rho": rho,
                                "predicted_critical_scale": predicted,
                                "divergence": outcome["divergence"],
                                "target_growth_ratio": outcome["target_growth_ratio"],
                                "source_stable": bool(outcome["source_stable"]),
                                "diverged": bool(
                                    not math.isfinite(outcome["target_growth_ratio"])
                                    or outcome["target_growth_ratio"] > GROWTH_RATIO_THRESHOLD
                                ),
                                "predicted_diverged": bool(rho > PREDICTED_CRITICAL_RHO),
                            }
                        )
    return records


def score(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How well does rho = 1 predict which side of the boundary a run lands on?"""

    summary: dict[str, Any] = {}
    for optimizer in sorted({record["optimizer"] for record in records}):
        subset = [record for record in records if record["optimizer"] == optimizer]
        correct = sum(record["diverged"] == record["predicted_diverged"] for record in subset)
        diverged = [record for record in subset if record["diverged"]]
        converged = [record for record in subset if not record["diverged"]]
        # The phenomenon of interest: the source trains fine and its exact equivalent does not.
        broken = [record for record in subset if record["diverged"] and record["source_stable"]]
        summary[optimizer] = {
            "cells": len(subset),
            "prediction_accuracy": correct / len(subset) if subset else float("nan"),
            "diverged_cells": len(diverged),
            "equivalence_broken_cells": len(broken),
            "largest_rho_that_converged": (
                max(record["rho"] for record in converged) if converged else float("nan")
            ),
            "smallest_rho_that_diverged": (
                min(record["rho"] for record in diverged) if diverged else float("nan")
            ),
        }
    return summary
