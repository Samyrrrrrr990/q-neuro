"""Auditable measurements and candidate models for computational-law discovery."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class FrozenLaw:
    """A candidate law frozen after discovery and before confirmation is opened."""

    family: Literal["linear", "logarithmic", "saturating", "threshold", "interaction", "quadratic"]
    coefficients: tuple[float, ...]
    hyperparameter: float | None
    discovery_r2: float
    discovery_mae: float
    discovery_n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_law_from_dict(value: Mapping[str, Any]) -> FrozenLaw:
    """Parse a serialized law while rejecting unknown families and malformed coefficients."""

    family = str(value.get("family", ""))
    allowed = {
        "linear",
        "logarithmic",
        "saturating",
        "threshold",
        "interaction",
        "quadratic",
    }
    if family not in allowed:
        raise ValueError(f"unknown frozen law family: {family}")
    coefficients = tuple(float(item) for item in value.get("coefficients", ()))
    if not coefficients or not np.isfinite(coefficients).all():
        raise ValueError("frozen law coefficients must be non-empty and finite")
    hyperparameter_value = value.get("hyperparameter")
    hyperparameter = None if hyperparameter_value is None else float(hyperparameter_value)
    if hyperparameter is not None and not math.isfinite(hyperparameter):
        raise ValueError("frozen law hyperparameter must be finite")
    return FrozenLaw(
        family=family,  # type: ignore[arg-type]
        coefficients=coefficients,
        hyperparameter=hyperparameter,
        discovery_r2=float(value["discovery_r2"]),
        discovery_mae=float(value["discovery_mae"]),
        discovery_n=int(value["discovery_n"]),
    )


def _matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operators must be square matrices")
    if not np.isfinite(matrix).all():
        raise ValueError("operators must be finite")
    return matrix


def normalized_commutator(operator_a: np.ndarray, operator_b: np.ndarray) -> float:
    """Return ``||AB-BA||_F / (||A||_F ||B||_F)`` with a zero-safe denominator."""

    first = _matrix(operator_a)
    second = _matrix(operator_b)
    if first.shape != second.shape:
        raise ValueError("operators must have the same shape")
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    return float(np.linalg.norm(first @ second - second @ first) / denominator)


def state_conditioned_commutator(
    operator_a: np.ndarray, operator_b: np.ndarray, states: np.ndarray
) -> float:
    """Measure normalized order divergence on a declared state distribution."""

    first = _matrix(operator_a)
    second = _matrix(operator_b)
    values = np.asarray(states)
    if values.ndim != 2 or values.shape[1] != first.shape[0]:
        raise ValueError("states must have shape [samples, operator_dimension]")
    commuted = values @ (first @ second - second @ first).T
    forward = values @ (first @ second).T
    reverse = values @ (second @ first).T
    numerator = np.linalg.norm(commuted, axis=1)
    denominator = 0.5 * (np.linalg.norm(forward, axis=1) + np.linalg.norm(reverse, axis=1))
    ratios = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return float(ratios.mean())


def order_sensitivity_index(first: np.ndarray, second: np.ndarray) -> float:
    """Mean total-variation distance between predictions for paired evidence orders."""

    forward = np.asarray(first, dtype=np.float64)
    reverse = np.asarray(second, dtype=np.float64)
    if forward.shape != reverse.shape or forward.ndim != 2:
        raise ValueError("probability arrays must have the same [pairs, classes] shape")
    if np.any(forward < 0.0) or np.any(reverse < 0.0):
        raise ValueError("probabilities must be nonnegative")
    if not np.allclose(forward.sum(axis=1), 1.0) or not np.allclose(reverse.sum(axis=1), 1.0):
        raise ValueError("probability rows must sum to one")
    return float(0.5 * np.abs(forward - reverse).sum(axis=1).mean())


def counterfactual_order_divergence(first: np.ndarray, second: np.ndarray) -> float:
    """Mean Jensen-Shannon divergence in nats for paired evidence orders."""

    forward = np.asarray(first, dtype=np.float64)
    reverse = np.asarray(second, dtype=np.float64)
    _ = order_sensitivity_index(forward, reverse)
    midpoint = 0.5 * (forward + reverse)

    def kl(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        contribution = np.zeros_like(left)
        positive = left > 0.0
        contribution[positive] = left[positive] * np.log(
            left[positive] / np.clip(right[positive], 1e-15, None)
        )
        return contribution.sum(axis=1)

    return float((0.5 * kl(forward, midpoint) + 0.5 * kl(reverse, midpoint)).mean())


def discrete_mutual_information(first: Sequence[int], second: Sequence[int]) -> float:
    """Exact plug-in mutual information in nats for two discrete variables."""

    left = np.asarray(first)
    right = np.asarray(second)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape or left.size == 0:
        raise ValueError("variables must be non-empty paired one-dimensional arrays")
    left_values, left_inverse = np.unique(left, return_inverse=True)
    right_values, right_inverse = np.unique(right, return_inverse=True)
    joint = np.zeros((len(left_values), len(right_values)), dtype=np.float64)
    np.add.at(joint, (left_inverse, right_inverse), 1.0)
    joint /= left.size
    left_probability = joint.sum(axis=1, keepdims=True)
    right_probability = joint.sum(axis=0, keepdims=True)
    independent = left_probability @ right_probability
    positive = joint > 0.0
    return float(np.sum(joint[positive] * np.log(joint[positive] / independent[positive])))


def trajectory_geometry(trajectory: np.ndarray) -> dict[str, float]:
    """Return length, turning curvature, displacement, and reversibility proxies."""

    states = np.asarray(trajectory)
    if states.ndim != 3 or states.shape[1] < 2:
        raise ValueError("trajectory must have shape [samples, steps>=2, dimensions]")
    if np.iscomplexobj(states):
        states = np.concatenate([states.real, states.imag], axis=-1)
    velocity = np.diff(states, axis=1)
    step_lengths = np.linalg.norm(velocity, axis=-1)
    length = step_lengths.sum(axis=1)
    displacement = np.linalg.norm(states[:, -1] - states[:, 0], axis=-1)
    if velocity.shape[1] >= 2:
        first = velocity[:, :-1]
        second = velocity[:, 1:]
        denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        cosine = np.divide(
            np.sum(first * second, axis=-1),
            denominator,
            out=np.ones_like(denominator),
            where=denominator > 0,
        )
        curvature = np.arccos(np.clip(cosine, -1.0, 1.0)).sum(axis=1)
    else:
        curvature = np.zeros(states.shape[0])
    reversibility = np.divide(
        displacement,
        length,
        out=np.zeros_like(displacement),
        where=length > 0,
    )
    return {
        "trajectory_length": float(length.mean()),
        "trajectory_curvature": float(curvature.mean()),
        "net_displacement": float(displacement.mean()),
        "displacement_to_length": float(reversibility.mean()),
    }


def analytic_operator_pair(order_dependence: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct a two-dimensional pair spanning commuting to strongly noncommuting regimes."""

    if not 0.0 <= order_dependence <= 1.0:
        raise ValueError("order_dependence must be in [0, 1]")
    theta = 0.5 * math.pi * order_dependence
    rotation = np.asarray(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    anisotropic_scale = np.asarray([[1.6, 0.0], [0.0, 0.625]], dtype=np.float64)
    return rotation, anisotropic_scale


def _design(
    family: str,
    order_dependence: np.ndarray,
    shift_strength: np.ndarray,
    hyperparameter: float | None,
) -> np.ndarray:
    order = np.asarray(order_dependence, dtype=np.float64)
    shift = np.asarray(shift_strength, dtype=np.float64)
    if order.shape != shift.shape or order.ndim != 1:
        raise ValueError("order dependence and shift strength must be paired vectors")
    interaction = order * shift
    if family == "linear":
        columns = [np.ones_like(order), order, shift]
    elif family == "logarithmic":
        columns = [np.ones_like(order), np.log1p(order), shift]
    elif family == "saturating":
        if hyperparameter is None or hyperparameter <= 0.0:
            raise ValueError("saturating law requires a positive scale")
        columns = [np.ones_like(order), order / (hyperparameter + order), shift]
    elif family == "threshold":
        if hyperparameter is None:
            raise ValueError("threshold law requires a threshold")
        columns = [np.ones_like(order), np.maximum(0.0, order - hyperparameter), shift]
    elif family == "interaction":
        columns = [np.ones_like(order), order, shift, interaction]
    elif family == "quadratic":
        columns = [
            np.ones_like(order),
            order,
            shift,
            np.square(order),
            np.square(shift),
            interaction,
        ]
    else:
        raise ValueError(f"unknown law family: {family}")
    return np.column_stack(columns)


def _fit(
    family: str,
    order: np.ndarray,
    shift: np.ndarray,
    advantage: np.ndarray,
    hyperparameter: float | None,
) -> FrozenLaw:
    design = _design(family, order, shift, hyperparameter)
    coefficients = np.linalg.lstsq(design, advantage, rcond=None)[0]
    prediction = design @ coefficients
    residual = advantage - prediction
    denominator = float(np.square(advantage - advantage.mean()).sum())
    r2 = 1.0 - float(np.square(residual).sum()) / denominator if denominator > 0.0 else 0.0
    return FrozenLaw(
        family=family,  # type: ignore[arg-type]
        coefficients=tuple(float(value) for value in coefficients),
        hyperparameter=hyperparameter,
        discovery_r2=r2,
        discovery_mae=float(np.abs(residual).mean()),
        discovery_n=len(advantage),
    )


def fit_candidate_laws(
    order_dependence: Sequence[float],
    shift_strength: Sequence[float],
    advantage: Sequence[float],
) -> dict[str, FrozenLaw]:
    """Fit prespecified candidates on discovery data only; do not pass confirmation data."""

    order = np.asarray(order_dependence, dtype=np.float64)
    shift = np.asarray(shift_strength, dtype=np.float64)
    outcome = np.asarray(advantage, dtype=np.float64)
    if order.shape != shift.shape or order.shape != outcome.shape or order.ndim != 1:
        raise ValueError("law variables must be paired one-dimensional vectors")
    if order.size < 8 or not np.isfinite(np.column_stack([order, shift, outcome])).all():
        raise ValueError("at least eight finite discovery cells are required")
    candidates = {
        family: _fit(family, order, shift, outcome, None)
        for family in ("linear", "logarithmic", "interaction", "quadratic")
    }
    for family, grid in (
        ("saturating", np.geomspace(0.025, 2.0, 41)),
        ("threshold", np.linspace(0.0, 0.9, 46)),
    ):
        fits = [_fit(family, order, shift, outcome, float(value)) for value in grid]
        candidates[family] = min(
            fits,
            key=lambda fit: (fit.discovery_mae, -fit.discovery_r2, fit.hyperparameter or 0.0),
        )
    return candidates


def freeze_best_candidate(candidates: Mapping[str, FrozenLaw]) -> FrozenLaw:
    """Freeze the minimum information-criterion candidate using a complexity penalty."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    complexity = {
        "linear": 0,
        "logarithmic": 1,
        "saturating": 2,
        "threshold": 2,
        "interaction": 3,
        "quadratic": 4,
    }

    def score(fit: FrozenLaw) -> tuple[float, int, str]:
        parameter_count = len(fit.coefficients) + int(fit.hyperparameter is not None)
        bic_proxy = fit.discovery_n * math.log(max(fit.discovery_mae**2, 1e-24)) + (
            parameter_count * math.log(fit.discovery_n)
        )
        return bic_proxy, complexity[fit.family], fit.family

    return min(candidates.values(), key=score)


def evaluate_frozen_law(
    law: FrozenLaw,
    order_dependence: Sequence[float],
    shift_strength: Sequence[float],
    advantage: Sequence[float],
) -> dict[str, float | int]:
    """Evaluate one already-frozen law on untouched confirmation cells."""

    order = np.asarray(order_dependence, dtype=np.float64)
    shift = np.asarray(shift_strength, dtype=np.float64)
    outcome = np.asarray(advantage, dtype=np.float64)
    if order.shape != shift.shape or order.shape != outcome.shape or order.ndim != 1:
        raise ValueError("confirmation variables must be paired one-dimensional vectors")
    if order.size == 0 or not np.isfinite(np.column_stack([order, shift, outcome])).all():
        raise ValueError("confirmation cells must be non-empty and finite")
    design = _design(law.family, order, shift, law.hyperparameter)
    prediction = design @ np.asarray(law.coefficients)
    residual = outcome - prediction
    denominator = float(np.square(outcome - outcome.mean()).sum())
    r2 = 1.0 - float(np.square(residual).sum()) / denominator if denominator > 0.0 else 0.0
    nonzero = (prediction != 0.0) | (outcome != 0.0)
    sign_accuracy = float(np.mean(np.sign(prediction[nonzero]) == np.sign(outcome[nonzero])))
    return {
        "confirmation_n": int(outcome.size),
        "r2": r2,
        "mean_absolute_error": float(np.abs(residual).mean()),
        "effect_sign_accuracy": sign_accuracy,
    }
