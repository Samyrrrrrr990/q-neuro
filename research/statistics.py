"""Preregistered statistical utilities for world-level Q-Neuro analyses.

Architecture claims must use independently generated worlds as the top-level unit. Functions in
this module are deterministic when supplied a seed and deliberately avoid per-example hypothesis
tests. Exploratory scripts may import these helpers but may not alter the frozen QN-GRAND-001
decision thresholds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HierarchicalObservation:
    """One scalar outcome nested within generator family, world, and training seed."""

    generator_family: str
    world: str
    training_seed: int
    value: float


def _finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must all be finite")
    return array


def trapezoidal_robustness_auc(severities: Sequence[float], values: Sequence[float]) -> float:
    """Integrate a robustness curve after validating a complete normalized severity grid."""

    severity = _finite_array(severities)
    outcome = _finite_array(values)
    if severity.shape != outcome.shape:
        raise ValueError("severities and values must have identical shapes")
    if np.any(np.diff(severity) <= 0.0):
        raise ValueError("severities must be strictly increasing")
    if severity[0] < 0.0 or severity[-1] > 1.0:
        raise ValueError("severities must lie in [0, 1]")
    return float(np.trapezoid(outcome, severity))


def robustness_slope(severities: Sequence[float], values: Sequence[float]) -> float:
    """Return the least-squares change in outcome per unit normalized severity."""

    severity = _finite_array(severities)
    outcome = _finite_array(values)
    if severity.shape != outcome.shape or severity.size < 2:
        raise ValueError("at least two paired severity/outcome values are required")
    centered = severity - severity.mean()
    denominator = float(np.square(centered).sum())
    if denominator == 0.0:
        raise ValueError("severity variance must be positive")
    return float((centered * (outcome - outcome.mean())).sum() / denominator)


def paired_summary(differences: Sequence[float]) -> dict[str, float | int]:
    """Summarize paired top-level differences without treating cases as replications."""

    values = _finite_array(differences)
    mean = float(values.mean())
    median = float(np.median(values))
    standard_deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
    trimmed_count = math.floor(0.10 * values.size)
    ordered = np.sort(values)
    trimmed = ordered[trimmed_count : values.size - trimmed_count] if trimmed_count else ordered
    return {
        "n": int(values.size),
        "mean": mean,
        "median": median,
        "standard_deviation": standard_deviation,
        "trimmed_mean_10pct": float(trimmed.mean()),
        "worst_decile": float(np.quantile(values, 0.10)),
        "probability_of_superiority": float(np.mean(values > 0.0)),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm familywise-error adjusted p-values in the original order."""

    values = _finite_array(p_values)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = float((values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> list[float]:
    """Return monotone Benjamini-Hochberg adjusted p-values in the original order."""

    values = _finite_array(p_values)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 1.0
    for reverse_rank in range(values.size - 1, -1, -1):
        rank = reverse_rank + 1
        candidate = float(values.size * sorted_values[reverse_rank] / rank)
        running = min(running, candidate)
        adjusted_sorted[reverse_rank] = min(1.0, running)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def paired_sign_flip_pvalue(
    differences: Sequence[float], *, permutations: int = 200_000, seed: int = 0
) -> float:
    """Two-sided paired randomization p-value, exact for up to 20 top-level units."""

    values = _finite_array(differences)
    observed = abs(float(values.mean()))
    if values.size <= 20:
        assignments = 1 << values.size
        exceedances = 0
        for mask in range(assignments):
            signs = np.fromiter(
                (1.0 if mask & (1 << index) else -1.0 for index in range(values.size)),
                dtype=np.float64,
                count=values.size,
            )
            exceedances += abs(float(np.mean(signs * values))) >= observed - 1e-15
        return float(exceedances / assignments)
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(permutations, values.size))
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    return float((np.count_nonzero(null >= observed - 1e-15) + 1) / (permutations + 1))


def _hierarchical_mean(
    nested: Mapping[str, Mapping[str, Sequence[float]]],
    family_indices: np.ndarray,
    rng: np.random.Generator,
) -> float:
    family_means: list[float] = []
    family_names = list(nested)
    for family_index in family_indices:
        worlds = nested[family_names[int(family_index)]]
        world_names = list(worlds)
        sampled_world_indices = rng.integers(0, len(world_names), size=len(world_names))
        world_means: list[float] = []
        for world_index in sampled_world_indices:
            seed_values = _finite_array(worlds[world_names[int(world_index)]])
            sampled_seed_indices = rng.integers(0, seed_values.size, size=seed_values.size)
            world_means.append(float(seed_values[sampled_seed_indices].mean()))
        family_means.append(float(np.mean(world_means)))
    return float(np.mean(family_means))


def hierarchical_bootstrap(
    observations: Sequence[HierarchicalObservation],
    *,
    resamples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap generator family, world, then training seed with equal family weighting."""

    if not observations:
        raise ValueError("at least one observation is required")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    nested_lists: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        if not math.isfinite(observation.value):
            raise ValueError("observation values must be finite")
        nested_lists[observation.generator_family][observation.world].append(observation.value)
    nested: dict[str, dict[str, Sequence[float]]] = {
        family: dict(worlds) for family, worlds in nested_lists.items()
    }
    rng = np.random.default_rng(seed)
    family_count = len(nested)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        family_indices = rng.integers(0, family_count, size=family_count)
        samples[index] = _hierarchical_mean(nested, family_indices, rng)
    alpha = 1.0 - confidence
    family_point_means = [
        np.mean([np.mean(values) for values in worlds.values()]) for worlds in nested.values()
    ]
    return {
        "estimate": float(np.mean(family_point_means)),
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "generator_families": int(family_count),
        "worlds": int(sum(len(worlds) for worlds in nested.values())),
        "observations": len(observations),
    }


def simulate_paired_power(
    *,
    effect: float,
    standard_deviation: float,
    worlds: int,
    alpha: float = 0.05,
    simulations: int = 50_000,
    seed: int = 0,
) -> float:
    """Conservative normal-approximation power simulation for pilot world-count selection."""

    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive")
    if worlds < 2 or simulations <= 0:
        raise ValueError("worlds must be at least two and simulations positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    # The preregistered alpha is 0.05. Reject unsupported thresholds rather than silently applying
    # a mismatched critical value in this dependency-light planning utility.
    if not math.isclose(alpha, 0.05):
        raise ValueError("this planning implementation currently supports alpha=0.05 only")
    rng = np.random.default_rng(seed)
    samples = rng.normal(effect, standard_deviation, size=(simulations, worlds))
    standard_errors = samples.std(axis=1, ddof=1) / math.sqrt(worlds)
    statistics = np.divide(
        samples.mean(axis=1),
        standard_errors,
        out=np.zeros(simulations, dtype=np.float64),
        where=standard_errors > 0.0,
    )
    # 1.96 is slightly anti-conservative at the smallest candidate n=32. The full analysis uses
    # bootstrap intervals and sign flips; this simulation is only a declared planning heuristic.
    return float(np.mean(np.abs(statistics) > 1.96))


def select_world_count(
    *,
    standard_deviation: float,
    minimum_effect: float,
    candidates: Sequence[int],
    target_power: float,
    simulations: int = 50_000,
    seed: int = 0,
) -> dict[str, object]:
    """Apply the frozen smallest-candidate power rule and disclose any target shortfall."""

    if not candidates or sorted(set(candidates)) != list(candidates):
        raise ValueError("candidates must be unique and strictly increasing")
    estimates = {
        int(worlds): simulate_paired_power(
            effect=minimum_effect,
            standard_deviation=standard_deviation,
            worlds=int(worlds),
            simulations=simulations,
            seed=seed + int(worlds),
        )
        for worlds in candidates
    }
    selected = next(
        (worlds for worlds in candidates if estimates[int(worlds)] >= target_power),
        candidates[-1],
    )
    return {
        "selected_worlds": int(selected),
        "target_power": float(target_power),
        "estimated_power": float(estimates[int(selected)]),
        "target_reached": bool(estimates[int(selected)] >= target_power),
        "candidate_power": estimates,
    }
