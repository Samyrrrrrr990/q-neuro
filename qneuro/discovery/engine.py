"""Small, auditable primitives for computational-law discovery."""

from __future__ import annotations

from typing import Any


def dominates(
    first: dict[str, Any],
    second: dict[str, Any],
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> bool:
    weakly_better = all(first[key] >= second[key] for key in maximize) and all(
        first[key] <= second[key] for key in minimize
    )
    strictly_better = any(first[key] > second[key] for key in maximize) or any(
        first[key] < second[key] for key in minimize
    )
    return weakly_better and strictly_better


def pareto_frontier(
    records: list[dict[str, Any]],
    *,
    maximize: tuple[str, ...],
    minimize: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Return deterministic non-dominated records for declared objectives."""

    frontier = [
        candidate
        for candidate in records
        if not any(
            dominates(other, candidate, maximize, minimize)
            for other in records
            if other["candidate_id"] != candidate["candidate_id"]
        )
    ]
    return sorted(frontier, key=lambda value: value["candidate_id"])


def detect_surprises(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag preregistered metric patterns; flags are prompts for analysis, not discoveries."""

    surprises: list[dict[str, Any]] = []
    for record in records:
        gap = record["in_domain_top1"] - record["shifted_top1"]
        if record["in_domain_top1"] >= 0.90 and gap >= 0.45:
            surprises.append(
                {
                    "type": "generalization_reversal",
                    "candidate_id": record["candidate_id"],
                    "severity": gap,
                    "message": "High source accuracy coexists with a large unseen-world collapse.",
                }
            )
        if (
            record["in_domain_top1"] >= 0.90
            and record.get("ambiguity_nll") is not None
            and record["ambiguity_nll"] >= 2.2
        ):
            surprises.append(
                {
                    "type": "accuracy_ambiguity_tension",
                    "candidate_id": record["candidate_id"],
                    "severity": record["ambiguity_nll"],
                    "message": "High source accuracy coexists with poor irreducible-ambiguity NLL.",
                }
            )
        if record["counterfactual_pair_accuracy"] <= 0.05 and record["in_domain_top1"] >= 0.65:
            surprises.append(
                {
                    "type": "order_blind_success",
                    "candidate_id": record["candidate_id"],
                    "severity": record["in_domain_top1"],
                    "message": "Aggregate source accuracy hides near-total chronology-pair failure.",
                }
            )
    return sorted(surprises, key=lambda value: (-value["severity"], value["candidate_id"]))
