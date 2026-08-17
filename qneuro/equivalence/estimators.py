"""Candidate defect estimators, the baselines they must beat, and leave-one-family-out scoring.

Gate D of `docs/ML2_PREREGISTRATION_001.md` requires a candidate to beat **every** baseline in
section 6.4 on at least two discovery families. The hard baseline is raw one-step predictive
divergence: if the elaborate accumulated quantity reduces to it, that is a negative result and
section 8 requires it to be reported as one.

Everything is fitted in log space, because the quantities span many orders of magnitude and the
target of interest is the magnitude of the final gap, not its raw value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

#: Candidate estimators under test.
#:
#: ``one_step_defect`` is deliberately **not** here. With a mapped initialization ``e_0 = 0``, so the
#: first re-coupled step's defect is bit-for-bit the first step's predictive divergence — it was
#: identical in all 216 rows of QE-000009. Listing it as a candidate would have smuggled a baseline
#: into the candidate set and let it "tie" its way past the gate.
CANDIDATES = (
    "cumulative_defect",
    "amplified_defect",
)

#: Section 6.4 baselines. Every one must be beaten for a candidate to count on a family.
BASELINES = (
    "one_step_predictive_divergence",
    "total_gradient_norm",
    "loss_decrease",
    "learning_rate",
    "parameter_count",
    "mean_amplification",
)

#: Floor for log transforms: below this a measured divergence is float32 rounding, not signal.
FLOOR = 1e-9


def _log(value: float) -> float:
    return math.log10(max(abs(float(value)), FLOOR))


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """Ordinary least squares in log space. Returns (intercept, slope)."""

    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance <= 0.0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / variance
    return mean_y - slope * mean_x, slope


def _score(predictions: Sequence[float], truths: Sequence[float]) -> dict[str, float]:
    n = len(truths)
    if n == 0:
        return {"r2": float("nan"), "mae": float("nan"), "sign_accuracy": float("nan")}
    mean_truth = sum(truths) / n
    total = sum((y - mean_truth) ** 2 for y in truths)
    residual = sum((p - y) ** 2 for p, y in zip(predictions, truths, strict=True))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    mae = sum(abs(p - y) for p, y in zip(predictions, truths, strict=True)) / n
    return {"r2": r2, "mae": mae, "n": float(n)}


def evaluate_leave_one_family_out(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    target: str = "final_divergence",
) -> dict[str, Any]:
    """Fit each feature on all-but-one family and score it on the held-out family.

    Univariate and log-log by design. A richer model would fit the discovery cells better and tell
    us less about whether the quantity generalizes, which is the only question Gate D asks.
    """

    families = sorted({row["family"] for row in rows})
    results: dict[str, dict[str, Any]] = {}

    for feature in features:
        per_family: dict[str, dict[str, float]] = {}
        for held_out in families:
            train = [row for row in rows if row["family"] != held_out]
            test = [row for row in rows if row["family"] == held_out]
            if not train or not test:
                continue
            intercept, slope = _fit_line(
                [_log(row[feature]) for row in train], [_log(row[target]) for row in train]
            )
            predictions = [intercept + slope * _log(row[feature]) for row in test]
            truths = [_log(row[target]) for row in test]
            per_family[held_out] = {**_score(predictions, truths), "slope": slope}
        # Within-family fits separate "there is no relationship" from "the relationship does not
        # transfer". Those are different findings and must not be collapsed into one number.
        within: dict[str, dict[str, float]] = {}
        for family in families:
            subset = [row for row in rows if row["family"] == family]
            if len(subset) < 3:
                continue
            xs = [_log(row[feature]) for row in subset]
            ys = [_log(row[target]) for row in subset]
            intercept, slope = _fit_line(xs, ys)
            within[family] = {
                **_score([intercept + slope * x for x in xs], ys),
                "slope": slope,
                "intercept": intercept,
            }

        finite = [v["r2"] for v in per_family.values() if not math.isnan(v["r2"])]
        within_finite = [v["r2"] for v in within.values() if not math.isnan(v["r2"])]
        results[feature] = {
            "per_family": per_family,
            "within_family": within,
            "mean_r2": sum(finite) / len(finite) if finite else float("nan"),
            "worst_r2": min(finite) if finite else float("nan"),
            "mean_within_family_r2": (
                sum(within_finite) / len(within_finite) if within_finite else float("nan")
            ),
            "families_with_positive_r2": sum(1 for value in finite if value > 0.0),
        }

    # How far apart the families sit on the target scale. When the ranges do not overlap, a single
    # global line cannot span them and leave-one-family-out is hopeless regardless of the feature.
    spread: dict[str, dict[str, float]] = {}
    for family in families:
        values = sorted(_log(row[target]) for row in rows if row["family"] == family)
        spread[family] = {
            "min": values[0],
            "median": values[len(values) // 2],
            "max": values[-1],
            "n": float(len(values)),
        }

    return {"families": families, "features": results, "target_spread_log10": spread}


def gate_d_verdict(evaluation: dict[str, Any]) -> dict[str, Any]:
    """A candidate passes on a family when it beats every baseline's held-out R-squared there."""

    features = evaluation["features"]
    families = evaluation["families"]
    verdicts: dict[str, Any] = {}

    for candidate in CANDIDATES:
        if candidate not in features:
            continue
        wins: list[str] = []
        for family in families:
            candidate_entry = features[candidate]["per_family"].get(family)
            if candidate_entry is None or math.isnan(candidate_entry["r2"]):
                continue
            beaten = True
            for baseline in BASELINES:
                baseline_entry = features.get(baseline, {}).get("per_family", {}).get(family)
                if baseline_entry is None or math.isnan(baseline_entry["r2"]):
                    continue
                if candidate_entry["r2"] <= baseline_entry["r2"]:
                    beaten = False
                    break
            if beaten and candidate_entry["r2"] > 0.0:
                wins.append(family)
        verdicts[candidate] = {
            "families_beating_all_baselines": wins,
            "count": len(wins),
            "passes_gate_d": len(wins) >= 2,
        }

    # The same comparison run within family. This does not substitute for Gate D — the gate is
    # explicitly out-of-family — but it identifies whether a failure is "no signal" or "no
    # transferable calibration", which decides which kill condition in section 8 applies.
    within_verdicts: dict[str, Any] = {}
    for candidate in CANDIDATES:
        if candidate not in features:
            continue
        wins = []
        for family in families:
            entry = features[candidate].get("within_family", {}).get(family)
            if entry is None or math.isnan(entry["r2"]):
                continue
            beaten = all(
                entry["r2"]
                > features.get(baseline, {})
                .get("within_family", {})
                .get(family, {"r2": -1e18})["r2"]
                for baseline in BASELINES
            )
            if beaten and entry["r2"] > 0.0:
                wins.append(family)
        within_verdicts[candidate] = {"families_beating_all_baselines": wins, "count": len(wins)}

    passing = [name for name, entry in verdicts.items() if entry["passes_gate_d"]]
    return {
        "per_candidate": verdicts,
        "within_family_per_candidate": within_verdicts,
        "any_candidate_passes": bool(passing),
        "passing_candidates": passing,
        "requirement": "beat every section 6.4 baseline on at least two discovery families",
        "note": (
            "Gate D is an out-of-family test. Within-family results are reported alongside it to "
            "diagnose the failure mode and may not be substituted for the gate."
        ),
    }
