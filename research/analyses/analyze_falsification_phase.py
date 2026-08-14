"""Synthesize the next-phase falsification evidence at family/world/seed level."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.statistics import (
    HierarchicalObservation,
    hierarchical_bootstrap,
    paired_sign_flip_pvalue,
    paired_summary,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research" / "analyses" / "generated" / "falsification_phase.json"


def load_result(experiment_id: str, filename: str = "metrics.json") -> dict[str, Any]:
    return json.loads(
        (ROOT / "experiments" / "results" / experiment_id / filename).read_text(encoding="utf-8")
    )


def _confirmation_world_seed_effects(
    effects: list[dict[str, Any]],
) -> tuple[list[HierarchicalObservation], list[float]]:
    by_seed: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for effect in effects:
        key = (effect["family"], int(effect["world_seed"]), int(effect["training_seed"]))
        by_seed[key].append(float(effect["difference"]))
    observations = [
        HierarchicalObservation(family, str(world), seed, float(np.mean(values)))
        for (family, world, seed), values in sorted(by_seed.items())
    ]
    by_world: dict[tuple[str, int], list[float]] = defaultdict(list)
    for observation in observations:
        by_world[(observation.generator_family, int(observation.world))].append(observation.value)
    world_effects = [float(np.mean(values)) for values in by_world.values()]
    return observations, world_effects


def _exact_real_differences(confirmation: dict[str, Any]) -> dict[str, float]:
    index = {
        (
            row["family"],
            row["train_size"],
            row["training_seed"],
            row["world_seed"],
            row["severity"],
            row["model"],
        ): row
        for row in confirmation["records"]
    }
    output: dict[str, float] = {}
    for metric in ("top1", "nll", "ece"):
        differences = []
        for key, row in index.items():
            if key[-1] != "complex_operator":
                continue
            exact = index[(*key[:-1], "exact_real_block_operator")]
            differences.append(abs(float(row["metrics"][metric]) - float(exact["metrics"][metric])))
        output[f"maximum_absolute_{metric}_difference"] = max(differences)
    return output


def main() -> None:
    historical = load_result("QN-000008")
    pilot = load_result("QN-000031")
    mechanism = load_result("QN-000033")
    discovery = load_result("QN-000040")
    confirmation = load_result("QN-000042")
    grand = load_result("QN-GRAND-001", "decision.json")
    preflight = load_result("QN-GRAND-001", "preflight.json")

    observations, world_effects = _confirmation_world_seed_effects(confirmation["paired_effects"])
    bootstrap = hierarchical_bootstrap(observations, resamples=20_000, seed=20260814)
    sign_flip = paired_sign_flip_pvalue(world_effects, permutations=200_000, seed=20260814)
    nested_effects = np.asarray(
        [float(item["difference"]) for item in confirmation["paired_effects"]]
    )
    historical_moderate = historical["paired_world_effects"]["moderate"]["two_channel_operator"][
        "top1"
    ]

    result = {
        "scope": "synthetic and nonclinical computational evidence",
        "outcome_category": "A_falsified_intrinsic_complex_advantage",
        "comparator_warning": (
            "QN-000008 compares complex with two-channel real; later studies compare complex with "
            "a cellwise best-real envelope containing an exact real implementation."
        ),
        "historical_within_neuroworld": {
            "experiment_id": "QN-000008",
            "comparator": "two_channel_operator",
            "moderate_shift_mean_difference": historical_moderate["mean"],
            "ci_low": historical_moderate["ci95_low"],
            "ci_high": historical_moderate["ci95_high"],
        },
        "power_pilot": {
            "experiment_id": "QN-000031",
            "train_size_1000": pilot["power_plan"]["world_effects_by_training_size"]["1000"],
            "selected_worlds": pilot["power_plan"]["selection"]["selected_worlds"],
            "estimated_power": pilot["power_plan"]["selection"]["estimated_power"],
            "outcome_eligible": pilot["outcome_eligible"],
        },
        "mechanism_discovery": {
            "experiment_id": "QN-000033",
            "outcome_eligible": mechanism["outcome_eligible"],
            "interpretation": (
                "Exact real-block computation reproduces complex computation; destructive phase "
                "interventions show phase use but not uniquely complex capacity."
            ),
        },
        "reduced_discovery": {
            "experiment_id": "QN-000040",
            "nested_effects": len(discovery["paired_effects"]),
            "positive_nested_effects": sum(
                float(item["difference"]) > 0.0 for item in discovery["paired_effects"]
            ),
            "mean_nested_effect": float(
                np.mean([float(item["difference"]) for item in discovery["paired_effects"]])
            ),
            "frozen_candidate": "quadratic",
            "candidate_discovery_r2": discovery["candidate_laws"]["quadratic"]["discovery_r2"],
            "candidate_discovery_mae": discovery["candidate_laws"]["quadratic"]["discovery_mae"],
            "outcome_eligible": discovery["outcome_eligible"],
        },
        "heldout_confirmation": {
            "experiment_id": "QN-000042",
            "families": sorted({item["family"] for item in confirmation["paired_effects"]}),
            "worlds": len(
                {(item["family"], item["world_seed"]) for item in confirmation["paired_effects"]}
            ),
            "training_seeds": len(
                {item["training_seed"] for item in confirmation["paired_effects"]}
            ),
            "nested_effects": len(nested_effects),
            "positive_nested_effects": int(np.count_nonzero(nested_effects > 0.0)),
            "zero_nested_effects": int(np.count_nonzero(nested_effects == 0.0)),
            "nested_summary": paired_summary(nested_effects),
            "world_summary": paired_summary(world_effects),
            "hierarchical_bootstrap": bootstrap,
            "world_sign_flip_pvalue_two_sided": sign_flip,
            "best_real_winner_counts": dict(
                Counter(item["best_real_model"] for item in confirmation["paired_effects"])
            ),
            "exact_real_equivalence": _exact_real_differences(confirmation),
            "law_confirmation": confirmation["law_confirmation"],
            "architecture_effect": confirmation["architecture_effect"],
            "outcome_eligible": confirmation["outcome_eligible"],
        },
        "qn_grand_001": {
            "status": grand["status"],
            "executed": grand["qn_grand_001_executed"],
            "sealed_benchmark_opened": grand["sealed_benchmark_opened"],
            "blocking_failures": preflight["blocking_failures"],
        },
        "final_interpretation": (
            "The evaluated evidence falsifies an intrinsic complex-arithmetic advantage: an exact "
            "real block is functionally equivalent and stronger real controls remove the observed "
            "robustness gain. The selected quantitative law fails held-out magnitude prediction. "
            "QN-GRAND-001 remains unexecuted because mandatory readiness gates failed."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
