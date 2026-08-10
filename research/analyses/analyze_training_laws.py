"""Analyze paired effects, resource scaling, and Pareto structure in the training-law suite."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from qneuro.metrics import aggregate_seed_metrics

ROOT = Path(__file__).resolve().parents[2]


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "training_law_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed full training-law suite found")
    return max(candidates)[1]


def sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(sum(differences) / len(differences))
    null = [
        abs(
            sum(sign * value for sign, value in zip(signs, differences, strict=True))
            / len(differences)
        )
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def summarize(differences: list[float]) -> dict[str, Any]:
    summary = aggregate_seed_metrics([{"difference": value} for value in differences])["difference"]
    summary["differences"] = differences
    summary["exact_two_sided_sign_flip_p"] = sign_flip_pvalue(differences)
    return summary


def pareto_frontier(points: dict[str, tuple[float, float]]) -> list[str]:
    """Return methods not dominated for minimum time and maximum shifted top-1."""

    frontier = []
    for method, (seconds, top1) in points.items():
        dominated = any(
            other_seconds <= seconds
            and other_top1 >= top1
            and (other_seconds < seconds or other_top1 > top1)
            for other, (other_seconds, other_top1) in points.items()
            if other != method
        )
        if not dominated:
            frontier.append(method)
    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "analyses" / "generated" / "training_law_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    methods = list(result["summary"])
    train_sizes = sorted(
        {int(size) for values in result["summary"].values() for size in values}
    )
    largest = str(max(train_sizes))
    reference = "adamw"
    world_effects: dict[str, Any] = {}
    seed_effects: dict[str, Any] = {}
    for method in methods:
        if method == reference:
            continue
        reference_worlds = result["summary"][reference][largest]["shifted"]["by_world"]
        method_worlds = result["summary"][method][largest]["shifted"]["by_world"]
        world_effects[method] = {
            metric: summarize(
                [
                    method_worlds[world][metric]["mean"]
                    - reference_worlds[world][metric]["mean"]
                    for world in reference_worlds
                ]
            )
            for metric in ("top1", "nll", "counterfactual_pair_accuracy")
        }
        method_runs = sorted(
            [
                run
                for run in result["runs"]
                if run["method"] == method and int(run["train_cases"]) == int(largest)
            ],
            key=lambda value: value["seed"],
        )
        reference_runs = sorted(
            [
                run
                for run in result["runs"]
                if run["method"] == reference and int(run["train_cases"]) == int(largest)
            ],
            key=lambda value: value["seed"],
        )
        seed_effects[method] = {
            metric: summarize(
                [
                    candidate["in_domain_metrics"][metric]
                    - baseline["in_domain_metrics"][metric]
                    for candidate, baseline in zip(method_runs, reference_runs, strict=True)
                ]
            )
            for metric in ("top1", "ambiguity_pair_nll", "training_seconds")
        }
    resource_scaling = {
        method: {
            "time_ratio_largest_to_smallest": (
                result["summary"][method][str(max(train_sizes))]["in_domain"][
                    "training_seconds"
                ]["mean"]
                / result["summary"][method][str(min(train_sizes))]["in_domain"][
                    "training_seconds"
                ]["mean"]
            ),
            "backward_passes_largest": result["summary"][method][largest]["in_domain"][
                "backward_passes"
            ]["mean"],
            "autograd_gradient_calls_largest": result["summary"][method][largest]["in_domain"][
                "autograd_gradient_calls"
            ]["mean"],
        }
        for method in methods
    }
    points = {
        method: (
            result["summary"][method][largest]["in_domain"]["training_seconds"]["mean"],
            result["summary"][method][largest]["shifted"]["across_worlds"]["top1"]["mean"],
        )
        for method in methods
    }
    phase_runs = [
        run
        for run in result["runs"]
        if run["method"] == "phase_gradient" and str(run["train_cases"]) == largest
    ]
    phase_diagnostics = {
        key: float(
            sum(run["resources"]["gradient_diagnostics"][key] for run in phase_runs)
            / len(phase_runs)
        )
        for key in phase_runs[0]["resources"]["gradient_diagnostics"]
    }
    output = {
        "source_experiment": result_directory.name,
        "architecture": result["architecture"],
        "largest_train_size": int(largest),
        "shift_effect_unit": "unseen world seed (n=3), after averaging training seeds",
        "in_domain_effect_unit": "training seed (n=3)",
        "method_minus_adamw_at_largest_size": {
            "shifted_world_effects": world_effects,
            "in_domain_seed_effects": seed_effects,
        },
        "resource_scaling": resource_scaling,
        "time_shift_pareto_frontier": pareto_frontier(points),
        "time_shift_points": {
            method: {"training_seconds": value[0], "shifted_top1": value[1]}
            for method, value in points.items()
        },
        "phase_gradient_diagnostics": phase_diagnostics,
        "warnings": [
            "CPU process RSS deltas are noisy and often quantized at zero; do not infer a memory win.",
            "Only one architecture and one simulator family are tested.",
            "Auxiliary-label methods receive mechanism/localization supervision unavailable to diagnosis-only controls.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
