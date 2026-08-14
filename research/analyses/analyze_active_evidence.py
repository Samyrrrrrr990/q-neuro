"""Compute paired architecture and acquisition-policy effects for active evidence."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from qneuro.metrics import aggregate_seed_metrics

ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    "mlp",
    "transformer",
    "gru",
    "real_operator",
    "two_channel_operator",
    "complex_operator",
)
STRATEGIES = ("random", "fixed_information", "expected_information_gain")
METRICS = (
    "accuracy_auc",
    "final_accuracy",
    "final_nll",
    "resolution_rate",
    "mean_queries_to_resolution_penalized",
    "policy_seconds",
)


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "active_evidence_acquisition":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed active-evidence result found")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "research"
        / "analyses"
        / "generated"
        / "active_evidence_paired_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    lookup: dict[tuple[str, int, str], dict[str, float]] = {}
    for run in result["runs"]:
        for strategy, policy in run["policies"].items():
            lookup[(run["model"], int(run["seed"]), strategy)] = policy
    seeds = sorted({int(run["seed"]) for run in result["runs"]})

    policy_effects: dict[str, Any] = {}
    for model in MODELS:
        policy_effects[model] = {}
        for comparator in ("random", "fixed_information"):
            policy_effects[model][f"expected_information_gain_minus_{comparator}"] = {
                metric: summarize(
                    [
                        float(lookup[(model, seed, "expected_information_gain")][metric])
                        - float(lookup[(model, seed, comparator)][metric])
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            }

    architecture_effects: dict[str, Any] = {}
    for comparator in MODELS[:-1]:
        architecture_effects[f"complex_minus_{comparator}"] = {
            strategy: {
                metric: summarize(
                    [
                        float(lookup[("complex_operator", seed, strategy)][metric])
                        - float(lookup[(comparator, seed, strategy)][metric])
                        for seed in seeds
                    ]
                )
                for metric in METRICS
            }
            for strategy in STRATEGIES
        }

    output = {
        "source_experiment": result_directory.name,
        "statistical_unit": "training seed",
        "n": len(seeds),
        "difference_convention": "first named condition minus second, paired by seed",
        "inference_warning": (
            "With n=3, exact two-sided sign-flip p-values cannot be below 0.25; "
            "all effects are exploratory."
        ),
        "policy_effects": policy_effects,
        "architecture_effects": architecture_effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
