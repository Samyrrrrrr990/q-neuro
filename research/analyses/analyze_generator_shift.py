"""Compute paired seed effects for the latest generator-shift experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from qneuro.metrics import aggregate_seed_metrics

ROOT = Path(__file__).resolve().parents[2]


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "experiment_zero_generator_shift":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed generator-shift result found")
    return max(candidates)[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "research"
        / "analyses"
        / "generated"
        / "generator_shift_paired_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    lookup: dict[tuple[str, int, int, str], dict[str, float]] = {}
    seeds_by_size: dict[int, set[int]] = defaultdict(set)
    for run in result["runs"]:
        model = run["model"]
        size = int(run["train_cases"])
        seed = int(run["seed"])
        seeds_by_size[size].add(seed)
        for environment, metrics in run["environment_metrics"].items():
            lookup[(model, size, seed, environment)] = metrics

    comparisons: dict[str, Any] = {}
    baselines = ("transformer", "gru", "real_operator", "two_channel_operator")
    metrics = ("top1", "nll", "ece", "counterfactual_pair_accuracy")
    environments = tuple(result["summary"]["complex_operator"]["250"])
    for size in sorted(seeds_by_size):
        comparisons[str(size)] = {}
        for environment in environments:
            comparisons[str(size)][environment] = {}
            for baseline in baselines:
                comparisons[str(size)][environment][baseline] = {}
                for metric in metrics:
                    differences = [
                        {
                            "difference": lookup[("complex_operator", size, seed, environment)][
                                metric
                            ]
                            - lookup[(baseline, size, seed, environment)][metric]
                        }
                        for seed in sorted(seeds_by_size[size])
                    ]
                    comparisons[str(size)][environment][baseline][metric] = aggregate_seed_metrics(
                        differences
                    )["difference"]

    output = {
        "source_experiment": result_directory.name,
        "definition": "complex operator metric minus baseline metric, paired by training seed",
        "lower_is_better_metrics": ["nll", "ece"],
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
