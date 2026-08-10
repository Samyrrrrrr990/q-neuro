"""Compute paired effects for the replicated orthogonal NeuroWorld task suite."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from qneuro.metrics import aggregate_seed_metrics

ROOT = Path(__file__).resolve().parents[2]
COMPARATORS = (
    "mlp",
    "transformer",
    "gru",
    "real_operator",
    "two_channel_operator",
)
METRICS = {
    "base": (
        "top1",
        "ambiguity_twin_mass",
        "ambiguity_twin_balance",
        "ambiguity_pair_nll",
        "hidden_ood_auroc_msp",
        "hidden_representation_ood_auroc",
        "hidden_representation_silhouette",
    ),
    "composition": (
        "composition_top1",
        "composition_generalization_gap",
    ),
    "unknown_disease": (
        "unknown_id_top1",
        "ood_auroc_msp",
        "ood_auroc_energy",
        "representation_ood_auroc",
    ),
}


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "neuro_task_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed NeuroWorld task-suite result found")
    return max(candidates)[1]


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Two-sided paired randomization p-value over all sign assignments."""

    observed = abs(sum(differences) / len(differences))
    null_values = [
        abs(sum(sign * value for sign, value in zip(signs, differences, strict=True)) / len(values))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        for values in (differences,)
    ]
    return sum(value >= observed - 1e-12 for value in null_values) / len(null_values)


def paired_summary(differences: list[float]) -> dict[str, Any]:
    aggregate = aggregate_seed_metrics([{"difference": value} for value in differences])[
        "difference"
    ]
    values = torch.tensor(differences, dtype=torch.float64)
    standard_deviation = float(values.std(unbiased=True)) if len(differences) > 1 else 0.0
    aggregate.update(
        {
            "cohen_dz": (
                float(values.mean()) / standard_deviation
                if standard_deviation > 0.0
                else math.copysign(float("inf"), float(values.mean()))
                if float(values.mean()) != 0.0
                else 0.0
            ),
            "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
            "differences": differences,
        }
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "research"
        / "analyses"
        / "generated"
        / "neuro_task_suite_paired_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    lookup: dict[tuple[str, int, str], dict[str, float]] = {}
    for run in result["runs"]:
        for context, context_result in run["contexts"].items():
            lookup[(run["model"], int(run["seed"]), context)] = context_result["metrics"]
    seeds = sorted({int(run["seed"]) for run in result["runs"]})

    comparisons: dict[str, Any] = {}
    for baseline in COMPARATORS:
        comparisons[baseline] = {}
        for context, metric_names in METRICS.items():
            comparisons[baseline][context] = {}
            for metric in metric_names:
                differences = [
                    lookup[("complex_operator", seed, context)][metric]
                    - lookup[(baseline, seed, context)][metric]
                    for seed in seeds
                ]
                comparisons[baseline][context][metric] = paired_summary(differences)

    output = {
        "source_experiment": result_directory.name,
        "statistical_unit": "training seed",
        "n": len(seeds),
        "definition": "complex operator metric minus comparator metric, paired by training seed",
        "lower_is_better_metrics": [
            "ambiguity_pair_nll",
            "composition_generalization_gap (absolute value, not signed value)",
        ],
        "inference_warning": (
            "With n=3 paired seeds, the smallest attainable two-sided exact sign-flip p-value "
            "is 0.25; intervals and effect sizes are descriptive and exploratory."
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
