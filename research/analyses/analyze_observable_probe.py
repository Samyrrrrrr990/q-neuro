"""Analyze frozen hierarchical probes and their relation to robustness."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from qneuro.metrics import aggregate_seed_metrics

ROOT = Path(__file__).resolve().parents[2]
PROPERTIES = ("mechanism", "localization", "temporality", "context")


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "hierarchical_observable_probe":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed observable probe experiment found")
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
        default=ROOT / "research" / "analyses" / "generated" / "observable_probe_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    source = json.loads(
        (ROOT / "experiments" / "results" / result["source_experiment"] / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    lookup = {
        (run["model"], int(run["seed"]), property_name, probe_type): metrics
        for run in result["runs"]
        for property_name, probes in run["probes"].items()
        for probe_type, metrics in probes.items()
    }
    seeds = sorted({int(run["seed"]) for run in result["runs"]})

    complex_comparisons: dict[str, Any] = {}
    for baseline in ("real_operator", "gru", "state_space", "two_channel_operator"):
        complex_comparisons[baseline] = {
            property_name: summarize(
                [
                    lookup[("complex_operator", seed, property_name, "linear")]["accuracy"]
                    - lookup[(baseline, seed, property_name, "linear")]["accuracy"]
                    for seed in seeds
                ]
            )
            for property_name in PROPERTIES
        }

    hermitian_effects: dict[str, Any] = {}
    for model in ("complex_operator", "hamiltonian", "hybrid_dynamics"):
        hermitian_effects[model] = {
            property_name: {
                metric: summarize(
                    [
                        lookup[(model, seed, property_name, "hermitian")][metric]
                        - lookup[(model, seed, property_name, "linear")][metric]
                        for seed in seeds
                    ]
                )
                for metric in ("accuracy", "nll")
            }
            for property_name in PROPERTIES
        }

    models = list(result["summary"])
    mean_probe = np.asarray(
        [
            np.mean(
                [
                    result["summary"][model][property_name]["linear"]["accuracy"]["mean"]
                    for property_name in PROPERTIES
                ]
            )
            for model in models
        ]
    )
    shift_top1 = np.asarray(
        [source["summary"][model]["moderate"]["across_worlds"]["top1"]["mean"] for model in models]
    )
    correlation = float(np.corrcoef(mean_probe, shift_top1)[0, 1])
    output = {
        "source_experiment": result_directory.name,
        "frozen_model_experiment": result["source_experiment"],
        "unit": "training seed (n=3) for paired probe effects",
        "complex_minus_baseline_linear_accuracy": complex_comparisons,
        "hermitian_minus_linear": hermitian_effects,
        "cross_model_probe_robustness_pearson_r": correlation,
        "cross_model_warning": "Cross-model correlation is descriptive over 18 non-independent architectures.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
