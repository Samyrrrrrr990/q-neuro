"""Compute paired computational-law effects for the dynamics mechanism suite."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from qneuro.metrics import aggregate_seed_metrics

ROOT = Path(__file__).resolve().parents[2]
CONTRASTS = {
    "hamiltonian_minus_real_operator": ("hamiltonian", "real_operator"),
    "hybrid_minus_real_operator": ("hybrid_dynamics", "real_operator"),
    "hybrid_minus_hamiltonian": ("hybrid_dynamics", "hamiltonian"),
    "hybrid_minus_dissipative": ("hybrid_dynamics", "dissipative"),
    "adaptive_minus_fixed_attractor": ("adaptive_attractor", "energy_attractor"),
    "density_minus_real_operator": ("density_dynamics", "real_operator"),
    "complex_minus_hamiltonian": ("complex_operator", "hamiltonian"),
}
IN_DOMAIN_METRICS = (
    "top1",
    "nll",
    "ece",
    "counterfactual_pair_accuracy",
    "ambiguity_pair_nll",
    "ambiguity_twin_mass",
    "shuffle_delta",
    "training_seconds",
)
SHIFT_METRICS = ("top1", "nll", "ece", "counterfactual_pair_accuracy")


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "dynamics_mechanism_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed dynamics suite found")
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
    result = aggregate_seed_metrics([{"difference": value} for value in differences])["difference"]
    result["differences"] = differences
    result["exact_two_sided_sign_flip_p"] = sign_flip_pvalue(differences)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "analyses" / "generated" / "dynamics_suite_paired_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    run_lookup = {
        (run["model"], int(run["seed"])): run["in_domain_metrics"] for run in result["runs"]
    }
    seeds = sorted({int(run["seed"]) for run in result["runs"]})

    contrasts: dict[str, Any] = {}
    for name, (first, second) in CONTRASTS.items():
        in_domain = {
            metric: summarize(
                [
                    float(run_lookup[(first, seed)][metric])
                    - float(run_lookup[(second, seed)][metric])
                    for seed in seeds
                ]
            )
            for metric in IN_DOMAIN_METRICS
        }
        first_worlds = result["summary"][first]["moderate"]["worlds"]
        second_worlds = result["summary"][second]["moderate"]["worlds"]
        shifted = {
            metric: summarize(
                [
                    float(first_worlds[world][metric]["mean"])
                    - float(second_worlds[world][metric]["mean"])
                    for world in sorted(first_worlds, key=int)
                ]
            )
            for metric in SHIFT_METRICS
        }
        contrasts[name] = {"in_domain": in_domain, "moderate_shift": shifted}

    output = {
        "source_experiment": result_directory.name,
        "difference_convention": "first named model minus second named model",
        "in_domain_statistical_unit": "training seed (n=3)",
        "shift_statistical_unit": "unseen world seed after averaging training seeds (n=3)",
        "inference_warning": (
            "All exact two-sided sign-flip p-values are bounded below by 0.25 at n=3; "
            "the mechanism suite is exploratory."
        ),
        "contrasts": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
