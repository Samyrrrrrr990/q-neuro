"""Compute paired effects for phase, order, evidence-sign, and density-rank ablations."""

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
    "complex_operator_minus_complex_accumulator": ("complex_operator", "complex_accumulator"),
    "complex_operator_minus_two_channel": ("complex_operator", "two_channel_operator"),
    "complex_operator_minus_magnitude_readout": (
        "complex_operator",
        "complex_magnitude_readout",
    ),
    "complex_operator_minus_no_negative_evidence": (
        "complex_operator",
        "complex_no_negative",
    ),
    "adaptive_minus_fixed_attractor": ("adaptive_attractor", "energy_attractor"),
    "hamiltonian_minus_dissipative": ("hamiltonian", "dissipative"),
    "hybrid_minus_hamiltonian": ("hybrid_dynamics", "hamiltonian"),
    "density_rank2_minus_rank1": ("density_rank2", "density_rank1"),
    "density_rank4_minus_rank2": ("density_rank4", "density_rank2"),
}
METRICS = (
    "top1",
    "nll",
    "ece",
    "counterfactual_pair_accuracy",
    "ambiguity_pair_nll",
    "ambiguity_twin_mass",
)


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "critical_ablation_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed critical ablation suite found")
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
        / "critical_ablation_paired_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    run_lookup = {
        (run["model"], int(run["seed"])): run["in_domain_metrics"] for run in result["runs"]
    }
    seeds = sorted({int(run["seed"]) for run in result["runs"]})
    output_contrasts: dict[str, Any] = {}
    for name, (first, second) in CONTRASTS.items():
        in_domain = {
            metric: summarize(
                [
                    float(run_lookup[(first, seed)][metric])
                    - float(run_lookup[(second, seed)][metric])
                    for seed in seeds
                ]
            )
            for metric in METRICS
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
            for metric in ("top1", "nll", "ece", "counterfactual_pair_accuracy")
        }
        output_contrasts[name] = {"in_domain": in_domain, "moderate_shift": shifted}
    output = {
        "source_experiment": result_directory.name,
        "difference_convention": "first named model minus second named model",
        "in_domain_unit": "training seed (n=3)",
        "shift_unit": "unseen world seed after training-seed averaging (n=3)",
        "warning": "Exact two-sided sign-flip p-values are bounded below by 0.25 at n=3.",
        "contrasts": output_contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
