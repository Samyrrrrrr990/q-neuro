"""Analyze realized hard-halting effects against soft and fixed-depth attractor inference."""

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
        if config.get("experiment") != "hard_velocity_halting":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed full hard-halting experiment found")
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
        default=ROOT / "research" / "analyses" / "generated" / "hard_halting_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    comparisons: dict[str, Any] = {}
    for baseline in ("soft", "fixed_final"):
        comparisons[baseline] = {"in_domain": {}, "shifted": {}}
        for metric in ("top1", "nll", "ece"):
            comparisons[baseline]["in_domain"][metric] = summarize(
                [
                    run["in_domain_metrics"]["hard"][metric]
                    - run["in_domain_metrics"][baseline][metric]
                    for run in result["runs"]
                ]
            )
            hard_worlds = result["summary"]["hard"]["shifted"]["by_world"]
            baseline_worlds = result["summary"][baseline]["shifted"]["by_world"]
            comparisons[baseline]["shifted"][metric] = summarize(
                [
                    hard_worlds[world][metric]["mean"] - baseline_worlds[world][metric]["mean"]
                    for world in hard_worlds
                ]
            )
    latency_ratios = {
        baseline: [
            run["in_domain_metrics"]["hard"]["latency_ms_per_case"]
            / run["in_domain_metrics"][baseline]["latency_ms_per_case"]
            for run in result["runs"]
        ]
        for baseline in ("soft", "fixed_final")
    }
    output = {
        "source_experiment": result_directory.name,
        "checkpoint_source": result["source_experiment"],
        "hard_minus_baseline": comparisons,
        "latency_ratio_hard_over_baseline": {
            key: summarize(values) for key, values in latency_ratios.items()
        },
        "selected_thresholds": [run["selected_velocity_threshold"] for run in result["runs"]],
        "validation_constraints_all_satisfied": all(
            run["validation_constraint_satisfied"] for run in result["runs"]
        ),
        "hard_halt_distribution": {
            f"step_{step}": result["summary"]["hard"]["in_domain"][f"halt_fraction_step_{step}"][
                "mean"
            ]
            for step in range(1, 9)
        },
        "interpretation_guardrail": (
            "Every case halts at the minimum of two states. This is realized truncation, but not "
            "case-adaptive computation; a fixed two-state attractor is the simpler equivalent."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
