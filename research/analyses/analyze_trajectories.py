"""Summarize signed-evidence trajectory effects without treating tokens as independent units."""

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
        if config.get("experiment") == "complex_state_trajectory_study":
            metrics = config_path.parent / "metrics.json"
            if metrics.exists():
                candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed trajectory study found")
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
        default=ROOT / "research" / "analyses" / "generated" / "trajectory_effects.json",
    )
    args = parser.parse_args()
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    signed_effect = [
        run["metrics"]["mean_positive_token_delta_true_probability"]
        - run["metrics"]["mean_negative_token_delta_true_probability"]
        for run in result["runs"]
    ]
    output = {
        "source_experiment": result_directory.name,
        "checkpoint_source": result["source_experiment"],
        "unit": "training seed (n=3); token changes are averaged within seed before comparison",
        "positive_minus_negative_token_delta_true_probability": summarize(signed_effect),
        "summary": result["summary"],
        "guardrails": [
            "Observed-negative tokens are not guaranteed to contradict the true diagnosis; they may exclude alternatives.",
            "Revival means later true-label probability returns to its pre-drop level, not clinical belief revision.",
            "The visual case and pair use deterministic generation order, not performance-based selection.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
