"""Run the frozen NeuroWorld shortcut/leakage gate before new outcome experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from experiments.run_generator_shift import build_world
from neuroworld.validity import audit_dataset
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[1]


def gate_violations(audit: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if len(audit["case_consistency_errors"]) > int(thresholds["maximum_consistency_errors"]):
        violations.append("case_consistency_errors")
    for key in ("train_validation_duplicate_rate", "train_test_duplicate_rate"):
        if float(audit[key]) > float(thresholds["maximum_duplicate_rate"]):
            violations.append(key)
    if float(audit["maximum_class_prevalence_deviation"]) > float(
        thresholds["maximum_class_prevalence_deviation"]
    ):
        violations.append("maximum_class_prevalence_deviation")
    maximum_shortcuts = thresholds["maximum_shortcut_test_accuracy"]
    for score in audit["shortcuts"]:
        if score["name"] in maximum_shortcuts and float(score["test_accuracy"]) > float(
            maximum_shortcuts[score["name"]]
        ):
            violations.append(f"shortcut:{score['name']}")
    return violations


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_simulator_red_team"]
    source_environment = environment_record(ROOT, command=command)
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    preregistration_document = config.get(
        "preregistration_document", "docs/PREREGISTRATION_NEXT_PHASE.md"
    )
    preregistration_path = ROOT / preregistration_document
    registry.register_preregistration(
        config["preregistration_id"],
        str(config.get("preregistration_version", "2.0.0")),
        preregistration_document,
        file_sha256(preregistration_path),
    )
    registry.register_hypothesis(
        "H6",
        "The observed phenomenon may be an artifact of NeuroWorld.",
        status="open",
    )
    experiment_id, result_directory = registry.reserve(config, ROOT / "experiments" / "results")
    registry.attach_protocol(
        experiment_id,
        config["preregistration_id"],
        config["hypothesis_id"],
        command,
    )
    try:
        config_path = result_directory / "config.yaml"
        environment_path = result_directory / "environment.json"
        metrics_path = result_directory / "metrics.json"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        environment_path.write_text(
            json.dumps(
                source_environment,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        world = build_world(config["world"])
        dataset = config["dataset"]
        train = world.generate(int(dataset["train_cases"]), int(dataset["train_seed"]))
        validation = world.generate(
            int(dataset["validation_cases"]), int(dataset["validation_seed"])
        )
        test = world.generate(int(dataset["test_cases"]), int(dataset["test_seed"]))
        audit = audit_dataset(train, validation, test)
        replay = world.generate(int(dataset["test_cases"]), int(dataset["test_seed"]))
        deterministic_replay = all(
            first.label == second.label
            and first.evidence.tobytes() == second.evidence.tobytes()
            and first.tokens.tobytes() == second.tokens.tobytes()
            for first, second in zip(test, replay, strict=True)
        )
        violations = gate_violations(audit, config["failure_thresholds"])
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "gate_passed": not violations,
            "gate_violations": violations,
            "deterministic_replay": deterministic_replay,
            "audit": audit,
            "scientific_interpretation": (
                "No frozen shortcut threshold failed. This does not prove simulator validity."
                if not violations
                else "QN-GRAND-001 is blocked until every violation is resolved and preregistered."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shortcut_summaries = {
            score["name"]: {
                "validation_accuracy": {"mean": score["validation_accuracy"], "std": 0.0},
                "test_accuracy": {"mean": score["test_accuracy"], "std": 0.0},
            }
            for score in audit["shortcuts"]
        }
        registry.complete(
            experiment_id,
            shortcut_summaries,
            [
                ("configuration", config_path),
                ("environment", environment_path),
                ("metrics", metrics_path),
            ],
        )
        return experiment_id, result_directory, result
    except Exception:
        registry.fail(experiment_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "simulator_red_team.yaml",
    )
    parser.add_argument("--allow-dirty", action="store_true", help="reserved for smoke debugging")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_id, directory, result = run(config, allow_dirty=args.allow_dirty)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "result_directory": str(directory),
                "gate_passed": result["gate_passed"],
                "violations": result["gate_violations"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
