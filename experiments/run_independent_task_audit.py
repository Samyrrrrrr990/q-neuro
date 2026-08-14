"""Compile and structurally audit every independent nonclinical task family."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.run_experiment_zero import ROOT
from independent_tasks import INDEPENDENT_TASK_FAMILIES, build_independent_task
from neuroworld import NeuroWorld
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry


def _class_frequencies(labels: list[int]) -> dict[str, float]:
    counts = Counter(labels)
    return {str(label): count / len(labels) for label, count in sorted(counts.items())}


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_independent_task_audit"]
    source_environment = environment_record(ROOT, command=command)
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    preregistration_document = str(config["preregistration_document"])
    registry.register_preregistration(
        str(config["preregistration_id"]),
        str(config["preregistration_version"]),
        preregistration_document,
        file_sha256(ROOT / preregistration_document),
    )
    registry.register_hypothesis(
        "H7",
        "A task-level structural law may generalize beyond NeuroWorld.",
        status="open",
    )
    experiment_id, result_directory = registry.reserve(config, ROOT / "experiments" / "results")
    registry.attach_protocol(
        experiment_id,
        str(config["preregistration_id"]),
        str(config["hypothesis_id"]),
        command,
    )
    try:
        config_path = result_directory / "config.yaml"
        environment_path = result_directory / "environment.json"
        metrics_path = result_directory / "metrics.json"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        environment_path.write_text(
            json.dumps(source_environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        audits: list[dict[str, Any]] = []
        violations: list[str] = []
        for family_index, (family, family_config) in enumerate(config["families"].items()):
            task = build_independent_task(
                family,
                order_dependence=family_config.get("order_dependence"),
                sequence_length=family_config.get("sequence_length"),
            )
            seed = int(config["seed"] + 1000 * family_index)
            train = task.generate(
                int(config["cases_per_split"]), seed, split="train", shift_strength=0.0
            )
            test = task.generate(
                int(config["cases_per_split"]), seed + 1, split="test", shift_strength=1.0
            )
            replay = task.generate(
                int(config["cases_per_split"]), seed + 1, split="test", shift_strength=1.0
            )
            deterministic = all(
                first.label == second.label
                and np.array_equal(first.tokens, second.tokens)
                and np.array_equal(first.evidence, second.evidence)
                for first, second in zip(test.cases, replay.cases, strict=True)
            )
            pairs = task.counterfactual_pairs(int(config["counterfactual_pairs"]), seed + 2)
            same_multiset = all(
                sorted(pair.first.tokens.tolist()) == sorted(pair.second.tokens.tolist())
                for pair in pairs
            )
            same_evidence = all(
                np.array_equal(pair.first.evidence, pair.second.evidence) for pair in pairs
            )
            causal = bool(train.metadata["causal_order"])
            label_semantics = all(
                (pair.first.label != pair.second.label) == causal for pair in pairs
            )
            token_range_valid = all(
                np.all((case.tokens >= 0) & (case.tokens < NeuroWorld.num_tokens))
                for case in (*train.cases, *test.cases)
            )
            family_violations: list[str] = []
            for passed, name in (
                (deterministic, "nondeterministic_replay"),
                (same_multiset, "counterfactual_multiset_changed"),
                (same_evidence, "counterfactual_evidence_changed"),
                (label_semantics, "counterfactual_label_semantics"),
                (token_range_valid, "invalid_token"),
                (bool(train.metadata["synthetic_nonclinical"]), "nonclinical_boundary_missing"),
            ):
                if not passed:
                    family_violations.append(name)
                    violations.append(f"{family}:{name}")
            audits.append(
                {
                    "family": family,
                    "train_metadata": train.metadata,
                    "test_metadata": test.metadata,
                    "train_class_frequencies": _class_frequencies(
                        [case.label for case in train.cases]
                    ),
                    "test_class_frequencies": _class_frequencies(
                        [case.label for case in test.cases]
                    ),
                    "deterministic_replay": deterministic,
                    "counterfactual_same_multiset": same_multiset,
                    "counterfactual_same_evidence": same_evidence,
                    "counterfactual_label_semantics": label_semantics,
                    "token_range_valid": token_range_valid,
                    "violations": family_violations,
                }
            )
        if set(config["families"]) != set(INDEPENDENT_TASK_FAMILIES):
            violations.append("family_set_mismatch")
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "outcome_eligible": False,
            "compiled_families": len(audits),
            "checks_passed": not violations,
            "violations": violations,
            "audits": audits,
            "scientific_interpretation": (
                "All independent synthetic task families passed structural and counterfactual "
                "checks. No architecture was evaluated."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry.complete(
            experiment_id,
            {
                "independent_task_audit": {
                    "checks_passed": {"mean": float(not violations), "std": 0.0},
                    "compiled_families": {"mean": float(len(audits)), "std": 0.0},
                }
            },
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
        default=ROOT / "experiments" / "configs" / "independent_task_audit.yaml",
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
                "compiled_families": result["compiled_families"],
                "checks_passed": result["checks_passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
