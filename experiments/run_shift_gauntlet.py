"""Compile and audit every preregistered ShiftGauntlet intervention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.run_generator_shift import build_world
from neuroworld import NeuroWorld, ShiftGauntlet, ShiftSpec
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[1]


def intervention_violations(cases: tuple, audit: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    if not cases:
        violations.append("empty_output")
    if len({case.case_id for case in cases}) != len(cases):
        violations.append("duplicate_case_id")
    if any(np.any((case.evidence < -1) | (case.evidence > 1)) for case in cases):
        violations.append("invalid_evidence_value")
    if any(np.any((case.tokens < 0) | (case.tokens >= NeuroWorld.num_tokens)) for case in cases):
        violations.append("invalid_token")
    if any(len(case.tokens) < 1 for case in cases):
        violations.append("empty_sequence")
    if float(audit["spec"]["severity"]) == 0.0 and audit["changed_aligned_cases"] != 0:
        violations.append("zero_severity_not_identity")
    return violations


def build_specs(config: dict[str, Any]) -> list[ShiftSpec]:
    specs: list[ShiftSpec] = []
    severities = [float(value) for value in config["severities"]]
    base_seed = int(config["seed"])
    for family_index, (family, family_config) in enumerate(config["families"].items()):
        modes = family_config.get("modes", [None])
        splits = family_config.get("splits", ["test"])
        for mode_index, mode in enumerate(modes):
            for split_index, split in enumerate(splits):
                for severity_index, severity in enumerate(severities):
                    seed = (
                        base_seed
                        + 10_000 * family_index
                        + 1_000 * mode_index
                        + 100 * split_index
                        + severity_index
                    )
                    specs.append(
                        ShiftSpec(
                            family,  # type: ignore[arg-type]
                            severity,
                            seed,
                            mode=mode,
                            split=split,
                        )
                    )
    return specs


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_shift_gauntlet"]
    source_environment = environment_record(ROOT, command=command)
    preregistration_document = str(config["preregistration_document"])
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    registry.register_preregistration(
        str(config["preregistration_id"]),
        str(config["preregistration_version"]),
        preregistration_document,
        file_sha256(ROOT / preregistration_document),
    )
    registry.register_hypothesis(
        "H6",
        "The observed phenomenon may be an artifact of NeuroWorld.",
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

        world = build_world(config["world"])
        source_cases = world.generate(int(config["cases"]), int(config["case_seed"]))
        gauntlet = ShiftGauntlet()
        compiled: list[dict[str, Any]] = []
        violations: list[str] = []
        for spec in build_specs(config):
            shifted = gauntlet.apply(source_cases, spec)
            spec_violations = intervention_violations(shifted.cases, shifted.audit)
            key = f"{spec.family}:{spec.mode or 'default'}:{spec.split}:{spec.severity:g}"
            violations.extend(f"{key}:{violation}" for violation in spec_violations)
            compiled.append({"key": key, "audit": shifted.audit, "violations": spec_violations})
        configured_families = set(config["families"])
        if configured_families != set(gauntlet.families):
            violations.append("configured_family_set_mismatch")
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "outcome_eligible": False,
            "compiled_interventions": len(compiled),
            "checks_passed": not violations,
            "violations": violations,
            "interventions": compiled,
            "scientific_interpretation": (
                "All declared interventions compiled and passed structural checks; no model "
                "performance claim was tested."
                if not violations
                else "The ShiftGauntlet is invalid until every structural violation is resolved."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry.complete(
            experiment_id,
            {
                "structural_checks": {
                    "checks_passed": {"mean": float(not violations), "std": 0.0},
                    "compiled_interventions": {"mean": float(len(compiled)), "std": 0.0},
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
        default=ROOT / "experiments" / "configs" / "shift_gauntlet.yaml",
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
                "checks_passed": result["checks_passed"],
                "compiled_interventions": result["compiled_interventions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
