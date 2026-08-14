"""Audit computational-law measurements and discovery/confirmation separation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.run_experiment_zero import ROOT
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry
from research.computational_laws import (
    analytic_operator_pair,
    evaluate_frozen_law,
    fit_candidate_laws,
    freeze_best_candidate,
    normalized_commutator,
    state_conditioned_commutator,
)


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_computational_law_suite"]
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
        frozen_law_path = result_directory / "pipeline_frozen_law.json"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        environment_path.write_text(
            json.dumps(source_environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rng = np.random.default_rng(int(config["analytic_state_seed"]))
        states = rng.normal(size=(int(config["analytic_states"]), 2))
        analytic_grid: list[dict[str, float]] = []
        for level_value in config["order_dependence_levels"]:
            level = float(level_value)
            first, second = analytic_operator_pair(level)
            analytic_grid.append(
                {
                    "order_dependence_control": level,
                    "normalized_commutator": normalized_commutator(first, second),
                    "state_conditioned_commutator": state_conditioned_commutator(
                        first, second, states
                    ),
                }
            )

        # This known interaction is a pipeline test, not evidence for a Q-Neuro law.
        discovery_order, discovery_shift = np.meshgrid(
            np.asarray(config["discovery_order_levels"], dtype=float),
            np.asarray(config["discovery_shift_levels"], dtype=float),
        )
        discovery_rng = np.random.default_rng(int(config["discovery_seed"]))
        discovery_advantage = (
            float(config["known_law"]["intercept"])
            + float(config["known_law"]["interaction"]) * discovery_order * discovery_shift
            + discovery_rng.normal(
                0.0, float(config["known_law"]["noise_standard_deviation"]), discovery_order.shape
            )
        )
        candidates = fit_candidate_laws(
            discovery_order.ravel(), discovery_shift.ravel(), discovery_advantage.ravel()
        )
        frozen = freeze_best_candidate(candidates)
        frozen_law_path.write_text(json.dumps(frozen.to_dict(), indent=2, sort_keys=True) + "\n")

        confirmation_order = np.asarray(config["confirmation_order_levels"], dtype=float)
        confirmation_shift = np.asarray(config["confirmation_shift_levels"], dtype=float)
        confirmation_advantage = (
            float(config["known_law"]["intercept"])
            + float(config["known_law"]["interaction"]) * confirmation_order * confirmation_shift
        )
        confirmation = evaluate_frozen_law(
            frozen, confirmation_order, confirmation_shift, confirmation_advantage
        )
        commutators = [row["normalized_commutator"] for row in analytic_grid]
        violations: list[str] = []
        if not np.isclose(commutators[0], 0.0):
            violations.append("commuting_endpoint_nonzero")
        if np.any(np.diff(commutators) < -1e-12):
            violations.append("commutator_grid_not_monotone")
        if frozen.family != str(config["known_law"]["expected_family"]):
            violations.append("known_law_family_not_recovered")
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "outcome_eligible": False,
            "architecture_claim_permitted": False,
            "analytic_grid": analytic_grid,
            "discovery_candidates": {
                name: candidate.to_dict() for name, candidate in candidates.items()
            },
            "pipeline_frozen_law": frozen.to_dict(),
            "pipeline_confirmation": confirmation,
            "checks_passed": not violations,
            "violations": violations,
            "scientific_interpretation": (
                "The measurement and split-separation pipeline recovered a deliberately embedded "
                "interaction law. This is a software validation, not an empirical discovery."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry.complete(
            experiment_id,
            {
                "law_pipeline": {
                    "checks_passed": {"mean": float(not violations), "std": 0.0},
                    "confirmation_r2": {"mean": float(confirmation["r2"]), "std": 0.0},
                    "confirmation_mae": {
                        "mean": float(confirmation["mean_absolute_error"]),
                        "std": 0.0,
                    },
                }
            },
            [
                ("configuration", config_path),
                ("environment", environment_path),
                ("frozen_pipeline_law", frozen_law_path),
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
        default=ROOT / "experiments" / "configs" / "computational_law_suite.yaml",
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
                "pipeline_frozen_family": result["pipeline_frozen_law"]["family"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
