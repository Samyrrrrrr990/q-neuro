"""Preflight QN-GRAND-001 and preserve a block without opening sealed data if any gate fails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from experiments.run_experiment_zero import ROOT
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(check_id: str, passed: bool, evidence: str, *, blocking: bool = True) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": bool(passed),
        "blocking": blocking,
        "evidence": evidence,
    }


def _exact_real_difference(confirmation: dict[str, Any], metric: str) -> float:
    index = {
        (
            record["family"],
            record["train_size"],
            record["training_seed"],
            record["world_seed"],
            record["severity"],
            record["model"],
        ): record
        for record in confirmation["records"]
    }
    differences: list[float] = []
    for key, complex_record in index.items():
        if key[-1] != "complex_operator":
            continue
        exact_record = index[(*key[:-1], "exact_real_block_operator")]
        differences.append(
            abs(float(complex_record["metrics"][metric]) - float(exact_record["metrics"][metric]))
        )
    return max(differences)


def build_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every frozen advancement gate using only already-open evidence."""

    evidence_paths = {name: ROOT / value for name, value in config["evidence"].items()}
    evidence = {name: _read_json(path) for name, path in evidence_paths.items()}
    power = evidence["power_pilot"]["power_plan"]["selection"]
    discovery = evidence["reduced_discovery"]
    confirmation = evidence["heldout_confirmation"]
    frozen = _read_json(ROOT / config["frozen_law_artifact"])
    grand_protocol = yaml.safe_load((ROOT / config["machine_protocol"]).read_text())

    model_metadata = {
        run["model"]: run["selected_trial"]["model_metadata"]
        for run in confirmation["training_runs"]
    }
    complex_parameters = int(model_metadata["complex_operator"]["parameter_count"])
    evaluated_parameter_match = all(
        abs(int(metadata["parameter_count"]) - complex_parameters) / complex_parameters
        <= float(config["matching"]["parameter_relative_tolerance"])
        for metadata in model_metadata.values()
    )
    tuning_counts = {
        model: max(
            len(candidate["tuning_trials"])
            for candidate in confirmation["training_runs"]
            if candidate["model"] == model
        )
        for model in model_metadata
    }
    required_real_models = set(grand_protocol["architectures"]["real_envelope"])
    evaluated_real_models = set(model_metadata) - {"complex_operator"}
    raw_prediction_files = list((ROOT / "experiments" / "results").glob("**/predictions.*"))
    exact_top1 = _exact_real_difference(confirmation, "top1")
    exact_nll = _exact_real_difference(confirmation, "nll")

    checks = [
        _check(
            "exact_real_equivalence",
            exact_top1 <= 1e-7 and exact_nll <= 1e-5,
            f"QN-000042 max top1 difference={exact_top1:.3g}; max NLL difference={exact_nll:.3g}",
        ),
        _check(
            "evaluated_parameter_matching",
            evaluated_parameter_match,
            f"Evaluated model parameter counts: {model_metadata}",
        ),
        _check(
            "simulator_red_team",
            evidence["red_team"].get("gate_passed") is True,
            "QN-000028 frozen shortcut gate",
        ),
        _check(
            "shift_gauntlet_structure",
            evidence["shift_gauntlet"].get("checks_passed") is True,
            "QN-000029 structural ShiftGauntlet audit",
        ),
        _check(
            "power_target",
            bool(power["target_reached"])
            and int(power["selected_worlds"]) >= int(config["minimum_required_worlds"])
            and float(power["estimated_power"]) >= float(config["required_power"]),
            f"QN-000031 selected {power['selected_worlds']} worlds at estimated power "
            f"{power['estimated_power']}",
        ),
        _check(
            "law_frozen_before_heldout_confirmation",
            frozen.get("status") == "frozen_provisional"
            and frozen["source"]["experiment_id"] == discovery["experiment_id"]
            and confirmation["frozen_law_id"] == frozen["candidate_id"],
            "QN-LAW-001 source QN-000040; held-out evaluation QN-000042",
        ),
        _check(
            "untouched_nonmedical_family",
            set(frozen["confirmation"]["families"]).isdisjoint(frozen["source"]["families"]),
            f"Held-out families: {frozen['confirmation']['families']}",
        ),
        _check(
            "clean_confirmatory_worktree",
            _read_json(evidence_paths["heldout_confirmation"].parent / "environment.json").get(
                "git_dirty"
            )
            is False,
            "QN-000042 environment.json",
        ),
        _check(
            "complete_preregistered_real_envelope",
            required_real_models.issubset(evaluated_real_models)
            and len(evaluated_real_models) >= int(config["required_real_envelope_size"]),
            f"Evaluated {len(evaluated_real_models)} of {len(required_real_models)} required real "
            f"controls; missing={sorted(required_real_models - evaluated_real_models)}",
        ),
        _check(
            "equal_or_real_favoring_search_budgets",
            tuning_counts.get("complex_operator", 0)
            >= int(config["required_search_trials"]["complex"])
            and all(
                tuning_counts.get(model, 0) >= int(config["required_search_trials"]["each_real"])
                for model in required_real_models
            ),
            f"Observed maximum tuning trials per evaluated model: {tuning_counts}",
        ),
        _check(
            "compute_matching_records",
            False,
            "Per-trial FLOPs and optimizer-step records required by QNF-PREREG-002 are absent.",
        ),
        _check(
            "full_shiftgauntlet_outcome_grid",
            False,
            "QN-000029 is structural and QN-000031 is a reduced pilot; no full confirmatory grid.",
        ),
        _check(
            "full_discovery_protocol",
            not discovery["protocol_deviations"] and frozen["scope"]["qn_grand_001_permitted"],
            f"QN-000040 deviations={discovery['protocol_deviations']}; frozen scope forbids grand use.",
        ),
        _check(
            "raw_predictions_preserved",
            bool(raw_prediction_files),
            f"Prediction artifacts found: {[path.as_posix() for path in raw_prediction_files]}",
        ),
    ]
    failures = [item["check_id"] for item in checks if item["blocking"] and not item["passed"]]
    return {
        "experiment_id": config["experiment_id"],
        "status": "passed" if not failures else "blocked",
        "sealed_benchmark_opened": False,
        "checks": checks,
        "blocking_failures": failures,
        "all_required_gates_pass": not failures,
        "evidence_sha256": {
            path.relative_to(ROOT).as_posix(): file_sha256(path) for path in evidence_paths.values()
        },
    }


def run(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    require_clean_worktree(ROOT)
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    preregistration_document = str(config["preregistration_document"])
    registry.register_preregistration(
        str(config["preregistration_id"]),
        str(config["preregistration_version"]),
        preregistration_document,
        file_sha256(ROOT / preregistration_document),
    )
    registry.register_hypothesis(
        "H0",
        "No intrinsic complex advantage remains after rigorous resource matching.",
        status="open",
    )
    experiment_id, result_directory = registry.reserve_named(
        str(config["experiment_id"]), config, ROOT / "experiments" / "results"
    )
    registry.attach_protocol(
        experiment_id,
        str(config["preregistration_id"]),
        str(config["hypothesis_id"]),
        [sys.executable, "-m", "experiments.run_qn_grand_001"],
    )
    config_path = result_directory / "config.yaml"
    environment_path = result_directory / "environment.json"
    preflight_path = result_directory / "preflight.json"
    decision_path = result_directory / "decision.json"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    environment_path.write_text(
        json.dumps(
            environment_record(
                ROOT, command=[sys.executable, "-m", "experiments.run_qn_grand_001"]
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    preflight = build_preflight(config)
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    if preflight["all_required_gates_pass"]:
        raise RuntimeError("all gates passed but the sealed execution engine is not available")
    decision = {
        "experiment_id": experiment_id,
        "status": "blocked_before_execution",
        "qn_grand_001_executed": False,
        "sealed_benchmark_opened": False,
        "primary_confirmatory_effect_estimated": False,
        "blocking_failures": preflight["blocking_failures"],
        "decision_rule": "Any failed required preflight gate blocks sealed data access.",
        "scientific_interpretation": (
            "QN-GRAND-001 was not executed. The primary confirmatory architecture claim remains "
            "untested under the grand protocol; reduced prior evidence cannot be relabeled."
        ),
    }
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    registry.block(
        experiment_id,
        [
            ("configuration", config_path),
            ("environment", environment_path),
            ("preflight", preflight_path),
            ("decision", decision_path),
        ],
    )
    return result_directory, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "qn_grand_001.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    directory, decision = run(config)
    print(json.dumps({**decision, "result_directory": str(directory)}, sort_keys=True))


if __name__ == "__main__":
    main()
