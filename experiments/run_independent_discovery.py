"""Run the reduced, outcome-ineligible independent-task law discovery split."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.run_experiment_zero import ROOT, evaluate, evaluate_counterfactuals, make_loader
from experiments.run_generator_shift import train_with_validation_tuning
from independent_tasks import GENERATOR_VERSION, build_independent_task
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry
from research.computational_laws import fit_candidate_laws


def _metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key in {"top1", "nll", "ece", "brier", "order_accuracy", "non_order_accuracy"}
    }


def _effects_and_laws(
    records: list[dict[str, Any]], model_names: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, float]]]:
    index = {
        (
            record["family"],
            record["train_size"],
            record["training_seed"],
            record["world_seed"],
            record["severity"],
            record["model"],
        ): record
        for record in records
    }
    real_models = [name for name in model_names if name != "complex_operator"]
    effects: list[dict[str, Any]] = []
    for key, complex_record in index.items():
        if key[-1] != "complex_operator":
            continue
        cell = key[:-1]
        real_records = [index[(*cell, model)] for model in real_models]
        best_real = max(
            real_records,
            key=lambda record: (record["metrics"]["top1"], record["model"]),
        )
        effects.append(
            {
                "family": cell[0],
                "train_size": cell[1],
                "training_seed": cell[2],
                "world_seed": cell[3],
                "severity": cell[4],
                "order_dependence": complex_record["order_dependence"],
                "order_information": complex_record["order_information"],
                "complex_top1": complex_record["metrics"]["top1"],
                "best_real_model": best_real["model"],
                "best_real_top1": best_real["metrics"]["top1"],
                "difference": complex_record["metrics"]["top1"] - best_real["metrics"]["top1"],
            }
        )

    aggregate: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for effect in effects:
        aggregate[(effect["family"], float(effect["severity"]))].append(effect)
    law_cells = [
        {
            "family": family,
            "severity": severity,
            "order_dependence": float(np.mean([value["order_dependence"] for value in values])),
            "order_information": float(np.mean([value["order_information"] for value in values])),
            "advantage": float(np.mean([value["difference"] for value in values])),
            "standard_deviation": float(np.std([value["difference"] for value in values], ddof=1)),
            "n_nested_cells": len(values),
        }
        for (family, severity), values in sorted(aggregate.items())
    ]
    candidates = fit_candidate_laws(
        [cell["order_information"] for cell in law_cells],
        [cell["severity"] for cell in law_cells],
        [cell["advantage"] for cell in law_cells],
    )
    return effects, {name: law.to_dict() for name, law in candidates.items()}, law_cells


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output["description"] += " [SMOKE PROFILE; NOT DISCOVERY EVIDENCE]"
    output["profile"] = "smoke"
    output["families"] = {
        name: output["families"][name]
        for name in (
            "hidden_causal_machine",
            "analytic_noncommutative",
            "analytic_commutative",
        )
    }
    output["dataset"].update(
        train_sizes=[120],
        validation_cases=80,
        test_cases_per_world=50,
        counterfactual_pairs_per_world=10,
        world_seeds=[82001, 82003],
    )
    output["training"].update(seeds=[1103], epochs=2, patience=2)
    output["models"]["names"] = [
        "complex_operator",
        "exact_real_block_operator",
        "state_space",
    ]
    output["models"]["learning_rates"] = {
        name: output["models"]["learning_rates"][name] for name in output["models"]["names"]
    }
    return output


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_independent_discovery"]
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
        if str(config["generator_version"]) != GENERATOR_VERSION:
            raise ValueError("independent-task generator version mismatch")
        device = torch.device(config["training"]["device"])
        records: list[dict[str, Any]] = []
        training_runs: list[dict[str, Any]] = []
        counterfactual_records: list[dict[str, Any]] = []
        for family_index, (family, family_config) in enumerate(config["families"].items()):
            source_task = build_independent_task(
                family,
                order_dependence=family_config.get("order_dependence"),
                sequence_length=family_config.get("sequence_length"),
                world_seed=int(config["source_world_seed"] + family_index),
            )
            maximum_size = max(int(value) for value in config["dataset"]["train_sizes"])
            train_pool = list(
                source_task.generate(
                    maximum_size,
                    int(config["dataset"]["train_seed"]),
                    split="train",
                ).cases
            )
            validation_cases = list(
                source_task.generate(
                    int(config["dataset"]["validation_cases"]),
                    int(config["dataset"]["validation_seed"]),
                    split="validation",
                ).cases
            )
            for train_size_value in config["dataset"]["train_sizes"]:
                train_size = int(train_size_value)
                for training_seed_value in config["training"]["seeds"]:
                    training_seed = int(training_seed_value)
                    for model_name in config["models"]["names"]:
                        model, selected, trials = train_with_validation_tuning(
                            model_name,
                            training_seed,
                            train_pool[:train_size],
                            validation_cases,
                            config,
                            device,
                        )
                        training_runs.append(
                            {
                                "family": family,
                                "train_size": train_size,
                                "training_seed": training_seed,
                                "model": model_name,
                                "selected_trial": selected,
                                "tuning_trials": trials,
                            }
                        )
                        for world_seed_value in config["dataset"]["world_seeds"]:
                            world_seed = int(world_seed_value)
                            task = build_independent_task(
                                family,
                                order_dependence=family_config.get("order_dependence"),
                                sequence_length=family_config.get("sequence_length"),
                                world_seed=world_seed,
                            )
                            for severity_value in config["severities"]:
                                severity = float(severity_value)
                                dataset = task.generate(
                                    int(config["dataset"]["test_cases_per_world"]),
                                    int(config["dataset"]["test_seed"]),
                                    split="test",
                                    shift_strength=severity,
                                )
                                metrics = evaluate(
                                    model,
                                    make_loader(
                                        list(dataset.cases),
                                        int(config["training"]["batch_size"]),
                                        False,
                                        training_seed,
                                    ),
                                    device,
                                    n_bins=int(config["evaluation"]["calibration_bins"]),
                                    seed=training_seed,
                                )
                                records.append(
                                    {
                                        "family": family,
                                        "train_size": train_size,
                                        "training_seed": training_seed,
                                        "world_seed": world_seed,
                                        "severity": severity,
                                        "model": model_name,
                                        "order_dependence": float(
                                            dataset.metadata["analytic_normalized_commutator"]
                                        ),
                                        "order_information": float(
                                            dataset.metadata[
                                                "empirical_observed_order_target_mutual_information"
                                            ]
                                        ),
                                        "metrics": _metric_subset(metrics),
                                    }
                                )
                            pairs = list(
                                task.counterfactual_pairs(
                                    int(config["dataset"]["counterfactual_pairs_per_world"]),
                                    int(config["dataset"]["counterfactual_seed"]),
                                )
                            )
                            pair_metrics = evaluate_counterfactuals(
                                model,
                                pairs,
                                int(config["training"]["batch_size"]),
                                device,
                            )
                            counterfactual_records.append(
                                {
                                    "family": family,
                                    "train_size": train_size,
                                    "training_seed": training_seed,
                                    "world_seed": world_seed,
                                    "model": model_name,
                                    "causal_order": task.definition.causal_order,
                                    "metrics": pair_metrics,
                                }
                            )
                        print(
                            f"{experiment_id} family={family} n={train_size} "
                            f"seed={training_seed} model={model_name}",
                            flush=True,
                        )
                        del model
                        gc.collect()
        effects, candidates, law_cells = _effects_and_laws(records, list(config["models"]["names"]))
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "profile": config.get("profile", "reduced_discovery"),
            "split": "discovery",
            "outcome_eligible": False,
            "general_law_claim_permitted": False,
            "protocol_deviations": config["protocol_deviations"],
            "training_runs": training_runs,
            "records": records,
            "counterfactual_records": counterfactual_records,
            "paired_effects": effects,
            "law_cells": law_cells,
            "candidate_laws": candidates,
            "scientific_interpretation": (
                "Reduced discovery results may propose a provisional law but cannot satisfy the "
                "preregistered general-law or primary architecture claim."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry.complete(
            experiment_id,
            {
                "discovery": {
                    "training_runs": {"mean": float(len(training_runs)), "std": 0.0},
                    "law_cells": {"mean": float(len(law_cells)), "std": 0.0},
                }
            },
            [
                ("configuration", config_path),
                ("environment", environment_path),
                ("metrics", metrics_path),
            ],
        )
        return experiment_id, result_directory, result
    except Exception as error:
        (result_directory / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "independent_discovery",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        registry.record_failure(
            experiment_id,
            "independent_discovery",
            type(error).__name__,
            str(error),
        )
        registry.fail(experiment_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "independent_discovery.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="reserved for smoke debugging")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = smoke_config(config)
    experiment_id, directory, result = run(config, allow_dirty=args.allow_dirty)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "result_directory": str(directory),
                "profile": result["profile"],
                "training_runs": len(result["training_runs"]),
                "records": len(result["records"]),
                "law_cells": len(result["law_cells"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
