"""Run active evidence acquisition with random, fixed, and expected-information policies."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata, evaluate, make_loader
from experiments.run_generator_shift import build_world, train_with_validation_tuning
from experiments.run_neuro_task_suite import save_checkpoint
from neuroworld import NeuroWorld, label_filtered_cases
from qneuro.evaluation import (
    active_trajectory,
    aggregate_active_trajectories,
    canonicalize_case,
    estimate_positive_likelihoods,
    global_information_order,
)
from qneuro.metrics import aggregate_seed_metrics
from qneuro.registry import ExperimentRegistry


def flatten_policy_metrics(policy: dict[str, object]) -> dict[str, float]:
    flattened = {
        key: float(policy[key])
        for key in (
            "accuracy_auc",
            "final_accuracy",
            "final_nll",
            "resolution_rate",
            "mean_queries_to_resolution_penalized",
            "median_queries_to_resolution_penalized",
            "policy_seconds",
        )
    }
    for point in policy["curve"]:
        queries = int(point["queries"])
        for metric in ("accuracy", "nll", "mean_confidence", "mean_entropy"):
            flattened[f"{metric}_at_{queries}_queries"] = float(point[metric])
    return flattened


def run(config: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    environment = environment_metadata()
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    experiment_id, result_directory = registry.reserve(config, ROOT / "experiments" / "results")
    print(f"{experiment_id}: writing to {result_directory}", flush=True)
    try:
        (result_directory / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (result_directory / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )
        world = build_world(config["world"])
        complete_world_config = dict(config["world"])
        complete_world_config["observation_probability"] = 1.0
        complete_world = build_world(complete_world_config)
        training_config = config["training"]
        evaluation_config = config["evaluation"]
        included_labels = set(range(8, NeuroWorld.num_diagnoses))
        train_cases = [
            canonicalize_case(case)
            for case in label_filtered_cases(
                world, int(training_config["train_cases"]), 8101, included_labels
            )
        ]
        validation_cases = [
            canonicalize_case(case)
            for case in label_filtered_cases(
                world, int(training_config["validation_cases"]), 8102, included_labels
            )
        ]
        test_cases = [
            canonicalize_case(case)
            for case in label_filtered_cases(
                complete_world, int(evaluation_config["test_cases"]), 8103, included_labels
            )
        ]
        positive_likelihoods = estimate_positive_likelihoods(train_cases)
        fixed_order = global_information_order(train_cases)
        device = torch.device(training_config["device"])
        batch_size = int(training_config["batch_size"])
        max_queries = int(evaluation_config["max_queries"])
        threshold = float(evaluation_config["confidence_threshold"])

        runs: list[dict[str, Any]] = []
        policy_records: dict[tuple[str, str], list[dict[str, float]]] = {}
        full_records: dict[str, list[dict[str, float]]] = {
            name: [] for name in config["models"]["names"]
        }
        checkpoint_directory = result_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_artifacts: list[tuple[str, Path]] = []
        for seed_value in training_config["seeds"]:
            seed = int(seed_value)
            for model_name in config["models"]["names"]:
                model, selected, tuning_trials = train_with_validation_tuning(
                    model_name,
                    seed,
                    train_cases,
                    validation_cases,
                    config,
                    device,
                )
                checkpoint_path = save_checkpoint(
                    model, selected, "active_evidence", seed, checkpoint_directory
                )
                checkpoint_artifacts.append(("checkpoint", checkpoint_path))
                full_metrics = evaluate(
                    model,
                    make_loader(test_cases, batch_size, False, seed),
                    device,
                    n_bins=int(evaluation_config["calibration_bins"]),
                    seed=seed,
                )
                full_records[model_name].append(full_metrics)
                policies: dict[str, Any] = {}
                for strategy_index, strategy in enumerate(evaluation_config["strategies"]):
                    start = time.perf_counter()
                    trajectories = [
                        (
                            case.label,
                            active_trajectory(
                                model,
                                case,
                                strategy,
                                max_queries,
                                fixed_order,
                                positive_likelihoods,
                                random_seed=(
                                    1_000_003 * seed + 101 * case.case_id + 10_007 * strategy_index
                                ),
                                device=device,
                            ),
                        )
                        for case in test_cases
                    ]
                    policy = aggregate_active_trajectories(trajectories, threshold)
                    policy["policy_seconds"] = time.perf_counter() - start
                    policies[strategy] = policy
                    policy_records.setdefault((model_name, strategy), []).append(
                        flatten_policy_metrics(policy)
                    )
                runs.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "full_information_metrics": full_metrics,
                        "policies": policies,
                        "selected_trial": selected,
                        "tuning_trials": tuning_trials,
                        "checkpoint": str(checkpoint_path),
                    }
                )
                print(
                    f"{experiment_id} seed={seed} model={model_name} "
                    f"full={full_metrics['top1']:.3f} "
                    + " ".join(
                        f"{strategy}={policies[strategy]['accuracy_auc']:.3f}"
                        for strategy in evaluation_config["strategies"]
                    ),
                    flush=True,
                )
                del model
                gc.collect()

        summary = {
            model_name: {
                "full_information": aggregate_seed_metrics(full_records[model_name]),
                "policies": {
                    strategy: aggregate_seed_metrics(policy_records[(model_name, strategy)])
                    for strategy in evaluation_config["strategies"]
                },
            }
            for model_name in config["models"]["names"]
        }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "task_scope": "factorial labels 8-19; temporal-order twins excluded",
            "fixed_information_order": fixed_order,
            "fixed_information_names": [NeuroWorld.finding_names[index] for index in fixed_order],
            "summary": summary,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            {
                f"{model}@{strategy}": summary[model]["policies"][strategy]
                for model in summary
                for strategy in evaluation_config["strategies"]
            },
            [
                ("configuration", result_directory / "config.yaml"),
                ("environment", result_directory / "environment.json"),
                ("metrics", metrics_path),
                *checkpoint_artifacts,
            ],
        )
        return experiment_id, result_directory, results
    except Exception:
        registry.fail(experiment_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "active_evidence.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["training"].update(
            seeds=[11], train_cases=300, validation_cases=100, epochs=2, patience=2
        )
        config["evaluation"].update(test_cases=12, max_queries=3)
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, summary in results["summary"].items():
        policy_summary = summary["policies"]
        print(
            f"{model:22s} full={summary['full_information']['top1']['mean']:.3f} "
            + " ".join(
                f"{strategy}={policy_summary[strategy]['accuracy_auc']['mean']:.3f}"
                for strategy in config["evaluation"]["strategies"]
            )
        )


if __name__ == "__main__":
    main()
