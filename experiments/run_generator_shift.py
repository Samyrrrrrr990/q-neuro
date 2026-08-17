"""Replicate Experiment Zero under generator shift with stronger real controls."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import (
    ROOT,
    environment_metadata,
    evaluate,
    evaluate_counterfactuals,
    make_loader,
    set_seed,
    train_one,
)
from neuroworld import NeuroWorld
from qneuro.metrics import aggregate_seed_metrics
from qneuro.model_factory import build_model
from qneuro.registry import ExperimentRegistry


def build_world(config: dict[str, Any]) -> NeuroWorld:
    return NeuroWorld(
        world_seed=int(config["world_seed"]),
        observation_probability=float(config["observation_probability"]),
        probability_mixing=float(config["probability_mixing"]),
        temporal_jitter=float(config["temporal_jitter"]),
        order_marker_visibility=float(config["order_marker_visibility"]),
        demographic_signal_strength=float(config.get("demographic_signal_strength", 1.0)),
        shared_nuisance_stages=bool(config.get("shared_nuisance_stages", False)),
    )


def train_with_validation_tuning(
    model_name: str,
    seed: int,
    train_cases: list,
    validation_cases: list,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]]]:
    model_config = config["models"]
    training_config = config["training"]
    learning_rates = [float(value) for value in model_config["learning_rates"][model_name]]
    trials: list[dict[str, Any]] = []
    best_model: torch.nn.Module | None = None
    best_record: dict[str, Any] | None = None
    best_validation_nll = float("inf")

    for learning_rate in learning_rates:
        set_seed(seed)
        candidate, model_metadata = build_model(
            model_name,
            int(model_config["parameter_budget"]),
            int(model_config["operator_rank"]),
            int(model_config["max_sequence_length"]),
            float(model_config["step_size"]),
        )
        candidate_training = copy.deepcopy(training_config)
        candidate_training["learning_rate"] = learning_rate
        train_loader = make_loader(train_cases, int(training_config["batch_size"]), True, seed)
        validation_loader = make_loader(
            validation_cases, int(training_config["batch_size"]), False, seed
        )
        candidate, resources = train_one(
            candidate, train_loader, validation_loader, candidate_training, device
        )
        validation_metrics = evaluate(
            candidate,
            validation_loader,
            device,
            n_bins=int(config["evaluation"]["calibration_bins"]),
            seed=seed,
        )
        trial = {
            "learning_rate": learning_rate,
            "validation_metrics": validation_metrics,
            "resources": resources,
            "model_metadata": model_metadata,
        }
        trials.append(trial)
        if validation_metrics["nll"] < best_validation_nll:
            best_validation_nll = validation_metrics["nll"]
            if best_model is not None:
                del best_model
            best_model = candidate
            best_record = trial
        else:
            del candidate
            gc.collect()

    if best_model is None or best_record is None:
        raise RuntimeError(f"no valid hyperparameter trial for {model_name}")
    return best_model, best_record, trials


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

        train_world = build_world(config["train_world"])
        dataset_config = config["dataset"]
        train_sizes = [int(size) for size in dataset_config["train_sizes"]]
        train_pool = train_world.generate(max(train_sizes), seed=1001)
        validation_cases = train_world.generate(int(dataset_config["validation_cases"]), seed=2001)

        evaluation_sets: dict[str, dict[str, Any]] = {}
        for environment_name, world_config in config["evaluation_worlds"].items():
            world = build_world(world_config)
            evaluation_sets[environment_name] = {
                "cases": world.generate(int(dataset_config["test_cases"]), seed=3001),
                "counterfactuals": world.counterfactual_pairs(
                    int(dataset_config["counterfactual_pairs"]), seed=4001
                ),
            }

        training_config = config["training"]
        model_config = config["models"]
        evaluation_config = config["evaluation"]
        device = torch.device(training_config["device"])
        raw_results: dict[tuple[str, int, str], list[dict[str, float]]] = defaultdict(list)
        runs: list[dict[str, Any]] = []

        for size in train_sizes:
            train_cases = train_pool[:size]
            for seed_value in training_config["seeds"]:
                seed = int(seed_value)
                for model_name in model_config["names"]:
                    trained, selected, tuning_trials = train_with_validation_tuning(
                        model_name,
                        seed,
                        train_cases,
                        validation_cases,
                        config,
                        device,
                    )
                    environment_metrics: dict[str, dict[str, float]] = {}
                    for environment_name, evaluation_set in evaluation_sets.items():
                        test_loader = make_loader(
                            evaluation_set["cases"],
                            int(training_config["batch_size"]),
                            False,
                            seed,
                        )
                        metrics = evaluate(
                            trained,
                            test_loader,
                            device,
                            n_bins=int(evaluation_config["calibration_bins"]),
                            seed=seed,
                        )
                        metrics.update(
                            evaluate_counterfactuals(
                                trained,
                                evaluation_set["counterfactuals"],
                                int(training_config["batch_size"]),
                                device,
                            )
                        )
                        metrics["selected_learning_rate"] = float(selected["learning_rate"])
                        metrics["selected_training_seconds"] = float(
                            selected["resources"]["training_seconds"]
                        )
                        metrics["total_tuning_seconds"] = float(
                            sum(trial["resources"]["training_seconds"] for trial in tuning_trials)
                        )
                        metrics["parameter_count"] = float(
                            selected["model_metadata"]["parameter_count"]
                        )
                        environment_metrics[environment_name] = metrics
                        raw_results[(model_name, size, environment_name)].append(metrics)

                    runs.append(
                        {
                            "train_cases": size,
                            "seed": seed,
                            "model": model_name,
                            "selected_trial": selected,
                            "tuning_trials": tuning_trials,
                            "environment_metrics": environment_metrics,
                        }
                    )
                    id_metrics = environment_metrics["in_domain"]
                    shift_metrics = environment_metrics["noisy_sparse_shift"]
                    print(
                        f"{experiment_id} n={size} seed={seed} model={model_name} "
                        f"lr={selected['learning_rate']:.4g} id={id_metrics['top1']:.4f} "
                        f"shift={shift_metrics['top1']:.4f} "
                        f"shift_order={shift_metrics['order_accuracy']:.4f}",
                        flush=True,
                    )
                    del trained
                    gc.collect()

        summary = {
            model_name: {
                str(size): {
                    environment_name: aggregate_seed_metrics(
                        raw_results[(model_name, size, environment_name)]
                    )
                    for environment_name in evaluation_sets
                }
                for size in train_sizes
            }
            for model_name in model_config["names"]
        }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "summary": summary,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        flat_summary = {
            f"{model}@n={size}@{environment_name}": metrics
            for model, size_map in summary.items()
            for size, environment_map in size_map.items()
            for environment_name, metrics in environment_map.items()
        }
        registry.complete(
            experiment_id,
            flat_summary,
            [
                ("configuration", result_directory / "config.yaml"),
                ("environment", result_directory / "environment.json"),
                ("metrics", metrics_path),
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
        default=ROOT / "experiments" / "configs" / "experiment_zero_generator_shift.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["dataset"].update(
            train_sizes=[250], validation_cases=200, test_cases=300, counterfactual_pairs=50
        )
        config["training"].update(seeds=[11], epochs=2, patience=2)
        config["models"]["learning_rates"] = {
            name: [rates[0]] for name, rates in config["models"]["learning_rates"].items()
        }
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, size_map in results["summary"].items():
        values = []
        for size, environments in size_map.items():
            in_domain = environments["in_domain"]["top1"]["mean"]
            shifted = environments["noisy_sparse_shift"]["top1"]["mean"]
            values.append(f"n={size}:id={in_domain:.3f}/shift={shifted:.3f}")
        print(f"{model:22s} {', '.join(values)}")


if __name__ == "__main__":
    main()
