"""Run nested-data learning curves for the Experiment Zero model families."""

from __future__ import annotations

import argparse
import json
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

        dataset_config = config["dataset"]
        training_config = config["training"]
        model_config = config["models"]
        evaluation_config = config["evaluation"]
        train_sizes = [int(size) for size in dataset_config["train_sizes"]]
        world = NeuroWorld(
            world_seed=int(dataset_config["world_seed"]),
            observation_probability=float(dataset_config["observation_probability"]),
        )
        train_pool = world.generate(max(train_sizes), seed=1001)
        validation_cases = world.generate(int(dataset_config["validation_cases"]), seed=2001)
        test_cases = world.generate(int(dataset_config["test_cases"]), seed=3001)
        counterfactuals = world.counterfactual_pairs(
            int(dataset_config["counterfactual_pairs"]), seed=4001
        )
        device = torch.device(training_config["device"])

        raw_results: dict[tuple[str, int], list[dict[str, float]]] = {
            (name, size): [] for name in model_config["names"] for size in train_sizes
        }
        runs: list[dict[str, Any]] = []
        for size in train_sizes:
            train_cases = train_pool[:size]
            for seed_value in training_config["seeds"]:
                seed = int(seed_value)
                for model_name in model_config["names"]:
                    set_seed(seed)
                    model, model_metadata = build_model(
                        model_name,
                        int(model_config["parameter_budget"]),
                        int(model_config["operator_rank"]),
                        int(model_config["max_sequence_length"]),
                        float(model_config["step_size"]),
                    )
                    train_loader = make_loader(
                        train_cases, int(training_config["batch_size"]), True, seed
                    )
                    validation_loader = make_loader(
                        validation_cases, int(training_config["batch_size"]), False, seed
                    )
                    test_loader = make_loader(
                        test_cases, int(training_config["batch_size"]), False, seed
                    )
                    trained, resources = train_one(
                        model, train_loader, validation_loader, training_config, device
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
                            counterfactuals,
                            int(training_config["batch_size"]),
                            device,
                        )
                    )
                    metrics.update(
                        training_seconds=float(resources["training_seconds"]),
                        peak_rss_delta_gib=float(resources["peak_rss_delta_bytes"] / 1024**3),
                        parameter_count=float(model_metadata["parameter_count"]),
                    )
                    raw_results[(model_name, size)].append(metrics)
                    runs.append(
                        {
                            "train_cases": size,
                            "seed": seed,
                            "model": model_name,
                            "model_metadata": model_metadata,
                            "metrics": metrics,
                            "resources": resources,
                        }
                    )
                    print(
                        f"{experiment_id} n={size} seed={seed} model={model_name} "
                        f"top1={metrics['top1']:.4f} order={metrics['order_accuracy']:.4f}",
                        flush=True,
                    )

        summary = {
            model_name: {
                str(size): aggregate_seed_metrics(raw_results[(model_name, size)])
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
            f"{model}@n={size}": metrics
            for model, size_map in summary.items()
            for size, metrics in size_map.items()
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
        default=ROOT / "experiments" / "configs" / "experiment_zero_sample_efficiency.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, size_map in results["summary"].items():
        curve = ", ".join(
            f"n={size}:{metrics['top1']['mean']:.3f}" for size, metrics in size_map.items()
        )
        print(f"{model:18s} {curve}")


if __name__ == "__main__":
    main()
