"""Evaluate emergent hierarchical observables in frozen Q-Neuro and baseline states."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata, make_loader
from experiments.run_generator_shift import build_world
from qneuro.evaluation import collect_representations
from qneuro.metrics import aggregate_seed_metrics
from qneuro.model_factory import build_model
from qneuro.observables import (
    HermitianObservableProbe,
    LinearObservableProbe,
    factorial_property_labels,
    fit_probe,
)
from qneuro.registry import ExperimentRegistry


def factorial_subset(
    representation: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = labels >= 8
    return representation[selected], labels[selected]


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
        source_directory = ROOT / "experiments" / "results" / config["source_experiment"]
        source_config = yaml.safe_load(
            (source_directory / "config.yaml").read_text(encoding="utf-8")
        )
        source_metrics = json.loads((source_directory / "metrics.json").read_text(encoding="utf-8"))
        world = build_world(source_config["train_world"])
        train_cases = world.generate(int(source_config["dataset"]["train_cases"]), seed=9101)
        test_cases = world.generate(int(source_config["dataset"]["test_cases"]), seed=9103)
        batch_size = int(config["evaluation"]["batch_size"])
        device = torch.device(config["evaluation"]["device"])
        source_runs = {
            (run_record["model"], int(run_record["seed"])): run_record
            for run_record in source_metrics["runs"]
        }
        seeds = sorted({seed for _, seed in source_runs})
        records: dict[tuple[str, str, str], list[dict[str, float]]] = {}
        runs: list[dict[str, Any]] = []
        probe_directory = result_directory / "checkpoints"
        probe_directory.mkdir(parents=True, exist_ok=True)
        probe_artifacts: list[tuple[str, Path]] = []
        for model_name in config["models"]["names"]:
            for seed in seeds:
                model, metadata = build_model(
                    model_name,
                    int(source_config["models"]["parameter_budget"]),
                    int(source_config["models"]["operator_rank"]),
                    int(source_config["models"]["max_sequence_length"]),
                    float(source_config["models"]["step_size"]),
                )
                checkpoint = torch.load(
                    source_runs[(model_name, seed)]["checkpoint"],
                    map_location=device,
                    weights_only=True,
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(device)
                train_representation, train_labels = collect_representations(
                    model, make_loader(train_cases, batch_size, False, seed), device
                )
                test_representation, test_labels = collect_representations(
                    model, make_loader(test_cases, batch_size, False, seed), device
                )
                train_representation, train_labels = factorial_subset(
                    train_representation, train_labels
                )
                test_representation, test_labels = factorial_subset(
                    test_representation, test_labels
                )
                train_properties = factorial_property_labels(train_labels)
                test_properties = factorial_property_labels(test_labels)
                probe_results: dict[str, dict[str, dict[str, float]]] = {}
                started = time.perf_counter()
                for property_name, class_count in config["probe"]["properties"].items():
                    linear_probe, linear_metrics = fit_probe(
                        lambda dimension=train_representation.shape[1], count=int(class_count): (
                            LinearObservableProbe(dimension, count)
                        ),
                        train_representation,
                        train_properties[property_name],
                        test_representation,
                        test_properties[property_name],
                        seed=seed,
                        epochs=int(config["probe"]["epochs"]),
                        learning_rate=float(config["probe"]["learning_rate"]),
                        weight_decay=float(config["probe"]["weight_decay"]),
                    )
                    probe_path = (
                        probe_directory / f"{model_name}_{property_name}_linear_seed{seed}.pt"
                    )
                    torch.save(linear_probe.state_dict(), probe_path)
                    probe_artifacts.append(("observable_probe", probe_path))
                    probe_results.setdefault(property_name, {})["linear"] = linear_metrics
                    records.setdefault((model_name, property_name, "linear"), []).append(
                        linear_metrics
                    )
                    if model_name in config["models"]["hermitian_observable_models"]:
                        if train_representation.shape[1] % 2:
                            raise RuntimeError("complex representation has an odd real width")
                        state_dim = train_representation.shape[1] // 2
                        hermitian_probe, hermitian_metrics = fit_probe(
                            lambda dimension=state_dim, count=int(class_count): (
                                HermitianObservableProbe(dimension, count)
                            ),
                            train_representation,
                            train_properties[property_name],
                            test_representation,
                            test_properties[property_name],
                            seed=seed,
                            epochs=int(config["probe"]["epochs"]),
                            learning_rate=float(config["probe"]["learning_rate"]),
                            weight_decay=float(config["probe"]["weight_decay"]),
                            standardize=False,
                        )
                        probe_path = (
                            probe_directory
                            / f"{model_name}_{property_name}_hermitian_seed{seed}.pt"
                        )
                        torch.save(hermitian_probe.state_dict(), probe_path)
                        probe_artifacts.append(("observable_probe", probe_path))
                        probe_results[property_name]["hermitian"] = hermitian_metrics
                        records.setdefault((model_name, property_name, "hermitian"), []).append(
                            hermitian_metrics
                        )
                runs.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "model_metadata": metadata,
                        "train_factorial_cases": int(train_labels.numel()),
                        "test_factorial_cases": int(test_labels.numel()),
                        "probe_seconds": time.perf_counter() - started,
                        "probes": probe_results,
                    }
                )
                print(
                    f"{experiment_id} seed={seed} model={model_name} "
                    + " ".join(
                        f"{name}={probe_results[name]['linear']['accuracy']:.3f}"
                        for name in config["probe"]["properties"]
                    ),
                    flush=True,
                )
        summary = {
            model: {
                property_name: {
                    probe_type: aggregate_seed_metrics(records[(model, property_name, probe_type)])
                    for probe_type in (
                        ("linear", "hermitian")
                        if model in config["models"]["hermitian_observable_models"]
                        else ("linear",)
                    )
                }
                for property_name in config["probe"]["properties"]
            }
            for model in config["models"]["names"]
        }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "source_experiment": config["source_experiment"],
            "probe_scope": "frozen representations; factorial labels 8-19 only",
            "summary": summary,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            {
                f"{model}@{property_name}@{probe_type}": metrics
                for model, properties in summary.items()
                for property_name, probe_types in properties.items()
                for probe_type, metrics in probe_types.items()
            },
            [
                ("configuration", result_directory / "config.yaml"),
                ("environment", result_directory / "environment.json"),
                ("metrics", metrics_path),
                *probe_artifacts,
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
        default=ROOT / "experiments" / "configs" / "observable_probe.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["models"]["names"] = ["real_operator", "complex_operator"]
        config["models"]["hermitian_observable_models"] = ["complex_operator"]
        config["probe"]["epochs"] = 10
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, properties in results["summary"].items():
        print(
            f"{model:22s} "
            + " ".join(
                f"{name}={values['linear']['accuracy']['mean']:.3f}"
                for name, values in properties.items()
            )
        )


if __name__ == "__main__":
    main()
