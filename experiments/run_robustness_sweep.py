"""Run the hierarchical multi-world Q-Neuro robustness confirmation gate."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.run_experiment_zero import (
    ROOT,
    collect_outputs,
    environment_metadata,
    evaluate_counterfactuals,
    make_loader,
)
from experiments.run_generator_shift import build_world, train_with_validation_tuning
from neuroworld import NeuroWorld
from qneuro.calibration import apply_temperature, fit_temperature
from qneuro.metrics import aggregate_seed_metrics, classification_metrics
from qneuro.registry import ExperimentRegistry


def metrics_from_outputs(
    outputs: dict[str, torch.Tensor], temperature: float, n_bins: int
) -> dict[str, float]:
    raw = classification_metrics(
        outputs["logits"],
        outputs["labels"],
        outputs["is_order"],
        outputs["order_complete"],
        n_bins=n_bins,
    )
    calibrated = classification_metrics(
        apply_temperature(outputs["logits"], temperature),
        outputs["labels"],
        outputs["is_order"],
        outputs["order_complete"],
        n_bins=n_bins,
    )
    raw["calibrated_nll"] = calibrated["nll"]
    raw["calibrated_ece"] = calibrated["ece"]
    raw["temperature"] = float(temperature)
    return raw


def aggregate_world_hierarchy(
    records: dict[tuple[str, str, int], list[dict[str, float]]],
    model_names: list[str],
    severities: list[str],
    world_seeds: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in model_names:
        summary[model] = {}
        for severity in severities:
            worlds = {
                str(world_seed): aggregate_seed_metrics(records[(model, severity, world_seed)])
                for world_seed in world_seeds
            }
            world_mean_records: list[dict[str, float]] = []
            for world_seed in world_seeds:
                aggregates = worlds[str(world_seed)]
                world_mean_records.append(
                    {
                        metric: float(values["mean"])
                        for metric, values in aggregates.items()
                        if isinstance(values.get("mean"), (float, int))
                    }
                )
            summary[model][severity] = {
                "worlds": worlds,
                "across_worlds": aggregate_seed_metrics(world_mean_records),
            }
    return summary


def paired_world_effects(
    summary: dict[str, Any], model_names: list[str], severities: list[str]
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    baselines = [name for name in model_names if name != "complex_operator"]
    for severity in severities:
        effects[severity] = {}
        complex_worlds = summary["complex_operator"][severity]["worlds"]
        for baseline in baselines:
            effects[severity][baseline] = {}
            baseline_worlds = summary[baseline][severity]["worlds"]
            common_metrics = set.intersection(
                *(set(values) for values in [*complex_worlds.values(), *baseline_worlds.values()])
            )
            for metric in sorted(common_metrics):
                differences = []
                for world_seed in sorted(complex_worlds, key=int):
                    complex_mean = complex_worlds[world_seed][metric].get("mean")
                    baseline_mean = baseline_worlds[world_seed][metric].get("mean")
                    if isinstance(complex_mean, (float, int)) and isinstance(
                        baseline_mean, (float, int)
                    ):
                        differences.append({"difference": complex_mean - baseline_mean})
                if differences:
                    effects[severity][baseline][metric] = aggregate_seed_metrics(differences)[
                        "difference"
                    ]
    return effects


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
        evaluation_config = config["evaluation"]
        model_names = list(config["models"]["names"])
        world_seeds = [int(seed) for seed in dataset_config["unseen_world_seeds"]]
        severities = list(config["shift_severities"])
        device = torch.device(training_config["device"])

        train_world = build_world(config["train_world"])
        train_cases = train_world.generate(int(dataset_config["train_cases"]), seed=1001)
        validation_cases = train_world.generate(int(dataset_config["validation_cases"]), seed=2001)
        in_domain_cases = train_world.generate(
            int(dataset_config["test_cases_per_world"]), seed=3001
        )

        shifted_sets: dict[tuple[str, int], dict[str, Any]] = {}
        for severity, severity_config in config["shift_severities"].items():
            for world_seed in world_seeds:
                world = NeuroWorld(world_seed=world_seed, **severity_config)
                shifted_sets[(severity, world_seed)] = {
                    "cases": world.generate(int(dataset_config["test_cases_per_world"]), seed=3001),
                    "counterfactuals": world.counterfactual_pairs(
                        int(dataset_config["counterfactual_pairs_per_world"]), seed=4001
                    ),
                }

        records: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
        in_domain_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        runs: list[dict[str, Any]] = []
        checkpoint_artifacts: list[tuple[str, Path]] = []
        checkpoint_directory = result_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)

        for seed_value in training_config["seeds"]:
            training_seed = int(seed_value)
            for model_name in model_names:
                model, selected, tuning_trials = train_with_validation_tuning(
                    model_name,
                    training_seed,
                    train_cases,
                    validation_cases,
                    config,
                    device,
                )
                validation_loader = make_loader(
                    validation_cases, int(training_config["batch_size"]), False, training_seed
                )
                validation_outputs = collect_outputs(
                    model, validation_loader, device, seed=training_seed
                )
                temperature = fit_temperature(
                    validation_outputs["logits"], validation_outputs["labels"]
                )

                checkpoint_path = checkpoint_directory / f"{model_name}_seed{training_seed}.pt"
                torch.save(
                    {
                        "model_state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "model_metadata": selected["model_metadata"],
                        "training_seed": training_seed,
                        "selected_learning_rate": selected["learning_rate"],
                        "temperature": temperature,
                    },
                    checkpoint_path,
                )
                checkpoint_artifacts.append(("checkpoint", checkpoint_path))

                in_domain_loader = make_loader(
                    in_domain_cases, int(training_config["batch_size"]), False, training_seed
                )
                in_domain_outputs = collect_outputs(
                    model, in_domain_loader, device, seed=training_seed
                )
                in_domain_metrics = metrics_from_outputs(
                    in_domain_outputs,
                    temperature,
                    int(evaluation_config["calibration_bins"]),
                )
                in_domain_records[model_name].append(in_domain_metrics)

                shifted_metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
                for severity in severities:
                    for world_seed in world_seeds:
                        evaluation_set = shifted_sets[(severity, world_seed)]
                        test_loader = make_loader(
                            evaluation_set["cases"],
                            int(training_config["batch_size"]),
                            False,
                            training_seed,
                        )
                        outputs = collect_outputs(model, test_loader, device, seed=training_seed)
                        metrics = metrics_from_outputs(
                            outputs,
                            temperature,
                            int(evaluation_config["calibration_bins"]),
                        )
                        metrics.update(
                            evaluate_counterfactuals(
                                model,
                                evaluation_set["counterfactuals"],
                                int(training_config["batch_size"]),
                                device,
                            )
                        )
                        metrics["selected_learning_rate"] = float(selected["learning_rate"])
                        metrics["parameter_count"] = float(
                            selected["model_metadata"]["parameter_count"]
                        )
                        records[(model_name, severity, world_seed)].append(metrics)
                        shifted_metrics[severity][str(world_seed)] = metrics

                runs.append(
                    {
                        "training_seed": training_seed,
                        "model": model_name,
                        "selected_trial": selected,
                        "tuning_trials": tuning_trials,
                        "temperature": temperature,
                        "in_domain_metrics": in_domain_metrics,
                        "shifted_metrics": shifted_metrics,
                        "checkpoint": str(checkpoint_path),
                    }
                )
                severity_means = {
                    severity: float(
                        np.mean(
                            [shifted_metrics[severity][str(seed)]["top1"] for seed in world_seeds]
                        )
                    )
                    for severity in severities
                }
                print(
                    f"{experiment_id} seed={training_seed} model={model_name} "
                    f"id={in_domain_metrics['top1']:.3f} "
                    + " ".join(
                        f"{severity}={value:.3f}" for severity, value in severity_means.items()
                    ),
                    flush=True,
                )
                del model
                gc.collect()

        summary = aggregate_world_hierarchy(records, model_names, severities, world_seeds)
        for model_name in model_names:
            summary[model_name]["in_domain"] = aggregate_seed_metrics(in_domain_records[model_name])
        effects = paired_world_effects(summary, model_names, severities)
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "hierarchical_unit": "unseen_world_seed",
            "summary": summary,
            "paired_world_effects": effects,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        flat_summary = {
            f"{model}@{severity}": severity_map["across_worlds"]
            for model, model_summary in summary.items()
            for severity, severity_map in model_summary.items()
            if severity != "in_domain"
        }
        registry.complete(
            experiment_id,
            flat_summary,
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
        default=ROOT / "experiments" / "configs" / "robustness_world_sweep.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["dataset"].update(
            train_cases=250,
            validation_cases=200,
            test_cases_per_world=200,
            counterfactual_pairs_per_world=30,
            unseen_world_seeds=config["dataset"]["unseen_world_seeds"][:2],
        )
        config["shift_severities"] = {
            key: config["shift_severities"][key] for key in ("nuisance", "severe")
        }
        config["training"].update(seeds=[11], epochs=2, patience=2)
        config["models"]["learning_rates"] = {
            name: [rates[0]] for name, rates in config["models"]["learning_rates"].items()
        }
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, model_summary in results["summary"].items():
        values = [f"id={model_summary['in_domain']['top1']['mean']:.3f}"]
        for severity in config["shift_severities"]:
            top1 = model_summary[severity]["across_worlds"]["top1"]["mean"]
            values.append(f"{severity}={top1:.3f}")
        print(f"{model:22s} {' '.join(values)}")


if __name__ == "__main__":
    main()
