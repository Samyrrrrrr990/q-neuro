"""Run composition, ambiguity, unknown-disease, and hidden-syndrome experiments."""

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
    collect_outputs,
    environment_metadata,
    evaluate,
    make_loader,
)
from experiments.run_generator_shift import build_world, train_with_validation_tuning
from neuroworld import (
    NeuroWorld,
    ambiguous_order_pairs,
    composition_reference_cases,
    composition_split,
    hidden_syndrome_cases,
    label_filtered_cases,
)
from qneuro.evaluation import (
    ambiguity_pair_metrics,
    binary_auroc,
    collect_representations,
    nearest_centroid_scores,
    ood_metrics,
    silhouette_binary,
)
from qneuro.metrics import aggregate_seed_metrics
from qneuro.registry import ExperimentRegistry


def save_checkpoint(
    model: torch.nn.Module,
    selected: dict[str, Any],
    context: str,
    seed: int,
    directory: Path,
) -> Path:
    path = directory / f"{context}_{selected['model_metadata']['name']}_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_metadata": selected["model_metadata"],
            "selected_learning_rate": selected["learning_rate"],
            "training_seed": seed,
            "context": context,
        },
        path,
    )
    return path


def representation_ood_metrics(
    model: torch.nn.Module,
    train_cases: list,
    id_cases: list,
    ood_cases: list,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    train_representation, train_labels = collect_representations(
        model, make_loader(train_cases, batch_size, False, 0), device
    )
    id_representation, _ = collect_representations(
        model, make_loader(id_cases, batch_size, False, 0), device
    )
    ood_representation, _ = collect_representations(
        model, make_loader(ood_cases, batch_size, False, 0), device
    )
    id_distance = nearest_centroid_scores(train_representation, train_labels, id_representation)
    ood_distance = nearest_centroid_scores(train_representation, train_labels, ood_representation)
    return {
        "representation_ood_auroc": binary_auroc(id_distance, ood_distance),
        "id_centroid_distance": float(id_distance.mean()),
        "ood_centroid_distance": float(ood_distance.mean()),
        "representation_silhouette": silhouette_binary(id_representation, ood_representation),
    }


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
        task_config = config["tasks"]
        training_config = config["training"]
        evaluation_config = config["evaluation"]
        model_names = list(config["models"]["names"])
        device = torch.device(training_config["device"])
        batch_size = int(training_config["batch_size"])

        base_config = task_config["base"]
        base_train = world.generate(int(base_config["train_cases"]), seed=5001)
        base_validation = world.generate(int(base_config["validation_cases"]), seed=5002)
        base_test = world.generate(int(base_config["test_cases"]), seed=5003)
        ambiguity_pairs = ambiguous_order_pairs(
            world, int(base_config["ambiguity_pairs"]), seed=5004
        )
        ambiguity_cases = [case for pair in ambiguity_pairs for case in (pair.first, pair.second)]
        hidden_cases = hidden_syndrome_cases(int(base_config["hidden_syndrome_cases"]), seed=5005)
        hidden_id_cases = base_test[: len(hidden_cases)]

        composition_config = task_config["composition"]
        composition_train, composition_validation, composition_test = composition_split(
            world,
            int(composition_config["train_cases"]),
            int(composition_config["validation_cases"]),
            int(composition_config["test_cases"]),
            seed=6001,
        )
        composition_reference = composition_reference_cases(
            world,
            int(composition_config["reference_cases"]),
            seed=6004,
        )

        unknown_config = task_config["unknown_disease"]
        heldout_label = int(unknown_config["heldout_label"])
        known_labels = set(range(NeuroWorld.num_diagnoses)) - {heldout_label}
        unknown_train = label_filtered_cases(
            world, int(unknown_config["train_cases"]), 7001, known_labels
        )
        unknown_validation = label_filtered_cases(
            world, int(unknown_config["validation_cases"]), 7002, known_labels
        )
        unknown_id_test = label_filtered_cases(
            world, int(unknown_config["id_test_cases"]), 7003, known_labels
        )
        unknown_ood_test = label_filtered_cases(
            world, int(unknown_config["ood_test_cases"]), 7004, {heldout_label}
        )

        task_records: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
        runs: list[dict[str, Any]] = []
        checkpoint_directory = result_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_artifacts: list[tuple[str, Path]] = []

        contexts = {
            "base": (base_train, base_validation),
            "composition": (composition_train, composition_validation),
            "unknown_disease": (unknown_train, unknown_validation),
        }
        for seed_value in training_config["seeds"]:
            seed = int(seed_value)
            for model_name in model_names:
                context_results: dict[str, Any] = {}
                for context, (train_cases, validation_cases) in contexts.items():
                    model, selected, tuning_trials = train_with_validation_tuning(
                        model_name,
                        seed,
                        train_cases,
                        validation_cases,
                        config,
                        device,
                    )
                    checkpoint_path = save_checkpoint(
                        model, selected, context, seed, checkpoint_directory
                    )
                    checkpoint_artifacts.append(("checkpoint", checkpoint_path))

                    if context == "base":
                        base_metrics = evaluate(
                            model,
                            make_loader(base_test, batch_size, False, seed),
                            device,
                            n_bins=int(evaluation_config["calibration_bins"]),
                            seed=seed,
                        )
                        ambiguity_outputs = collect_outputs(
                            model,
                            make_loader(ambiguity_cases, batch_size, False, seed),
                            device,
                            seed=seed,
                        )
                        base_metrics.update(
                            ambiguity_pair_metrics(
                                ambiguity_outputs["logits"], ambiguity_outputs["labels"]
                            )
                        )
                        hidden_outputs = collect_outputs(
                            model,
                            make_loader(hidden_cases, batch_size, False, seed),
                            device,
                            seed=seed,
                        )
                        hidden_id_outputs = collect_outputs(
                            model,
                            make_loader(hidden_id_cases, batch_size, False, seed),
                            device,
                            seed=seed,
                        )
                        base_metrics.update(
                            {
                                f"hidden_{key}": value
                                for key, value in ood_metrics(
                                    hidden_id_outputs["logits"], hidden_outputs["logits"]
                                ).items()
                            }
                        )
                        base_metrics.update(
                            {
                                f"hidden_{key}": value
                                for key, value in representation_ood_metrics(
                                    model,
                                    base_train[:1000],
                                    hidden_id_cases,
                                    hidden_cases,
                                    batch_size,
                                    device,
                                ).items()
                            }
                        )
                        metrics = base_metrics
                    elif context == "composition":
                        composition_metrics = evaluate(
                            model,
                            make_loader(composition_test, batch_size, False, seed),
                            device,
                            n_bins=int(evaluation_config["calibration_bins"]),
                            seed=seed,
                        )
                        reference_metrics = evaluate(
                            model,
                            make_loader(composition_reference, batch_size, False, seed),
                            device,
                            n_bins=int(evaluation_config["calibration_bins"]),
                            seed=seed,
                        )
                        metrics = {
                            f"composition_{key}": value
                            for key, value in composition_metrics.items()
                        }
                        metrics.update(
                            {f"reference_{key}": value for key, value in reference_metrics.items()}
                        )
                        metrics["composition_generalization_gap"] = (
                            reference_metrics["top1"] - composition_metrics["top1"]
                        )
                    else:
                        id_outputs = collect_outputs(
                            model,
                            make_loader(unknown_id_test, batch_size, False, seed),
                            device,
                            seed=seed,
                        )
                        ood_outputs = collect_outputs(
                            model,
                            make_loader(unknown_ood_test, batch_size, False, seed),
                            device,
                            seed=seed,
                        )
                        id_metrics = evaluate(
                            model,
                            make_loader(unknown_id_test, batch_size, False, seed),
                            device,
                            n_bins=int(evaluation_config["calibration_bins"]),
                            seed=seed,
                        )
                        metrics = {f"unknown_id_{key}": value for key, value in id_metrics.items()}
                        metrics.update(ood_metrics(id_outputs["logits"], ood_outputs["logits"]))
                        metrics.update(
                            representation_ood_metrics(
                                model,
                                unknown_train[:1000],
                                unknown_id_test,
                                unknown_ood_test,
                                batch_size,
                                device,
                            )
                        )
                        predictions = ood_outputs["logits"].argmax(dim=-1)
                        metrics["heldout_label_prediction_rate"] = float(
                            predictions.eq(heldout_label).float().mean()
                        )

                    metrics["parameter_count"] = float(
                        selected["model_metadata"]["parameter_count"]
                    )
                    metrics["training_seconds"] = float(selected["resources"]["training_seconds"])
                    task_records[(model_name, context)].append(metrics)
                    context_results[context] = {
                        "metrics": metrics,
                        "selected_trial": selected,
                        "tuning_trials": tuning_trials,
                        "checkpoint": str(checkpoint_path),
                    }
                    del model
                    gc.collect()

                runs.append({"seed": seed, "model": model_name, "contexts": context_results})
                print(
                    f"{experiment_id} seed={seed} model={model_name} "
                    f"base={context_results['base']['metrics']['top1']:.3f} "
                    f"composition={context_results['composition']['metrics']['composition_top1']:.3f} "
                    f"ood={context_results['unknown_disease']['metrics']['ood_auroc_msp']:.3f} "
                    f"hidden={context_results['base']['metrics']['hidden_representation_ood_auroc']:.3f}",
                    flush=True,
                )

        summary = {
            model_name: {
                context: aggregate_seed_metrics(task_records[(model_name, context)])
                for context in contexts
            }
            for model_name in model_names
        }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "heldout_disease_label": heldout_label,
            "summary": summary,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        flat_summary = {
            f"{model}@{context}": metrics
            for model, contexts_summary in summary.items()
            for context, metrics in contexts_summary.items()
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
        default=ROOT / "experiments" / "configs" / "neuro_task_suite.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["training"].update(seeds=[11], epochs=2, patience=2)
        config["tasks"]["base"].update(
            train_cases=300,
            validation_cases=100,
            test_cases=120,
            ambiguity_pairs=30,
            hidden_syndrome_cases=40,
        )
        config["tasks"]["composition"].update(
            train_cases=300, validation_cases=100, test_cases=120, reference_cases=120
        )
        config["tasks"]["unknown_disease"].update(
            train_cases=300,
            validation_cases=100,
            id_test_cases=120,
            ood_test_cases=80,
        )
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, contexts in results["summary"].items():
        print(
            f"{model:22s} base={contexts['base']['top1']['mean']:.3f} "
            f"composition={contexts['composition']['composition_top1']['mean']:.3f} "
            f"ood={contexts['unknown_disease']['ood_auroc_msp']['mean']:.3f} "
            f"hidden={contexts['base']['hidden_representation_ood_auroc']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
