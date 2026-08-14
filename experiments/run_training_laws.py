"""Compare global, phase-coded, local, hybrid, and zero-backprop training laws."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from torch import nn

from experiments.run_dynamics_suite import output_metrics
from experiments.run_experiment_zero import (
    ROOT,
    collect_outputs,
    environment_metadata,
    evaluate,
    evaluate_counterfactuals,
    make_loader,
    set_seed,
    to_device,
)
from experiments.run_generator_shift import build_world
from neuroworld import NeuroWorld, ambiguous_order_pairs
from qneuro.evaluation import ambiguity_pair_metrics
from qneuro.learning import (
    AuxiliaryTrainingModel,
    apply_pcgrad,
    apply_phase_gradient,
    fit_centroid_readout,
    local_plasticity_epoch,
    multi_objective_losses,
)
from qneuro.metrics import aggregate_seed_metrics
from qneuro.model_factory import build_model, parameter_count
from qneuro.models import ComplexOperatorState
from qneuro.registry import ExperimentRegistry

GLOBAL_METHODS = {
    "adamw",
    "sgd",
    "gradient_accumulation",
    "multiobjective_adamw",
    "pcgrad",
    "phase_gradient",
}


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _mean_diagnostics(values: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for value in values for key in value})
    return {key: float(np.mean([value[key] for value in values if key in value])) for key in keys}


def _train_global(
    base: ComplexOperatorState,
    method: str,
    train_cases: list,
    validation_cases: list,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[ComplexOperatorState, dict[str, Any]]:
    training = config["training"]
    method_config = config["methods"][method]
    accumulation_steps = int(method_config.get("accumulation_steps", 1))
    batch_size = int(training["batch_size"]) // accumulation_steps
    train_loader = make_loader(train_cases, batch_size, True, seed)
    validation_loader = make_loader(validation_cases, int(training["batch_size"]), False, seed)
    auxiliary = method in {"multiobjective_adamw", "pcgrad", "phase_gradient"}
    training_model: nn.Module = AuxiliaryTrainingModel(base) if auxiliary else base
    training_model.to(device)
    optimizer_class = torch.optim.SGD if method == "sgd" else torch.optim.AdamW
    optimizer = optimizer_class(
        training_model.parameters(),
        lr=float(method_config["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    process = psutil.Process(os.getpid())
    starting_rss = process.memory_info().rss
    peak_rss = starting_rss
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    gradient_diagnostics: list[dict[str, float]] = []
    backward_passes = 0
    autograd_gradient_calls = 0
    optimizer_steps = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(training["epochs"]) + 1):
        training_model.train()
        total_loss = 0.0
        total_examples = 0
        batch_count = 0
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device)
            if auxiliary:
                losses = multi_objective_losses(
                    training_model,
                    batch,
                    auxiliary_weight=float(training["auxiliary_weight"]),
                )
                loss = sum(losses.values())
            else:
                losses = {}
                loss = torch.nn.functional.cross_entropy(base(**batch), batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {method} training loss")
            if method == "pcgrad":
                optimizer.zero_grad(set_to_none=True)
                gradient_diagnostics.append(apply_pcgrad(losses, training_model))
                autograd_gradient_calls += len(losses)
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
            elif method == "phase_gradient":
                optimizer.zero_grad(set_to_none=True)
                gradient_diagnostics.append(apply_phase_gradient(losses, training_model))
                autograd_gradient_calls += len(losses)
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
            else:
                (loss / accumulation_steps).backward()
                backward_passes += 1
                batch_count += 1
                if batch_count % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(training_model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            total_loss += float(loss.detach()) * batch["label"].shape[0]
            total_examples += batch["label"].shape[0]
            peak_rss = max(peak_rss, process.memory_info().rss)
        if method not in {"pcgrad", "phase_gradient"} and batch_count % accumulation_steps:
            torch.nn.utils.clip_grad_norm_(training_model.parameters(), 5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        validation = evaluate(base, validation_loader, device, n_bins=10, seed=seed)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / total_examples,
                "validation_nll": validation["nll"],
                "validation_top1": validation["top1"],
            }
        )
        if validation["nll"] < best_nll - 1e-5:
            best_nll = validation["nll"]
            best_epoch = epoch
            best_state = _clone_state(training_model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"{method} produced no finite validation checkpoint")
    training_model.load_state_dict(best_state)
    return base, {
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "peak_rss_delta_bytes": max(0, peak_rss - starting_rss),
        "backward_passes": backward_passes,
        "autograd_gradient_calls": autograd_gradient_calls,
        "optimizer_steps": optimizer_steps,
        "local_epochs": 0,
        "history": history,
        "gradient_diagnostics": _mean_diagnostics(gradient_diagnostics),
        "training_parameter_count": parameter_count(training_model),
    }


def _train_local_or_hybrid(
    base: ComplexOperatorState,
    method: str,
    train_cases: list,
    validation_cases: list,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[ComplexOperatorState, dict[str, Any]]:
    training = config["training"]
    method_config = config["methods"][method]
    train_loader = make_loader(train_cases, int(training["batch_size"]), True, seed)
    centroid_loader = make_loader(train_cases, int(training["batch_size"]), False, seed)
    validation_loader = make_loader(validation_cases, int(training["batch_size"]), False, seed)
    base.to(device)
    process = psutil.Process(os.getpid())
    starting_rss = process.memory_info().rss
    peak_rss = starting_rss
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    local_epochs = int(method_config.get("local_epochs", 0))
    local_diagnostics: list[dict[str, float]] = []
    if method == "zerobackprop":
        local_diagnostics.append(fit_centroid_readout(base, centroid_loader))
        validation = evaluate(base, validation_loader, device, n_bins=10, seed=seed)
        history.append(
            {
                "epoch": 0,
                "train_loss": float("nan"),
                "validation_nll": validation["nll"],
                "validation_top1": validation["top1"],
            }
        )
        best_nll = validation["nll"]
        best_state = _clone_state(base)
    else:
        for epoch in range(1, local_epochs + 1):
            diagnostics = local_plasticity_epoch(
                base,
                train_loader,
                learning_rate=float(method_config["learning_rate"]),
                seed=seed,
            )
            diagnostics.update(fit_centroid_readout(base, centroid_loader))
            local_diagnostics.append(diagnostics)
            validation = evaluate(base, validation_loader, device, n_bins=10, seed=seed)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": diagnostics["local_transition_mse"],
                    "validation_nll": validation["nll"],
                    "validation_top1": validation["top1"],
                }
            )
            if validation["nll"] < best_nll:
                best_nll = validation["nll"]
                best_epoch = epoch
                best_state = _clone_state(base)
            peak_rss = max(peak_rss, process.memory_info().rss)
    if best_state is None:
        raise RuntimeError(f"{method} produced no local checkpoint")
    base.load_state_dict(best_state)
    local_seconds = time.perf_counter() - started
    if method == "hybrid_local_global":
        global_config = copy.deepcopy(config)
        global_config["methods"]["adamw"] = {
            "learning_rate": float(method_config["global_learning_rate"])
        }
        trained, resources = _train_global(
            base,
            "adamw",
            train_cases,
            validation_cases,
            global_config,
            seed,
            device,
        )
        resources["training_seconds"] += local_seconds
        resources["peak_rss_bytes"] = max(resources["peak_rss_bytes"], peak_rss)
        resources["peak_rss_delta_bytes"] = max(
            resources["peak_rss_delta_bytes"], peak_rss - starting_rss
        )
        resources["local_epochs"] = local_epochs
        resources["local_diagnostics"] = _mean_diagnostics(local_diagnostics)
        resources["history"] = history + resources["history"]
        return trained, resources
    return base, {
        "best_epoch": best_epoch,
        "training_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "peak_rss_delta_bytes": max(0, peak_rss - starting_rss),
        "backward_passes": 0,
        "autograd_gradient_calls": 0,
        "optimizer_steps": 0,
        "local_epochs": local_epochs,
        "history": history,
        "gradient_diagnostics": {},
        "local_diagnostics": _mean_diagnostics(local_diagnostics),
        "training_parameter_count": parameter_count(base),
    }


def train_method(
    method: str,
    train_cases: list,
    validation_cases: list,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[ComplexOperatorState, dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    model, metadata = build_model(
        "complex_operator",
        int(config["model"]["parameter_budget"]),
        int(config["model"]["operator_rank"]),
        int(config["model"]["max_sequence_length"]),
        float(config["model"]["step_size"]),
    )
    if not isinstance(model, ComplexOperatorState):
        raise TypeError("training-law suite requires ComplexOperatorState")
    if method in GLOBAL_METHODS:
        model, resources = _train_global(
            model, method, train_cases, validation_cases, config, seed, device
        )
    else:
        model, resources = _train_local_or_hybrid(
            model, method, train_cases, validation_cases, config, seed, device
        )
    return model, resources, metadata


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
        dataset = config["dataset"]
        training = config["training"]
        train_world = build_world(config["train_world"])
        train_sizes = [int(value) for value in dataset["train_sizes"]]
        train_pool = train_world.generate(max(train_sizes), seed=12001)
        validation_cases = train_world.generate(int(dataset["validation_cases"]), seed=12002)
        in_domain_cases = train_world.generate(int(dataset["test_cases"]), seed=12003)
        ambiguity_pairs = ambiguous_order_pairs(
            train_world, int(dataset["ambiguity_pairs"]), seed=12004
        )
        ambiguity_cases = [case for pair in ambiguity_pairs for case in (pair.first, pair.second)]
        in_domain_pairs = train_world.counterfactual_pairs(
            int(dataset["counterfactual_pairs"]), seed=12005
        )
        shifted_sets: dict[int, dict[str, Any]] = {}
        for world_seed_value in dataset["unseen_world_seeds"]:
            world_seed = int(world_seed_value)
            world = NeuroWorld(world_seed=world_seed, **config["shift"])
            shifted_sets[world_seed] = {
                "cases": world.generate(int(dataset["test_cases"]), seed=12003),
                "counterfactuals": world.counterfactual_pairs(
                    int(dataset["counterfactual_pairs"]), seed=12005
                ),
            }
        device = torch.device(training["device"])
        methods = list(config["methods"])
        records: dict[tuple[str, int, str], list[dict[str, float]]] = defaultdict(list)
        runs: list[dict[str, Any]] = []
        checkpoint_directory = result_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        artifacts: list[tuple[str, Path]] = []
        for train_size in train_sizes:
            train_cases = train_pool[:train_size]
            for seed_value in training["seeds"]:
                seed = int(seed_value)
                for method in methods:
                    model, resources, metadata = train_method(
                        method, train_cases, validation_cases, config, seed, device
                    )
                    in_outputs = collect_outputs(
                        model,
                        make_loader(in_domain_cases, int(training["batch_size"]), False, seed),
                        device,
                        seed=seed,
                    )
                    in_metrics = output_metrics(
                        in_outputs, int(config["evaluation"]["calibration_bins"])
                    )
                    in_metrics.update(
                        evaluate_counterfactuals(
                            model, in_domain_pairs, int(training["batch_size"]), device
                        )
                    )
                    ambiguity_outputs = collect_outputs(
                        model,
                        make_loader(ambiguity_cases, int(training["batch_size"]), False, seed),
                        device,
                        seed=seed,
                    )
                    in_metrics.update(
                        ambiguity_pair_metrics(
                            ambiguity_outputs["logits"], ambiguity_outputs["labels"]
                        )
                    )
                    in_metrics.update(
                        {
                            "training_seconds": float(resources["training_seconds"]),
                            "peak_rss_gib": float(resources["peak_rss_bytes"] / 1024**3),
                            "peak_rss_delta_gib": float(
                                resources["peak_rss_delta_bytes"] / 1024**3
                            ),
                            "backward_passes": float(resources["backward_passes"]),
                            "autograd_gradient_calls": float(resources["autograd_gradient_calls"]),
                            "optimizer_steps": float(resources["optimizer_steps"]),
                            "deploy_parameter_count": float(metadata["parameter_count"]),
                            "training_parameter_count": float(
                                resources["training_parameter_count"]
                            ),
                        }
                    )
                    records[(method, train_size, "in_domain")].append(in_metrics)
                    shifted_metrics: dict[str, dict[str, float]] = {}
                    for world_seed, evaluation_set in shifted_sets.items():
                        outputs = collect_outputs(
                            model,
                            make_loader(
                                evaluation_set["cases"],
                                int(training["batch_size"]),
                                False,
                                seed,
                            ),
                            device,
                            seed=seed,
                        )
                        metrics = output_metrics(
                            outputs, int(config["evaluation"]["calibration_bins"])
                        )
                        metrics.update(
                            evaluate_counterfactuals(
                                model,
                                evaluation_set["counterfactuals"],
                                int(training["batch_size"]),
                                device,
                            )
                        )
                        records[(method, train_size, f"world_{world_seed}")].append(metrics)
                        shifted_metrics[str(world_seed)] = metrics
                    checkpoint_path = checkpoint_directory / f"{method}_n{train_size}_seed{seed}.pt"
                    torch.save(
                        {
                            "model_state_dict": _clone_state(model),
                            "model_metadata": metadata,
                            "method": method,
                            "train_cases": train_size,
                            "seed": seed,
                        },
                        checkpoint_path,
                    )
                    artifacts.append(("checkpoint", checkpoint_path))
                    runs.append(
                        {
                            "method": method,
                            "train_cases": train_size,
                            "seed": seed,
                            "model_metadata": metadata,
                            "resources": resources,
                            "in_domain_metrics": in_metrics,
                            "shifted_metrics": shifted_metrics,
                            "checkpoint": str(checkpoint_path),
                        }
                    )
                    shift_top1 = np.mean([value["top1"] for value in shifted_metrics.values()])
                    print(
                        f"{experiment_id} n={train_size} seed={seed} method={method} "
                        f"id={in_metrics['top1']:.3f} shift={shift_top1:.3f} "
                        f"backward={resources['backward_passes']}",
                        flush=True,
                    )
                    del model
                    gc.collect()
        summary: dict[str, Any] = {}
        for method in methods:
            summary[method] = {}
            for train_size in train_sizes:
                worlds = {
                    str(world_seed): aggregate_seed_metrics(
                        records[(method, train_size, f"world_{world_seed}")]
                    )
                    for world_seed in shifted_sets
                }
                across_worlds: dict[str, Any] = {}
                metric_names = sorted(
                    set.intersection(*(set(values) for values in worlds.values()))
                )
                for metric_name in metric_names:
                    world_means = [worlds[str(seed)][metric_name]["mean"] for seed in shifted_sets]
                    across_worlds[metric_name] = aggregate_seed_metrics(
                        [{metric_name: value} for value in world_means]
                    )[metric_name]
                summary[method][str(train_size)] = {
                    "in_domain": aggregate_seed_metrics(records[(method, train_size, "in_domain")]),
                    "shifted": {"by_world": worlds, "across_worlds": across_worlds},
                }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "architecture": "complex_operator",
            "shift_statistical_unit": "unseen world seed",
            "summary": summary,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            {
                f"{method}@n={size}": values[str(size)]["shifted"]["across_worlds"]
                for method, values in summary.items()
                for size in train_sizes
            },
            [
                ("configuration", result_directory / "config.yaml"),
                ("environment", result_directory / "environment.json"),
                ("metrics", metrics_path),
                *artifacts,
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
        default=ROOT / "experiments" / "configs" / "training_laws.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["dataset"].update(
            train_sizes=[120],
            validation_cases=120,
            test_cases=160,
            ambiguity_pairs=30,
            counterfactual_pairs=30,
            unseen_world_seeds=[141421],
        )
        config["training"].update(seeds=[11], epochs=2, patience=2)
        config["methods"]["local_plasticity"]["local_epochs"] = 1
        config["methods"]["hybrid_local_global"]["local_epochs"] = 1
    experiment_id, result_directory, _ = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))


if __name__ == "__main__":
    main()
