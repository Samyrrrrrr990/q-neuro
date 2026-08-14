"""Calibrate and evaluate realized velocity-based hard halting on trained attractor states."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import (
    ROOT,
    collect_outputs,
    environment_metadata,
    make_loader,
    to_device,
)
from experiments.run_generator_shift import build_world
from neuroworld import CounterfactualPair, NeuroWorld
from qneuro.metrics import aggregate_seed_metrics, classification_metrics
from qneuro.model_factory import build_model
from qneuro.models import EnergyAttractorState
from qneuro.registry import ExperimentRegistry


def load_source_model(
    source_directory: Path,
    source_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> EnergyAttractorState:
    model, _ = build_model(
        "adaptive_attractor",
        int(source_config["models"]["parameter_budget"]),
        int(source_config["models"]["operator_rank"]),
        int(source_config["models"]["max_sequence_length"]),
        float(source_config["models"]["step_size"]),
    )
    if not isinstance(model, EnergyAttractorState):
        raise TypeError("hard-halting source must be EnergyAttractorState")
    checkpoint = torch.load(
        source_directory / "checkpoints" / f"adaptive_attractor_seed{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def collect_hard_outputs(
    model: EnergyAttractorState,
    cases: list,
    batch_size: int,
    device: torch.device,
    threshold: float,
    min_steps: int,
) -> dict[str, torch.Tensor]:
    logits_parts: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    is_order: list[torch.Tensor] = []
    order_complete: list[torch.Tensor] = []
    steps: list[torch.Tensor] = []
    for raw_batch in make_loader(cases, batch_size, False, 0):
        batch = to_device(raw_batch, device)
        logits, executed = model.hard_halt(
            batch["tokens"],
            batch["mask"],
            batch["vector"],
            velocity_threshold=threshold,
            min_steps=min_steps,
        )
        logits_parts.append(logits.cpu())
        labels.append(batch["label"].cpu())
        is_order.append(batch["is_order"].cpu())
        order_complete.append(batch["order_complete"].cpu())
        steps.append(executed.cpu())
    return {
        "logits": torch.cat(logits_parts),
        "labels": torch.cat(labels),
        "is_order": torch.cat(is_order),
        "order_complete": torch.cat(order_complete),
        "steps": torch.cat(steps),
    }


def hard_metrics(outputs: dict[str, torch.Tensor], bins: int, total_steps: int) -> dict[str, float]:
    metrics = classification_metrics(
        outputs["logits"],
        outputs["labels"],
        outputs["is_order"],
        outputs["order_complete"],
        n_bins=bins,
    )
    steps = outputs["steps"].float()
    metrics.update(
        {
            "mean_executed_steps": float(steps.mean()),
            "median_executed_steps": float(steps.median()),
            "p95_executed_steps": float(torch.quantile(steps, 0.95)),
            "step_reduction_fraction": float(1.0 - steps.mean() / total_steps),
        }
    )
    for step in range(1, total_steps + 1):
        metrics[f"halt_fraction_step_{step}"] = float(steps.eq(step).float().mean())
    return metrics


@torch.no_grad()
def evaluate_hard_pairs(
    model: EnergyAttractorState,
    pairs: list[CounterfactualPair],
    batch_size: int,
    device: torch.device,
    threshold: float,
    min_steps: int,
) -> dict[str, float]:
    cases = [case for pair in pairs for case in (pair.first, pair.second)]
    outputs = collect_hard_outputs(model, cases, batch_size, device, threshold, min_steps)
    prediction = outputs["logits"].argmax(dim=-1).reshape(-1, 2)
    labels = outputs["labels"].reshape(-1, 2)
    correct = prediction.eq(labels)
    return {
        "counterfactual_case_accuracy": float(correct.float().mean()),
        "counterfactual_pair_accuracy": float(correct.all(dim=1).float().mean()),
        "counterfactual_flip_rate": float(prediction[:, 0].ne(prediction[:, 1]).float().mean()),
    }


@torch.no_grad()
def velocity_candidates(
    model: EnergyAttractorState,
    cases: list,
    batch_size: int,
    device: torch.device,
    quantiles: list[float],
) -> list[float]:
    values: list[torch.Tensor] = []
    for raw_batch in make_loader(cases, batch_size, False, 0):
        batch = to_device(raw_batch, device)
        diagnostics = model.trajectory_diagnostics(batch["tokens"], batch["mask"], batch["vector"])
        values.append(diagnostics["velocity"][:, 1:].flatten().cpu())
    velocities = torch.cat(values)
    candidates = {-1.0}
    candidates.update(float(torch.quantile(velocities, value)) for value in quantiles)
    return sorted(candidates)


def calibrate_threshold(
    model: EnergyAttractorState,
    validation_cases: list,
    batch_size: int,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[float, bool, list[dict[str, float]]]:
    calibration = config["calibration"]
    min_steps = int(calibration["min_steps"])
    bins = int(config["evaluation"]["calibration_bins"])
    candidates = velocity_candidates(
        model,
        validation_cases,
        batch_size,
        device,
        [float(value) for value in calibration["velocity_quantiles"]],
    )
    curve: list[dict[str, float]] = []
    for threshold in candidates:
        outputs = collect_hard_outputs(
            model, validation_cases, batch_size, device, threshold, min_steps
        )
        metrics = hard_metrics(outputs, bins, model.steps)
        curve.append({"threshold": threshold, **metrics})
    fixed = next(value for value in curve if value["threshold"] == -1.0)
    feasible = [
        value
        for value in curve
        if value["top1"] >= fixed["top1"] - float(calibration["top1_tolerance"])
        and value["nll"] <= fixed["nll"] + float(calibration["nll_tolerance"])
    ]
    constraint_satisfied = bool(feasible)
    if feasible:
        selected = min(feasible, key=lambda value: (value["mean_executed_steps"], value["nll"]))
    else:
        selected = min(
            curve,
            key=lambda value: (
                max(0.0, fixed["top1"] - value["top1"]),
                max(0.0, value["nll"] - fixed["nll"]),
                value["mean_executed_steps"],
            ),
        )
    return float(selected["threshold"]), constraint_satisfied, curve


@torch.no_grad()
def latency_per_case_ms(
    model: EnergyAttractorState,
    cases: list,
    batch_size: int,
    device: torch.device,
    mode: str,
    threshold: float,
    min_steps: int,
    repetitions: int,
) -> float:
    batches = [to_device(batch, device) for batch in make_loader(cases, batch_size, False, 0)]

    def execute() -> None:
        for batch in batches:
            if mode == "soft":
                model(**batch)
            else:
                model.hard_halt(
                    batch["tokens"],
                    batch["mask"],
                    batch["vector"],
                    velocity_threshold=threshold,
                    min_steps=min_steps,
                )

    execute()
    started = time.perf_counter()
    for _ in range(repetitions):
        execute()
    return 1000.0 * (time.perf_counter() - started) / (repetitions * len(cases))


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
        dataset = source_config["dataset"]
        training = source_config["training"]
        train_world = build_world(source_config["train_world"])
        validation_cases = train_world.generate(int(dataset["validation_cases"]), seed=9102)
        in_domain_cases = train_world.generate(int(dataset["test_cases"]), seed=9103)
        in_domain_pairs = train_world.counterfactual_pairs(
            int(dataset["counterfactual_pairs"]), seed=9105
        )
        shifted_sets: dict[int, dict[str, Any]] = {}
        for seed_value in dataset["unseen_world_seeds"]:
            world_seed = int(seed_value)
            world = NeuroWorld(world_seed=world_seed, **source_config["shift"])
            shifted_sets[world_seed] = {
                "cases": world.generate(int(dataset["test_cases"]), seed=9103),
                "pairs": world.counterfactual_pairs(
                    int(dataset["counterfactual_pairs"]), seed=9105
                ),
            }
        device = torch.device(training["device"])
        batch_size = int(training["batch_size"])
        min_steps = int(config["calibration"]["min_steps"])
        bins = int(config["evaluation"]["calibration_bins"])
        repetitions = int(config["evaluation"]["latency_repetitions"])
        modes = ("soft", "fixed_final", "hard")
        in_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        shift_records: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
        runs: list[dict[str, Any]] = []
        for seed_value in training["seeds"]:
            seed = int(seed_value)
            model = load_source_model(source_directory, source_config, seed, device)
            threshold, constraint_satisfied, calibration_curve = calibrate_threshold(
                model, validation_cases, batch_size, device, config
            )
            mode_thresholds = {"fixed_final": -1.0, "hard": threshold}
            in_metrics: dict[str, dict[str, float]] = {}
            soft_outputs = collect_outputs(
                model, make_loader(in_domain_cases, batch_size, False, seed), device, seed=seed
            )
            soft_metrics = classification_metrics(
                soft_outputs["logits"],
                soft_outputs["labels"],
                soft_outputs["is_order"],
                soft_outputs["order_complete"],
                n_bins=bins,
            )
            soft_metrics["latency_ms_per_case"] = latency_per_case_ms(
                model,
                in_domain_cases,
                batch_size,
                device,
                "soft",
                threshold,
                min_steps,
                repetitions,
            )
            in_metrics["soft"] = soft_metrics
            in_records["soft"].append(soft_metrics)
            for mode, mode_threshold in mode_thresholds.items():
                outputs = collect_hard_outputs(
                    model,
                    in_domain_cases,
                    batch_size,
                    device,
                    mode_threshold,
                    min_steps,
                )
                metrics = hard_metrics(outputs, bins, model.steps)
                metrics.update(
                    evaluate_hard_pairs(
                        model,
                        in_domain_pairs,
                        batch_size,
                        device,
                        mode_threshold,
                        min_steps,
                    )
                )
                metrics["latency_ms_per_case"] = latency_per_case_ms(
                    model,
                    in_domain_cases,
                    batch_size,
                    device,
                    "hard",
                    mode_threshold,
                    min_steps,
                    repetitions,
                )
                in_metrics[mode] = metrics
                in_records[mode].append(metrics)
            shifted_metrics: dict[str, dict[str, dict[str, float]]] = {}
            for world_seed, evaluation_set in shifted_sets.items():
                shifted_metrics[str(world_seed)] = {}
                soft = collect_outputs(
                    model,
                    make_loader(evaluation_set["cases"], batch_size, False, seed),
                    device,
                    seed=seed,
                )
                soft_shift = classification_metrics(
                    soft["logits"],
                    soft["labels"],
                    soft["is_order"],
                    soft["order_complete"],
                    n_bins=bins,
                )
                shifted_metrics[str(world_seed)]["soft"] = soft_shift
                shift_records[("soft", world_seed)].append(soft_shift)
                for mode, mode_threshold in mode_thresholds.items():
                    outputs = collect_hard_outputs(
                        model,
                        evaluation_set["cases"],
                        batch_size,
                        device,
                        mode_threshold,
                        min_steps,
                    )
                    metrics = hard_metrics(outputs, bins, model.steps)
                    metrics.update(
                        evaluate_hard_pairs(
                            model,
                            evaluation_set["pairs"],
                            batch_size,
                            device,
                            mode_threshold,
                            min_steps,
                        )
                    )
                    shifted_metrics[str(world_seed)][mode] = metrics
                    shift_records[(mode, world_seed)].append(metrics)
            runs.append(
                {
                    "seed": seed,
                    "selected_velocity_threshold": threshold,
                    "validation_constraint_satisfied": constraint_satisfied,
                    "calibration_curve": calibration_curve,
                    "in_domain_metrics": in_metrics,
                    "shifted_metrics": shifted_metrics,
                }
            )
            print(
                f"{experiment_id} seed={seed} threshold={threshold:.6f} "
                f"hard_steps={in_metrics['hard']['mean_executed_steps']:.2f}/{model.steps} "
                f"fixed={in_metrics['fixed_final']['top1']:.3f} "
                f"hard={in_metrics['hard']['top1']:.3f} "
                f"latency={in_metrics['hard']['latency_ms_per_case']:.4f}ms",
                flush=True,
            )
        summary: dict[str, Any] = {}
        for mode in modes:
            by_world = {
                str(world_seed): aggregate_seed_metrics(shift_records[(mode, world_seed)])
                for world_seed in shifted_sets
            }
            common_metrics = sorted(set.intersection(*(set(value) for value in by_world.values())))
            across_worlds = {
                metric: aggregate_seed_metrics(
                    [
                        {metric: by_world[str(world_seed)][metric]["mean"]}
                        for world_seed in shifted_sets
                    ]
                )[metric]
                for metric in common_metrics
            }
            summary[mode] = {
                "in_domain": aggregate_seed_metrics(in_records[mode]),
                "shifted": {"by_world": by_world, "across_worlds": across_worlds},
            }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "source_experiment": config["source_experiment"],
            "threshold_selection": "training-world validation only",
            "shift_statistical_unit": "unseen world seed",
            "summary": summary,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            {
                f"{mode}@shifted": values["shifted"]["across_worlds"]
                for mode, values in summary.items()
            },
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
        default=ROOT / "experiments" / "configs" / "hard_halting.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run fewer latency repetitions")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["calibration"]["velocity_quantiles"] = [0.25, 0.75]
        config["evaluation"]["latency_repetitions"] = 2
    experiment_id, result_directory, _ = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))


if __name__ == "__main__":
    main()
