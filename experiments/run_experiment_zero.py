"""Run the first controlled Q-Neuro comparison and persist all artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from neuroworld import CounterfactualPair, NeuroWorld
from qneuro.data import CaseDataset, collate_cases, shuffled_tokens
from qneuro.metrics import aggregate_seed_metrics, classification_metrics
from qneuro.model_factory import build_model, parameter_count
from qneuro.models import ComplexOperatorState, RealOperatorState
from qneuro.registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def make_loader(
    cases: list,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CaseDataset(cases),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_cases,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_bins: int,
    shuffle_order: bool = False,
    phase_mode: str = "learned",
    seed: int = 0,
) -> dict[str, float]:
    model.eval()
    torch.manual_seed(seed)
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    order_parts: list[torch.Tensor] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = to_device(raw_batch, device)
        if shuffle_order:
            batch = shuffled_tokens(batch, seed + batch_index)
        logits = model(**batch, phase_mode=phase_mode)
        logits_parts.append(logits.detach().cpu())
        label_parts.append(batch["label"].detach().cpu())
        order_parts.append(batch["is_order"].detach().cpu())
    return classification_metrics(
        torch.cat(logits_parts), torch.cat(label_parts), torch.cat(order_parts), n_bins=n_bins
    )


@torch.no_grad()
def evaluate_counterfactuals(
    model: nn.Module,
    pairs: list[CounterfactualPair],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    cases = [case for pair in pairs for case in (pair.first, pair.second)]
    loader = make_loader(cases, batch_size, shuffle=False, seed=0)
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        predictions.append(model(**batch).argmax(dim=-1).cpu())
        labels.append(batch["label"].cpu())
    prediction = torch.cat(predictions).reshape(-1, 2)
    label = torch.cat(labels).reshape(-1, 2)
    correct = prediction.eq(label)
    return {
        "counterfactual_case_accuracy": float(correct.float().mean()),
        "counterfactual_pair_accuracy": float(correct.all(dim=1).float().mean()),
        "counterfactual_flip_rate": float(prediction[:, 0].ne(prediction[:, 1]).float().mean()),
    }


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * batch["label"].shape[0]
            total_examples += batch["label"].shape[0]
            peak_rss = max(peak_rss, process.memory_info().rss)

        validation = evaluate(model, validation_loader, device, n_bins=10)
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
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["patience"]):
                break

    if best_state is None:
        raise RuntimeError("training completed without a finite validation checkpoint")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return model, {
        "best_epoch": best_epoch,
        "training_seconds": elapsed,
        "peak_rss_bytes": peak_rss,
        "history": history,
    }


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def environment_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_commit": git_revision(),
        "git_dirty": git_is_dirty(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_memory_bytes": psutil.virtual_memory().total,
    }


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output["dataset"].update(
        train_cases=800,
        validation_cases=200,
        test_cases=300,
        counterfactual_pairs=50,
    )
    output["training"].update(seeds=[11], epochs=2, patience=2, batch_size=128)
    return output


def run(config: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    result_root = ROOT / "experiments" / "results"
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    experiment_id, result_directory = registry.reserve(config, result_root)
    print(f"{experiment_id}: writing to {result_directory}", flush=True)
    try:
        (result_directory / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        environment = environment_metadata()
        (result_directory / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )

        dataset_config = config["dataset"]
        world = NeuroWorld(
            world_seed=int(dataset_config["world_seed"]),
            observation_probability=float(dataset_config["observation_probability"]),
        )
        train_cases = world.generate(int(dataset_config["train_cases"]), seed=1001)
        validation_cases = world.generate(int(dataset_config["validation_cases"]), seed=2001)
        test_cases = world.generate(int(dataset_config["test_cases"]), seed=3001)
        counterfactuals = world.counterfactual_pairs(
            int(dataset_config["counterfactual_pairs"]), seed=4001
        )

        training_config = config["training"]
        model_config = config["models"]
        evaluation_config = config["evaluation"]
        device = torch.device(training_config["device"])
        all_seed_results: dict[str, list[dict[str, float]]] = {
            name: [] for name in model_config["names"]
        }
        detailed_runs: list[dict[str, Any]] = []

        for seed in training_config["seeds"]:
            seed = int(seed)
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
                print(
                    f"{experiment_id} seed={seed} model={model_name} "
                    f"parameters={parameter_count(model)}",
                    flush=True,
                )
                trained, resource_metrics = train_one(
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
                    {
                        "training_seconds": float(resource_metrics["training_seconds"]),
                        "peak_rss_gib": float(resource_metrics["peak_rss_bytes"] / 1024**3),
                        "parameter_count": float(model_metadata["parameter_count"]),
                    }
                )
                if bool(evaluation_config["shuffled_order"]):
                    shuffled = evaluate(
                        trained,
                        test_loader,
                        device,
                        n_bins=int(evaluation_config["calibration_bins"]),
                        shuffle_order=True,
                        seed=seed,
                    )
                    metrics["shuffled_top1"] = shuffled["top1"]
                    metrics["shuffle_delta"] = metrics["top1"] - shuffled["top1"]
                if isinstance(trained, ComplexOperatorState) and bool(
                    evaluation_config["complex_phase_ablations"]
                ):
                    zero_phase = evaluate(
                        trained,
                        test_loader,
                        device,
                        n_bins=int(evaluation_config["calibration_bins"]),
                        phase_mode="zero",
                        seed=seed,
                    )
                    randomized_phase = evaluate(
                        trained,
                        test_loader,
                        device,
                        n_bins=int(evaluation_config["calibration_bins"]),
                        phase_mode="randomized",
                        seed=seed,
                    )
                    metrics["zero_phase_top1"] = zero_phase["top1"]
                    metrics["randomized_phase_top1"] = randomized_phase["top1"]
                if isinstance(trained, (RealOperatorState, ComplexOperatorState)):
                    norms = [trained.commutator_norm(2 * pair, 2 * pair + 1) for pair in range(4)]
                    metrics["mean_order_marker_commutator_norm"] = float(np.mean(norms))

                all_seed_results[model_name].append(metrics)
                detailed_runs.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "model_metadata": model_metadata,
                        "metrics": metrics,
                        "resources": resource_metrics,
                    }
                )
                print(
                    f"{experiment_id} seed={seed} model={model_name} "
                    f"top1={metrics['top1']:.4f} order={metrics['order_accuracy']:.4f} "
                    f"nll={metrics['nll']:.4f}",
                    flush=True,
                )

        summaries = {
            model_name: aggregate_seed_metrics(results)
            for model_name, results in all_seed_results.items()
        }
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "summary": summaries,
            "runs": detailed_runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            summaries,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "experiment_zero.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = smoke_config(config)
        config["description"] += " [SMOKE PROFILE]"
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, metrics in results["summary"].items():
        print(
            f"{model:18s} top1={metrics['top1']['mean']:.4f} "
            f"order={metrics['order_accuracy']['mean']:.4f} nll={metrics['nll']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
