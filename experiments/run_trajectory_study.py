"""Extract actual complex-state trajectories and aggregate evidence-response diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata, make_loader, to_device
from experiments.run_generator_shift import build_world
from neuroworld import NeuroWorld
from qneuro.metrics import aggregate_seed_metrics
from qneuro.model_factory import build_model
from qneuro.models import ComplexOperatorState
from qneuro.registry import ExperimentRegistry


def load_source_model(
    source_directory: Path,
    source_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> ComplexOperatorState:
    model, _ = build_model(
        "complex_operator",
        int(source_config["models"]["parameter_budget"]),
        int(source_config["models"]["operator_rank"]),
        int(source_config["models"]["max_sequence_length"]),
        float(source_config["models"]["step_size"]),
    )
    if not isinstance(model, ComplexOperatorState):
        raise TypeError("trajectory source must be ComplexOperatorState")
    checkpoint = torch.load(
        source_directory / "checkpoints" / f"complex_operator_seed{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def _probabilities(model: ComplexOperatorState, trajectory: torch.Tensor) -> torch.Tensor:
    flat = trajectory.reshape(-1, trajectory.shape[-1])
    logits = model.measure(flat).reshape(*trajectory.shape[:2], -1)
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def aggregate_trajectories(
    model: ComplexOperatorState,
    cases: list,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    path_lengths: list[float] = []
    final_velocities: list[float] = []
    entropy_changes: list[float] = []
    positive_deltas: list[float] = []
    negative_deltas: list[float] = []
    negative_drop_cases = 0
    revival_cases = 0
    correct = 0
    count = 0
    for raw_batch in make_loader(cases, batch_size, False, 0):
        batch = to_device(raw_batch, device)
        trajectory = model.trajectory(batch["tokens"], batch["mask"], batch["vector"])
        probabilities = _probabilities(model, trajectory)
        for row in range(batch["tokens"].shape[0]):
            length = int(batch["mask"][row].sum())
            states = trajectory[row, : length + 1]
            case_probabilities = probabilities[row, : length + 1]
            label = int(batch["label"][row])
            true_probability = case_probabilities[:, label]
            velocity = torch.linalg.vector_norm(states[1:] - states[:-1], dim=-1) / math.sqrt(
                model.state_dim
            )
            path_lengths.append(float(velocity.sum().cpu()))
            final_velocities.append(float(velocity[-1].cpu()))
            entropy = -(
                case_probabilities * torch.log(case_probabilities.clamp_min(1e-12))
            ).sum(dim=-1)
            entropy_changes.append(float((entropy[-1] - entropy[0]).cpu()))
            delta = true_probability[1:] - true_probability[:-1]
            tokens = batch["tokens"][row, :length]
            positive_deltas.extend(float(value) for value in delta[tokens < 40].cpu())
            negative_deltas.extend(float(value) for value in delta[tokens >= 40].cpu())
            drop_indices = torch.nonzero((tokens >= 40) & (delta < -0.05), as_tuple=False).flatten()
            if drop_indices.numel():
                negative_drop_cases += 1
                revived = False
                for drop_index in drop_indices.tolist():
                    if bool((true_probability[drop_index + 2 :] >= true_probability[drop_index]).any()):
                        revived = True
                        break
                revival_cases += int(revived)
            correct += int(case_probabilities[-1].argmax() == label)
            count += 1
    return {
        "final_top1": correct / count,
        "mean_normalized_path_length": float(np.mean(path_lengths)),
        "mean_final_state_velocity": float(np.mean(final_velocities)),
        "mean_entropy_change": float(np.mean(entropy_changes)),
        "mean_positive_token_delta_true_probability": float(np.mean(positive_deltas)),
        "mean_negative_token_delta_true_probability": float(np.mean(negative_deltas)),
        "negative_contradiction_drop_case_fraction": negative_drop_cases / count,
        "revival_given_negative_drop_fraction": revival_cases / max(1, negative_drop_cases),
    }


@torch.no_grad()
def counterfactual_diagnostics(
    model: ComplexOperatorState,
    pairs: list,
    device: torch.device,
) -> dict[str, float]:
    final_distances: list[float] = []
    maximum_distances: list[float] = []
    correct_pairs = 0
    for pair in pairs:
        raw_batch = next(iter(make_loader([pair.first, pair.second], 2, False, 0)))
        batch = to_device(raw_batch, device)
        trajectory = model.trajectory(batch["tokens"], batch["mask"], batch["vector"])
        probabilities = _probabilities(model, trajectory)
        distance = torch.linalg.vector_norm(trajectory[0] - trajectory[1], dim=-1) / math.sqrt(
            model.state_dim
        )
        final_distances.append(float(distance[-1].cpu()))
        maximum_distances.append(float(distance.max().cpu()))
        prediction = probabilities[:, -1].argmax(dim=-1)
        correct_pairs += int(bool(prediction.eq(batch["label"]).all()))
    return {
        "counterfactual_pair_accuracy": correct_pairs / len(pairs),
        "mean_counterfactual_final_state_distance": float(np.mean(final_distances)),
        "mean_counterfactual_maximum_state_distance": float(np.mean(maximum_distances)),
    }


def _token_label(token: int) -> str:
    sign = "+" if token < NeuroWorld.num_findings else "−"
    return f"{sign}{NeuroWorld.finding_names[token % NeuroWorld.num_findings]}"


@torch.no_grad()
def selected_case_artifact(
    model: ComplexOperatorState,
    case,
    device: torch.device,
) -> dict[str, Any]:
    raw_batch = next(iter(make_loader([case], 1, False, 0)))
    batch = to_device(raw_batch, device)
    length = int(batch["mask"].sum())
    trajectory = model.trajectory(batch["tokens"], batch["mask"], batch["vector"])[
        0, : length + 1
    ]
    probabilities = _probabilities(model, trajectory[None])[0]
    readout = torch.complex(model.readout_real, model.readout_imag)
    amplitudes = torch.einsum("ts,ds->td", trajectory, readout.conj())
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=-1)
    velocity = torch.cat(
        [
            torch.zeros(1, device=device),
            torch.linalg.vector_norm(trajectory[1:] - trajectory[:-1], dim=-1)
            / math.sqrt(model.state_dim),
        ]
    )
    tokens = batch["tokens"][0, :length].cpu().tolist()
    return {
        "case_id": int(case.case_id),
        "label": int(case.label),
        "diagnosis_name": NeuroWorld.disease_names[case.label],
        "token_ids": tokens,
        "token_labels": [_token_label(value) for value in tokens],
        "probabilities": probabilities.cpu().tolist(),
        "amplitude_real": amplitudes.real.cpu().tolist(),
        "amplitude_imag": amplitudes.imag.cpu().tolist(),
        "entropy": entropy.cpu().tolist(),
        "velocity": velocity.cpu().tolist(),
    }


@torch.no_grad()
def selected_pair_artifact(
    model: ComplexOperatorState,
    pair,
    device: torch.device,
) -> dict[str, Any]:
    raw_batch = next(iter(make_loader([pair.first, pair.second], 2, False, 0)))
    batch = to_device(raw_batch, device)
    length = int(batch["mask"][0].sum())
    trajectory = model.trajectory(batch["tokens"], batch["mask"], batch["vector"])[
        :, : length + 1
    ]
    probabilities = _probabilities(model, trajectory)
    return {
        "labels": batch["label"].cpu().tolist(),
        "diagnosis_names": [NeuroWorld.disease_names[int(value)] for value in batch["label"]],
        "first_token_labels": [
            _token_label(value) for value in batch["tokens"][0, :length].cpu().tolist()
        ],
        "second_token_labels": [
            _token_label(value) for value in batch["tokens"][1, :length].cpu().tolist()
        ],
        "first_probabilities": probabilities[0].cpu().tolist(),
        "second_probabilities": probabilities[1].cpu().tolist(),
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
        source_directory = ROOT / "experiments" / "results" / config["source_experiment"]
        source_config = yaml.safe_load((source_directory / "config.yaml").read_text(encoding="utf-8"))
        world = build_world(source_config["train_world"])
        dataset = config["dataset"]
        cases = world.generate(int(dataset["cases"]), seed=int(dataset["case_seed"]))
        pairs = world.counterfactual_pairs(int(dataset["counterfactual_pairs"]), seed=int(dataset["pair_seed"]))
        selected_case = next(case for case in cases if case.label >= 8)
        selected_pair = pairs[0]
        device = torch.device(source_config["training"]["device"])
        seed_records: list[dict[str, float]] = []
        runs: list[dict[str, Any]] = []
        visual_artifact: dict[str, Any] | None = None
        for seed_value in config["seeds"]:
            seed = int(seed_value)
            model = load_source_model(source_directory, source_config, seed, device)
            metrics = aggregate_trajectories(
                model, cases, int(dataset["batch_size"]), device
            )
            metrics.update(counterfactual_diagnostics(model, pairs, device))
            seed_records.append(metrics)
            runs.append({"seed": seed, "metrics": metrics})
            if visual_artifact is None:
                visual_artifact = {
                    "seed": seed,
                    "case": selected_case_artifact(model, selected_case, device),
                    "counterfactual_pair": selected_pair_artifact(model, selected_pair, device),
                }
            print(
                f"{experiment_id} seed={seed} top1={metrics['final_top1']:.3f} "
                f"path={metrics['mean_normalized_path_length']:.3f} "
                f"drops={metrics['negative_contradiction_drop_case_fraction']:.3f} "
                f"revival={metrics['revival_given_negative_drop_fraction']:.3f}",
                flush=True,
            )
        if visual_artifact is None:
            raise RuntimeError("no trajectory artifact generated")
        trajectory_path = result_directory / "selected_trajectories.json"
        trajectory_path.write_text(
            json.dumps(visual_artifact, indent=2, sort_keys=True), encoding="utf-8"
        )
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "source_experiment": config["source_experiment"],
            "selection_rule": "first generated factorial case and first generated counterfactual pair",
            "summary": aggregate_seed_metrics(seed_records),
            "runs": runs,
            "visual_artifact": str(trajectory_path),
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            {"trajectory_summary": results["summary"]},
            [
                ("configuration", result_directory / "config.yaml"),
                ("environment", result_directory / "environment.json"),
                ("metrics", metrics_path),
                ("trajectory", trajectory_path),
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
        default=ROOT / "experiments" / "configs" / "trajectory_study.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_id, result_directory, _ = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))


if __name__ == "__main__":
    main()
