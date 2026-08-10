"""Benchmark conventional, attractor, Hamiltonian, and density computational laws."""

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
    to_device,
)
from experiments.run_generator_shift import build_world, train_with_validation_tuning
from experiments.run_robustness_sweep import aggregate_world_hierarchy, paired_world_effects
from neuroworld import NeuroWorld, ambiguous_order_pairs
from qneuro.evaluation import ambiguity_pair_metrics
from qneuro.metrics import aggregate_seed_metrics, classification_metrics
from qneuro.models import (
    ComplexEvidenceAccumulator,
    ComplexEvidenceMLP,
    ComplexOperatorState,
    DiagnosticDensityDynamics,
    EnergyAttractorState,
    HamiltonianDissipativeState,
)
from qneuro.registry import ExperimentRegistry


@torch.no_grad()
def mechanism_diagnostics(
    model: torch.nn.Module, cases: list, batch_size: int, device: torch.device
) -> dict[str, float]:
    records: dict[str, list[torch.Tensor]] = defaultdict(list)
    for raw_batch in make_loader(cases, batch_size, False, 0):
        batch = to_device(raw_batch, device)
        if isinstance(model, EnergyAttractorState):
            diagnostics = model.trajectory_diagnostics(
                batch["tokens"], batch["mask"], batch["vector"]
            )
            records["trajectory_initial_entropy"].append(diagnostics["entropy"][:, 0].cpu())
            records["trajectory_final_entropy"].append(diagnostics["entropy"][:, -1].cpu())
            records["trajectory_final_velocity"].append(diagnostics["velocity"][:, -1].cpu())
            if model.adaptive:
                model(**batch)
                if model._last_expected_steps is not None:
                    records["expected_diagnostic_steps"].append(
                        model._last_expected_steps.detach().cpu()
                    )
        elif isinstance(model, DiagnosticDensityDynamics):
            diagnostics = model.density_diagnostics(batch["tokens"], batch["mask"], batch["vector"])
            for key, value in diagnostics.items():
                records[f"density_{key}"].append(value.detach().real.cpu())
        elif isinstance(model, HamiltonianDissipativeState):
            state = model.evolve(batch["tokens"], batch["mask"], batch["vector"])
            records["state_norm"].append(torch.linalg.vector_norm(state, dim=-1).cpu())
            unit_phase = state / torch.abs(state).clamp_min(1e-8)
            records["phase_coherence"].append(torch.abs(unit_phase.mean(dim=-1)).cpu())
    return {
        key: float(torch.cat(values).float().mean()) for key, values in records.items() if values
    }


def output_metrics(outputs: dict[str, torch.Tensor], bins: int) -> dict[str, float]:
    return classification_metrics(
        outputs["logits"],
        outputs["labels"],
        outputs["is_order"],
        outputs["order_complete"],
        n_bins=bins,
    )


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
        world_seeds = [int(value) for value in dataset_config["unseen_world_seeds"]]
        device = torch.device(training_config["device"])
        batch_size = int(training_config["batch_size"])

        train_world = build_world(config["train_world"])
        train_cases = train_world.generate(int(dataset_config["train_cases"]), seed=9101)
        validation_cases = train_world.generate(int(dataset_config["validation_cases"]), seed=9102)
        in_domain_cases = train_world.generate(int(dataset_config["test_cases"]), seed=9103)
        ambiguity_pairs = ambiguous_order_pairs(
            train_world, int(dataset_config["ambiguity_pairs"]), seed=9104
        )
        ambiguity_cases = [case for pair in ambiguity_pairs for case in (pair.first, pair.second)]
        in_domain_pairs = train_world.counterfactual_pairs(
            int(dataset_config["counterfactual_pairs"]), seed=9105
        )
        shifted_sets: dict[int, dict[str, Any]] = {}
        for world_seed in world_seeds:
            world = NeuroWorld(world_seed=world_seed, **config["shift"])
            shifted_sets[world_seed] = {
                "cases": world.generate(int(dataset_config["test_cases"]), seed=9103),
                "counterfactuals": world.counterfactual_pairs(
                    int(dataset_config["counterfactual_pairs"]), seed=9105
                ),
            }

        records: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
        in_domain_records: dict[str, list[dict[str, float]]] = defaultdict(list)
        runs: list[dict[str, Any]] = []
        checkpoint_directory = result_directory / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_artifacts: list[tuple[str, Path]] = []
        for seed_value in training_config["seeds"]:
            seed = int(seed_value)
            for model_name in model_names:
                model, selected, tuning_trials = train_with_validation_tuning(
                    model_name, seed, train_cases, validation_cases, config, device
                )
                checkpoint_path = checkpoint_directory / f"{model_name}_seed{seed}.pt"
                torch.save(
                    {
                        "model_state_dict": {
                            key: value.detach().cpu() for key, value in model.state_dict().items()
                        },
                        "model_metadata": selected["model_metadata"],
                        "training_seed": seed,
                        "selected_learning_rate": selected["learning_rate"],
                    },
                    checkpoint_path,
                )
                checkpoint_artifacts.append(("checkpoint", checkpoint_path))

                in_loader = make_loader(in_domain_cases, batch_size, False, seed)
                in_outputs = collect_outputs(model, in_loader, device, seed=seed)
                in_metrics = output_metrics(in_outputs, int(evaluation_config["calibration_bins"]))
                in_metrics.update(
                    evaluate_counterfactuals(model, in_domain_pairs, batch_size, device)
                )
                shuffled_outputs = collect_outputs(
                    model, in_loader, device, shuffle_order=True, seed=seed
                )
                shuffled_metrics = output_metrics(
                    shuffled_outputs, int(evaluation_config["calibration_bins"])
                )
                in_metrics["shuffled_top1"] = shuffled_metrics["top1"]
                in_metrics["shuffle_delta"] = in_metrics["top1"] - shuffled_metrics["top1"]
                ambiguity_outputs = collect_outputs(
                    model,
                    make_loader(ambiguity_cases, batch_size, False, seed),
                    device,
                    seed=seed,
                )
                in_metrics.update(
                    ambiguity_pair_metrics(ambiguity_outputs["logits"], ambiguity_outputs["labels"])
                )
                in_metrics.update(mechanism_diagnostics(model, in_domain_cases, batch_size, device))
                if isinstance(
                    model,
                    (ComplexOperatorState, ComplexEvidenceMLP, ComplexEvidenceAccumulator),
                ):
                    for phase_mode in ("zero", "randomized"):
                        ablated = collect_outputs(
                            model, in_loader, device, phase_mode=phase_mode, seed=seed
                        )
                        in_metrics[f"{phase_mode}_phase_top1"] = output_metrics(
                            ablated, int(evaluation_config["calibration_bins"])
                        )["top1"]
                in_metrics.update(
                    {
                        "parameter_count": float(selected["model_metadata"]["parameter_count"]),
                        "training_seconds": float(selected["resources"]["training_seconds"]),
                        "peak_rss_gib": float(selected["resources"]["peak_rss_bytes"] / 1024**3),
                    }
                )
                in_domain_records[model_name].append(in_metrics)

                shifted_metrics: dict[str, dict[str, float]] = {}
                for world_seed, evaluation_set in shifted_sets.items():
                    outputs = collect_outputs(
                        model,
                        make_loader(evaluation_set["cases"], batch_size, False, seed),
                        device,
                        seed=seed,
                    )
                    metrics = output_metrics(outputs, int(evaluation_config["calibration_bins"]))
                    metrics.update(
                        evaluate_counterfactuals(
                            model, evaluation_set["counterfactuals"], batch_size, device
                        )
                    )
                    records[(model_name, "moderate", world_seed)].append(metrics)
                    shifted_metrics[str(world_seed)] = metrics

                runs.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "selected_trial": selected,
                        "tuning_trials": tuning_trials,
                        "in_domain_metrics": in_metrics,
                        "shifted_metrics": shifted_metrics,
                        "checkpoint": str(checkpoint_path),
                    }
                )
                print(
                    f"{experiment_id} seed={seed} model={model_name} "
                    f"id={in_metrics['top1']:.3f} ambiguity={in_metrics['ambiguity_pair_nll']:.3f} "
                    f"shift={np.mean([value['top1'] for value in shifted_metrics.values()]):.3f}",
                    flush=True,
                )
                del model
                gc.collect()

        summary = aggregate_world_hierarchy(records, model_names, ["moderate"], world_seeds)
        for model_name in model_names:
            summary[model_name]["in_domain"] = aggregate_seed_metrics(in_domain_records[model_name])
        effects = paired_world_effects(summary, model_names, ["moderate"])
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "hierarchical_shift_unit": "unseen world seed",
            "summary": summary,
            "paired_world_effects": effects,
            "runs": runs,
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        registry.complete(
            experiment_id,
            {
                f"{model}@moderate": values["moderate"]["across_worlds"]
                for model, values in summary.items()
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
        default=ROOT / "experiments" / "configs" / "dynamics_suite.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run a fast integration profile")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = copy.deepcopy(config)
        config["description"] += " [SMOKE PROFILE]"
        config["dataset"].update(
            train_cases=200,
            validation_cases=100,
            test_cases=120,
            ambiguity_pairs=20,
            counterfactual_pairs=20,
            unseen_world_seeds=config["dataset"]["unseen_world_seeds"][:1],
        )
        config["training"].update(seeds=[11], epochs=2, patience=2)
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    for model, values in results["summary"].items():
        print(
            f"{model:22s} id={values['in_domain']['top1']['mean']:.3f} "
            f"ambiguity={values['in_domain']['ambiguity_pair_nll']['mean']:.3f} "
            f"shift={values['moderate']['across_worlds']['top1']['mean']:.3f}"
        )


if __name__ == "__main__":
    main()
