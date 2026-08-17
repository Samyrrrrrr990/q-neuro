"""Run outcome-ineligible causal mechanism interventions under controlled shifts."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from experiments.run_experiment_zero import ROOT, collect_outputs, evaluate, make_loader
from experiments.run_generator_shift import build_world, train_with_validation_tuning
from neuroworld import ShiftGauntlet, ShiftSpec, ambiguous_order_pairs
from qneuro.evaluation import ambiguity_pair_metrics
from qneuro.models import ComplexOperatorState
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry


def _mechanism_diagnostics(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    commutator = getattr(model, "commutator_norm", None)
    if commutator is not None:
        diagnostics["mean_marker_commutator_norm"] = float(
            np.mean([commutator(2 * pair, 2 * pair + 1) for pair in range(4)])
        )
    evolve = getattr(model, "evolve", None)
    if evolve is not None:
        with torch.no_grad():
            state = evolve(batch["tokens"], batch["mask"], batch.get("vector"))
        if state.is_complex():
            diagnostics["state_mean_magnitude"] = float(torch.abs(state).mean())
            diagnostics["state_phase_resultant"] = float(
                torch.abs(torch.exp(1j * torch.angle(state)).mean(dim=-1)).mean()
            )
    return diagnostics


def _metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key
        in {
            "top1",
            "nll",
            "ece",
            "brier",
            "order_accuracy",
            "non_order_accuracy",
            "complete_order_accuracy",
        }
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        key = (record["model"], record["variant"], float(record["severity"]))
        for metric, value in record["metrics"].items():
            grouped[key][metric].append(float(value))
    summary: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for (model, variant, severity), metrics in grouped.items():
        summary[model][variant][str(severity)] = {
            metric: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
            }
            for metric, values in metrics.items()
        }
    return {model: dict(variants) for model, variants in summary.items()}


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output["description"] += " [SMOKE PROFILE; OUTCOME-INELIGIBLE]"
    output["profile"] = "smoke"
    keep = [
        "complex_operator",
        "commuting_operator",
        "phase_destroyed_training",
        "noncommutative_real_operator",
        "ambiguity_aware_real",
    ]
    output["models"]["names"] = keep
    output["models"]["learning_rates"] = {
        name: output["models"]["learning_rates"][name] for name in keep
    }
    output["dataset"].update(
        train_cases=150,
        validation_cases=80,
        test_cases_per_world=60,
        ambiguity_pairs=20,
        world_seeds=[6101],
    )
    output["training"].update(seeds=[1103], epochs=2, patience=2)
    output["severities"] = [0.0, 1.0]
    return output


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_mechanism_suite"]
    source_environment = environment_record(ROOT, command=command)
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    preregistration_document = str(config["preregistration_document"])
    registry.register_preregistration(
        str(config["preregistration_id"]),
        str(config["preregistration_version"]),
        preregistration_document,
        file_sha256(ROOT / preregistration_document),
    )
    registry.register_hypothesis(
        "H3",
        "Advantage is specific to target-relevant noncommutative evidence composition.",
        status="open",
    )
    experiment_id, result_directory = registry.reserve(config, ROOT / "experiments" / "results")
    registry.attach_protocol(
        experiment_id,
        str(config["preregistration_id"]),
        str(config["hypothesis_id"]),
        command,
    )
    try:
        config_path = result_directory / "config.yaml"
        environment_path = result_directory / "environment.json"
        metrics_path = result_directory / "metrics.json"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        environment_path.write_text(
            json.dumps(source_environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        world = build_world(config["train_world"])
        train_cases = world.generate(int(config["dataset"]["train_cases"]), seed=7101)
        validation_cases = world.generate(int(config["dataset"]["validation_cases"]), seed=7102)
        ambiguity_pairs = ambiguous_order_pairs(
            world, int(config["dataset"]["ambiguity_pairs"]), seed=7103
        )
        ambiguity_cases = [case for pair in ambiguity_pairs for case in (pair.first, pair.second)]
        device = torch.device(config["training"]["device"])
        gauntlet = ShiftGauntlet()
        training_runs: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for training_seed_value in config["training"]["seeds"]:
            training_seed = int(training_seed_value)
            for model_name in config["models"]["names"]:
                model, selected, trials = train_with_validation_tuning(
                    model_name,
                    training_seed,
                    train_cases,
                    validation_cases,
                    config,
                    device,
                )
                diagnostic_batch = next(
                    iter(
                        make_loader(
                            validation_cases, int(config["training"]["batch_size"]), False, 0
                        )
                    )
                )
                diagnostics = _mechanism_diagnostics(model, diagnostic_batch)
                ambiguity_outputs = collect_outputs(
                    model,
                    make_loader(
                        ambiguity_cases,
                        int(config["training"]["batch_size"]),
                        False,
                        training_seed,
                    ),
                    device,
                    seed=training_seed,
                )
                ambiguity = ambiguity_pair_metrics(
                    ambiguity_outputs["logits"], ambiguity_outputs["labels"]
                )
                phase_inference: dict[str, dict[str, float]] = {}
                if isinstance(model, ComplexOperatorState) and model_name == "complex_operator":
                    loader = make_loader(
                        validation_cases,
                        int(config["training"]["batch_size"]),
                        False,
                        training_seed,
                    )
                    for phase_mode in ("zero", "randomized"):
                        phase_inference[phase_mode] = _metric_subset(
                            evaluate(
                                model,
                                loader,
                                device,
                                n_bins=int(config["evaluation"]["calibration_bins"]),
                                phase_mode=phase_mode,
                                seed=training_seed,
                            )
                        )
                training_runs.append(
                    {
                        "training_seed": training_seed,
                        "model": model_name,
                        "selected_trial": selected,
                        "tuning_trials": trials,
                        "diagnostics": diagnostics,
                        "ambiguity_metrics": ambiguity,
                        "phase_destroyed_inference": phase_inference,
                    }
                )
                for world_seed_value in config["dataset"]["world_seeds"]:
                    world_seed = int(world_seed_value)
                    world_config = dict(config["evaluation_world"])
                    world_config["world_seed"] = world_seed
                    evaluation_world = build_world(world_config)
                    source_cases = evaluation_world.generate(
                        int(config["dataset"]["test_cases_per_world"]), seed=7104
                    )
                    for variant, variant_config in config["shift_variants"].items():
                        family = str(variant_config["family"])
                        mode = variant_config.get("mode")
                        variant_code = sum(
                            (index + 1) * ord(character) for index, character in enumerate(variant)
                        )
                        for severity_value in config["severities"]:
                            severity = float(severity_value)
                            shifted = gauntlet.apply(
                                source_cases,
                                ShiftSpec(
                                    family,  # type: ignore[arg-type]
                                    severity,
                                    int(
                                        config["shift_seed"]
                                        + world_seed
                                        + 101 * variant_code
                                        + 100 * severity
                                    ),
                                    mode=mode,
                                ),
                            )
                            metrics = evaluate(
                                model,
                                make_loader(
                                    list(shifted.cases),
                                    int(config["training"]["batch_size"]),
                                    False,
                                    training_seed,
                                ),
                                device,
                                n_bins=int(config["evaluation"]["calibration_bins"]),
                                seed=training_seed,
                            )
                            records.append(
                                {
                                    "training_seed": training_seed,
                                    "model": model_name,
                                    "world_seed": world_seed,
                                    "variant": variant,
                                    "severity": severity,
                                    "metrics": _metric_subset(metrics),
                                }
                            )
                print(
                    f"{experiment_id} seed={training_seed} model={model_name} "
                    f"ambiguity_nll={ambiguity['ambiguity_pair_nll']:.3f}",
                    flush=True,
                )
                del model
                gc.collect()
        summary = aggregate_records(records)
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "profile": config.get("profile", "mechanism_discovery"),
            "outcome_eligible": False,
            "training_runs": training_runs,
            "records": records,
            "summary": summary,
            "scientific_interpretation": (
                "Mechanism-discovery evidence only. Any selected relationship requires a newly "
                "frozen test on untouched tasks."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry.complete(
            experiment_id,
            {
                f"{model}:{variant}:{severity}": metrics
                for model, variants in summary.items()
                for variant, severities in variants.items()
                for severity, metrics in severities.items()
            },
            [
                ("configuration", config_path),
                ("environment", environment_path),
                ("metrics", metrics_path),
            ],
        )
        return experiment_id, result_directory, result
    except Exception:
        registry.fail(experiment_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "mechanism_suite.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="reserved for smoke debugging")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = smoke_config(config)
    experiment_id, directory, result = run(config, allow_dirty=args.allow_dirty)
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "result_directory": str(directory),
                "profile": result["profile"],
                "training_runs": len(result["training_runs"]),
                "records": len(result["records"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
