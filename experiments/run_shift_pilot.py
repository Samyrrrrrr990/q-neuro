"""Run the non-confirmatory ShiftGauntlet variance and power pilot."""

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

from experiments.run_experiment_zero import ROOT, evaluate, make_loader
from experiments.run_generator_shift import build_world, train_with_validation_tuning
from neuroworld import ShiftGauntlet, ShiftSpec
from qneuro.provenance import environment_record, file_sha256, require_clean_worktree
from qneuro.registry import ExperimentRegistry
from research.statistics import paired_summary, select_world_count, trapezoidal_robustness_auc


def variant_specs(config: dict[str, Any]) -> list[tuple[str, str | None, str]]:
    variants: list[tuple[str, str | None, str]] = []
    for family, family_config in config["shift_families"].items():
        modes = family_config.get("modes", [None])
        training_profile = str(family_config.get("training_profile", "base"))
        variants.extend((family, mode, training_profile) for mode in modes)
    return variants


def _shift_seed(base: int, family: str, mode: str | None, severity: float, world: int) -> int:
    family_index = sorted(ShiftGauntlet.families).index(family)
    mode_code = sum((index + 1) * ord(character) for index, character in enumerate(mode or ""))
    return int(base + 100_000 * family_index + 101 * mode_code + 17 * world + 10 * severity)


def _training_profile(
    cases: list,
    *,
    profile: str,
    seed: int,
    gauntlet: ShiftGauntlet,
) -> list:
    if profile == "base":
        return cases
    family = {
        "spurious": "spurious_correlation_inversion",
        "class_expansion": "class_expansion",
    }.get(profile)
    if family is None:
        raise ValueError(f"unknown training profile: {profile}")
    return list(
        gauntlet.apply(
            cases,
            ShiftSpec(family, 1.0, seed, split="train"),  # type: ignore[arg-type]
        ).cases
    )


def _metric_subset(metrics: dict[str, float]) -> dict[str, float]:
    keys = ("top1", "nll", "ece", "brier", "order_accuracy", "non_order_accuracy")
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _curves_and_effects(
    records: list[dict[str, Any]], model_names: list[str], severities: list[float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], dict[float, dict[str, float]]] = defaultdict(dict)
    for record in records:
        key = (
            record["train_size"],
            record["training_seed"],
            record["variant"],
            record["world_seed"],
            record["model"],
        )
        grouped[key][float(record["severity"])] = record["metrics"]
    curves: list[dict[str, Any]] = []
    curve_index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, severity_metrics in grouped.items():
        if sorted(severity_metrics) != severities:
            raise RuntimeError(f"incomplete robustness curve: {key}")
        train_size, training_seed, variant, world_seed, model = key
        curve = {
            "train_size": train_size,
            "training_seed": training_seed,
            "variant": variant,
            "world_seed": world_seed,
            "model": model,
            "top1_aurc": trapezoidal_robustness_auc(
                severities, [severity_metrics[value]["top1"] for value in severities]
            ),
            "nll_aurc": trapezoidal_robustness_auc(
                severities, [severity_metrics[value]["nll"] for value in severities]
            ),
            "ece_aurc": trapezoidal_robustness_auc(
                severities, [severity_metrics[value]["ece"] for value in severities]
            ),
        }
        curves.append(curve)
        curve_index[key] = curve
    effects: list[dict[str, Any]] = []
    real_models = [name for name in model_names if name != "complex_operator"]
    comparison_cells = {key[:-1] for key in curve_index if key[-1] == "complex_operator"}
    for cell in sorted(comparison_cells):
        complex_curve = curve_index[(*cell, "complex_operator")]
        available_real = [curve_index[(*cell, name)] for name in real_models]
        best_real = max(available_real, key=lambda item: (item["top1_aurc"], item["model"]))
        effects.append(
            {
                "train_size": cell[0],
                "training_seed": cell[1],
                "variant": cell[2],
                "world_seed": cell[3],
                "complex_top1_aurc": complex_curve["top1_aurc"],
                "best_real_model": best_real["model"],
                "best_real_top1_aurc": best_real["top1_aurc"],
                "difference": complex_curve["top1_aurc"] - best_real["top1_aurc"],
            }
        )
    return curves, effects


def _power_plan(effects: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    world_means: dict[tuple[int, int], list[float]] = defaultdict(list)
    for effect in effects:
        world_means[(int(effect["train_size"]), int(effect["world_seed"]))].append(
            float(effect["difference"])
        )
    by_size: dict[str, Any] = {}
    standard_deviations: list[float] = []
    for train_size in sorted({key[0] for key in world_means}):
        values = [
            float(np.mean(differences))
            for (size, _), differences in sorted(world_means.items())
            if size == train_size
        ]
        summary = paired_summary(values)
        by_size[str(train_size)] = summary
        standard_deviations.append(float(summary["standard_deviation"]))
    conservative_sd = max(standard_deviations)
    power_config = config["power"]
    plan = select_world_count(
        standard_deviation=max(conservative_sd, 1e-9),
        minimum_effect=float(power_config["minimum_effect"]),
        candidates=[int(value) for value in power_config["candidate_world_counts"]],
        target_power=float(power_config["target_power"]),
        simulations=int(power_config["simulations"]),
        seed=int(power_config["seed"]),
    )
    return {
        "world_effects_by_training_size": by_size,
        "conservative_standard_deviation": conservative_sd,
        "selection": plan,
    }


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(config)
    output["description"] += " [SMOKE PROFILE; NOT THE FROZEN PILOT]"
    output["profile"] = "smoke"
    output["models"]["names"] = ["complex_operator", "exact_real_block_operator", "gru"]
    output["models"]["learning_rates"] = {
        name: [rates[0]]
        for name, rates in output["models"]["learning_rates"].items()
        if name in output["models"]["names"]
    }
    output["dataset"].update(
        train_sizes=[120], validation_cases=80, test_cases_per_world=50, world_seeds=[5101, 5107]
    )
    output["training"].update(seeds=[1103], epochs=2, patience=2)
    output["power"].update(simulations=1000)
    return output


def run(config: dict[str, Any], *, allow_dirty: bool = False) -> tuple[str, Path, dict[str, Any]]:
    require_clean_worktree(ROOT, allow_dirty=allow_dirty)
    command = [sys.executable, "-m", "experiments.run_shift_pilot"]
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
        "H0",
        "No intrinsic complex advantage remains after rigorous resource matching.",
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
        gauntlet = ShiftGauntlet()
        source_world = build_world(config["train_world"])
        maximum_train_size = max(int(value) for value in config["dataset"]["train_sizes"])
        source_pool = source_world.generate(
            maximum_train_size, int(config["dataset"]["train_seed"])
        )
        validation_pool = source_world.generate(
            int(config["dataset"]["validation_cases"]), int(config["dataset"]["validation_seed"])
        )
        device = torch.device(config["training"]["device"])
        severities = sorted(float(value) for value in config["severities"])
        models: dict[tuple[int, int, str, str], torch.nn.Module] = {}
        training_runs: list[dict[str, Any]] = []
        profiles = sorted({profile for _, _, profile in variant_specs(config)})
        for train_size_value in config["dataset"]["train_sizes"]:
            train_size = int(train_size_value)
            for seed_value in config["training"]["seeds"]:
                training_seed = int(seed_value)
                for profile_index, profile in enumerate(profiles):
                    profile_seed = int(config["shift_seed"] + 1000 * profile_index + training_seed)
                    train_cases = _training_profile(
                        source_pool[:train_size],
                        profile=profile,
                        seed=profile_seed,
                        gauntlet=gauntlet,
                    )
                    validation_cases = _training_profile(
                        validation_pool,
                        profile=profile,
                        seed=profile_seed + 1,
                        gauntlet=gauntlet,
                    )
                    for model_name in config["models"]["names"]:
                        model, selected, trials = train_with_validation_tuning(
                            model_name,
                            training_seed,
                            train_cases,
                            validation_cases,
                            config,
                            device,
                        )
                        models[(train_size, training_seed, profile, model_name)] = model
                        training_runs.append(
                            {
                                "train_size": train_size,
                                "training_seed": training_seed,
                                "training_profile": profile,
                                "model": model_name,
                                "selected_trial": selected,
                                "tuning_trials": trials,
                            }
                        )
                        print(
                            f"{experiment_id} trained n={train_size} seed={training_seed} "
                            f"profile={profile} model={model_name}",
                            flush=True,
                        )

        records: list[dict[str, Any]] = []
        for world_seed_value in config["dataset"]["world_seeds"]:
            world_seed = int(world_seed_value)
            evaluation_world_config = dict(config["evaluation_world"])
            evaluation_world_config["world_seed"] = world_seed
            world = build_world(evaluation_world_config)
            cases = world.generate(
                int(config["dataset"]["test_cases_per_world"]),
                int(config["dataset"]["test_seed"]),
            )
            for family, mode, profile in variant_specs(config):
                variant = f"{family}:{mode or 'default'}"
                for severity in severities:
                    spec = ShiftSpec(
                        family,  # type: ignore[arg-type]
                        severity,
                        _shift_seed(int(config["shift_seed"]), family, mode, severity, world_seed),
                        mode=mode,
                        split="test",
                    )
                    shifted_cases = list(gauntlet.apply(cases, spec).cases)
                    for train_size_value in config["dataset"]["train_sizes"]:
                        train_size = int(train_size_value)
                        for training_seed_value in config["training"]["seeds"]:
                            training_seed = int(training_seed_value)
                            loader = make_loader(
                                shifted_cases,
                                int(config["training"]["batch_size"]),
                                False,
                                training_seed,
                            )
                            for model_name in config["models"]["names"]:
                                model = models[(train_size, training_seed, profile, model_name)]
                                metrics = evaluate(
                                    model,
                                    loader,
                                    device,
                                    n_bins=int(config["evaluation"]["calibration_bins"]),
                                    seed=training_seed,
                                )
                                records.append(
                                    {
                                        "train_size": train_size,
                                        "training_seed": training_seed,
                                        "training_profile": profile,
                                        "variant": variant,
                                        "world_seed": world_seed,
                                        "severity": severity,
                                        "model": model_name,
                                        "metrics": _metric_subset(metrics),
                                    }
                                )
            print(f"{experiment_id} evaluated world={world_seed}", flush=True)

        curves, effects = _curves_and_effects(records, list(config["models"]["names"]), severities)
        power_plan = _power_plan(effects, config)
        result = {
            "experiment_id": experiment_id,
            "status": "complete",
            "profile": config.get("profile", "pilot"),
            "outcome_eligible": False,
            "hierarchical_unit": "independently_generated_evaluation_world",
            "performance_claim_permitted": False,
            "training_runs": training_runs,
            "records": records,
            "robustness_curves": curves,
            "paired_effects": effects,
            "power_plan": power_plan,
            "scientific_interpretation": (
                "Variance-planning output only. Architecture effects are visible for audit but "
                "are not confirmatory and must not be reported as final evidence."
            ),
        }
        metrics_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        registry.complete(
            experiment_id,
            {
                "power_planning": {
                    "selected_worlds": {
                        "mean": float(power_plan["selection"]["selected_worlds"]),
                        "std": 0.0,
                    },
                    "conservative_standard_deviation": {
                        "mean": float(power_plan["conservative_standard_deviation"]),
                        "std": 0.0,
                    },
                }
            },
            [
                ("configuration", config_path),
                ("environment", environment_path),
                ("metrics", metrics_path),
            ],
        )
        for model in models.values():
            del model
        gc.collect()
        return experiment_id, result_directory, result
    except Exception:
        registry.fail(experiment_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "shift_pilot.yaml",
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
                "selected_worlds": result["power_plan"]["selection"]["selected_worlds"],
                "target_reached": result["power_plan"]["selection"]["target_reached"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
