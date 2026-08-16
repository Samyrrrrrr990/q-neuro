"""QE-000003: homogeneous scaling-orbit transport ladder.

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Discovery split.

This is the first family in the program with a genuinely non-zero covariance defect. The map is
exact, so every difference measured here is a property of the *training system*, not of the
represented function. The experiment climbs the transport ladder one rung at a time and attributes
the residual defect to a named component.

Derivations under test (see `qneuro/equivalence/scaling.py`):

* plain SGD is conjugate iff ``eta -> eta * s^2``;
* Adam needs ``eta -> eta * s`` and stays inexact through its ``eps`` term;
* weight decay and the gradient step cannot both be covariant under any single learning-rate
  policy, so a residual defect must survive full learning-rate transport whenever decay is on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence import HomogeneousScalingMap, paired_training_divergence
from qneuro.equivalence.microcosms import HomogeneousMLP

EXPERIMENT_ID = "QE-000003"
SCHEMA_VERSION = "1.0.0"

RUNGS = (
    ("T2", False, False, "mapped initialization only"),
    ("T4", True, False, "mapped initialization and optimizer state"),
    ("T4+", True, True, "mapped initialization, optimizer state, and transported learning rate"),
)

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "homogeneous_scaling_orbit_transport_ladder",
    "description": (
        "Exact continuous symmetry of a positively homogeneous network. Measures the covariance "
        "defect at each transport rung and attributes the residual to the gradient step, the "
        "optimizer state, the learning-rate policy, and weight decay."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "discovery",
    "split": "discovery",
    "outcome_eligible": False,
    "family": "scaling_orbit",
    "model": {"in_features": 6, "hidden": 12, "out_features": 4},
    "training": {
        "optimizers": ["sgd", "adamw"],
        "learning_rate": 0.001,
        "weight_decays": [0.0, 0.01],
        "warmup_steps": 10,
        "measured_steps": 20,
        "batch_size": 32,
        "device": "cpu",
    },
    "scales": [2.0, 3.0],
    "seeds": {"model": [0, 1, 2], "stream": 5},
    "protocol_deviations": [
        "Analytic microcosm only; no task generator and no architecture claim.",
        (
            "Discovery split: these cells may inform estimator design and may not be reused "
            "for confirmation."
        ),
    ],
}


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_config = config["model"]
    training = config["training"]
    eps = float(torch.finfo(torch.float32).eps)

    records: list[dict[str, Any]] = []
    for scale in config["scales"]:
        mapping = HomogeneousScalingMap(float(scale))
        for model_seed in config["seeds"]["model"]:
            torch.manual_seed(int(model_seed))
            source = HomogeneousMLP(
                model_config["in_features"], model_config["hidden"], model_config["out_features"]
            )
            target = mapping.build_target(source)
            for optimizer_name in training["optimizers"]:
                for weight_decay in training["weight_decays"]:
                    for rung, state, learning_rate_transport, description in RUNGS:
                        measured = paired_training_divergence(
                            source,
                            target,
                            mapping,
                            optimizer_name=optimizer_name,
                            warmup_steps=int(training["warmup_steps"]),
                            measured_steps=int(training["measured_steps"]),
                            transport_optimizer_state=state,
                            transport_learning_rate=learning_rate_transport,
                            seed=int(config["seeds"]["stream"]),
                            learning_rate=float(training["learning_rate"]),
                            weight_decay=float(weight_decay),
                        )
                        records.append(
                            {
                                "scale": float(scale),
                                "scale_is_power_of_two": mapping.is_bitwise_exact,
                                "model_seed": int(model_seed),
                                "optimizer": optimizer_name,
                                "weight_decay": float(weight_decay),
                                "rung": rung,
                                "rung_description": description,
                                "max_logit_divergence": measured["max_logit_divergence"],
                                "max_total_variation": measured["max_total_variation"],
                                "at_rounding_floor": bool(
                                    measured["max_logit_divergence"]
                                    <= 10.0 * eps * max(measured["reference_logit_scale"], 1.0)
                                ),
                            }
                        )

    def cells(**criteria: Any) -> list[dict[str, Any]]:
        return [
            record
            for record in records
            if all(record[key] == value for key, value in criteria.items())
        ]

    def worst(**criteria: Any) -> float:
        selected = cells(**criteria)
        return max(record["max_logit_divergence"] for record in selected) if selected else 0.0

    # Attribution: the only difference between these pairs of cells is the named component.
    attribution = {
        "sgd_gradient_step_conjugate_under_eta_s_squared": {
            "worst_defect": worst(optimizer="sgd", weight_decay=0.0, rung="T4+", scale=2.0),
            "expectation": "exactly zero for a power-of-two scale",
        },
        "sgd_weight_decay_residual": {
            "worst_defect": worst(optimizer="sgd", weight_decay=0.01, rung="T4+", scale=2.0),
            "expectation": "non-zero; no learning-rate policy transports decay and gradient jointly",
        },
        "adamw_gradient_step_conjugate_under_eta_s": {
            "worst_defect": worst(optimizer="adamw", weight_decay=0.0, rung="T4+", scale=2.0),
            "expectation": "rounding floor; inexact only through Adam's epsilon",
        },
        "adamw_decoupled_decay_residual": {
            "worst_defect": worst(optimizer="adamw", weight_decay=0.01, rung="T4+", scale=2.0),
            "expectation": "non-zero and larger than the weight_decay=0 cell",
        },
    }

    partial_transport_penalties = [
        record
        for record in records
        if record["rung"] == "T4"
        and record["max_logit_divergence"]
        > worst(
            scale=record["scale"],
            model_seed=record["model_seed"],
            optimizer=record["optimizer"],
            weight_decay=record["weight_decay"],
            rung="T2",
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "split": config["split"],
        "outcome_eligible": False,
        "status": "complete",
        "float32_eps": eps,
        "records": records,
        "attribution": attribution,
        "summary": {
            "cells": len(records),
            "worst_defect_at_T2": worst(rung="T2"),
            "worst_defect_at_T4": worst(rung="T4"),
            "worst_defect_at_T4_plus": worst(rung="T4+"),
            "non_degenerate_map": True,
            "partial_transport_can_increase_defect": bool(partial_transport_penalties),
            "partial_transport_penalty_cells": len(partial_transport_penalties),
        },
        "scientific_interpretation": (
            "Discovery evidence on an analytic microcosm. The map is exact, so every measured "
            "difference belongs to the training system rather than to the represented function. "
            "These cells may inform estimator design and may not be reused for confirmation. "
            "Nothing here supports any architecture claim or any transport-covariance law."
        ),
        "protocol_deviations": config["protocol_deviations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results")
    args = parser.parse_args()

    result = run(DEFAULT_CONFIG)
    directory = args.output / EXPERIMENT_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8"
    )
    (directory / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = result["summary"]
    print(f"{EXPERIMENT_ID}: {summary['cells']} cells")
    for rung in ("T2", "T4", "T4+"):
        print(
            f"  worst defect at {rung:<3}: {summary[f'worst_defect_at_{rung.replace("+", "_plus")}']:.3e}"
        )
    print(
        f"  partial transport increased the defect in {summary['partial_transport_penalty_cells']} cells"
    )


if __name__ == "__main__":
    main()
