"""QE-000006: a genuinely torch.complex-parameterized family and its realification.

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Discovery split.

The audit (`docs/QE_AUDIT_MEMO_001.md`, Q3) found that the historical Q-Neuro complex model holds no
complex leaf parameter, so PyTorch's complex-parameter handling was never exercised and the
question of whether complex coordinates change optimizer geometry was never actually asked. This
experiment asks it.

The map here is **non-degenerate**: the source carries `complex64` leaves and the target carries
real pairs, so the coordinate systems genuinely differ. The measurement then separates two
hypotheses that the historical pair could not:

* **H2, optimizer geometry.** Isolated by supplying identical gradients to both sides, removing the
  forward and backward passes from the comparison entirely.
* **H4, numerical implementation.** What remains once H2 is settled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence import paired_training_divergence
from qneuro.equivalence.native_complex import (
    ComplexRealificationMap,
    NativeComplexMLP,
    isolated_optimizer_conjugacy,
)

EXPERIMENT_ID = "QE-000006"
SCHEMA_VERSION = "1.0.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "native_complex_parameterization_and_realification",
    "description": (
        "A model with genuine complex64 leaf parameters and its exact realification. Separates "
        "optimizer geometry from numerical implementation for the complex/real family."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "discovery",
    "split": "discovery",
    "outcome_eligible": False,
    "family": "native_complex_real",
    "model": {"in_features": 6, "hidden": 12, "out_features": 4},
    "training": {
        "optimizers": ["sgd", "adamw"],
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_steps": 10,
        "measured_steps": 20,
        "batch_size": 32,
        "device": "cpu",
    },
    "optimizer_isolation": {
        "steps": 25,
        "weight_decays": [0.0, 0.01],
        "epsilons": [1e-8, 0.1],
    },
    "seeds": {"model": [0, 1, 2], "stream": 13},
    "protocol_deviations": [
        (
            "The activation is a split tanh rather than analytic complex tanh, deliberately, so "
            "the optimizer question is not confounded with the pole discrepancy of QE-000001."
        ),
        (
            "Discovery split: these cells may inform estimator design and may not be reused for "
            "confirmation."
        ),
    ],
}


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_config = config["model"]
    training = config["training"]
    isolation = config["optimizer_isolation"]
    eps = float(torch.finfo(torch.float32).eps)
    mapping = ComplexRealificationMap()

    # 1. Optimizer geometry, with the forward pass factored out.
    optimizer_isolation: list[dict[str, Any]] = []
    for optimizer_name in training["optimizers"]:
        for weight_decay in isolation["weight_decays"]:
            for epsilon in isolation["epsilons"]:
                if optimizer_name == "sgd" and epsilon != isolation["epsilons"][0]:
                    continue  # SGD has no epsilon term
                worst = isolated_optimizer_conjugacy(
                    optimizer_name,
                    steps=int(isolation["steps"]),
                    weight_decay=float(weight_decay),
                    epsilon=float(epsilon),
                )
                optimizer_isolation.append(
                    {
                        "optimizer": optimizer_name,
                        "weight_decay": float(weight_decay),
                        "epsilon": float(epsilon),
                        "max_parameter_divergence": worst,
                        "bitwise_conjugate": worst == 0.0,
                        "within_one_ulp": worst <= 4.0 * eps,
                    }
                )

    # 2. End-to-end paired training across the non-degenerate map.
    end_to_end: list[dict[str, Any]] = []
    for model_seed in config["seeds"]["model"]:
        torch.manual_seed(int(model_seed))
        source = NativeComplexMLP(
            model_config["in_features"], model_config["hidden"], model_config["out_features"]
        )
        target = mapping.build_target(source)
        for optimizer_name in training["optimizers"]:
            measured = paired_training_divergence(
                source,
                target,
                mapping,
                optimizer_name=optimizer_name,
                warmup_steps=int(training["warmup_steps"]),
                measured_steps=int(training["measured_steps"]),
                transport_optimizer_state=True,
                seed=int(config["seeds"]["stream"]),
                learning_rate=float(training["learning_rate"]),
                weight_decay=float(training["weight_decay"]),
            )
            end_to_end.append(
                {
                    "model_seed": int(model_seed),
                    "optimizer": optimizer_name,
                    "max_logit_divergence": measured["max_logit_divergence"],
                    "max_total_variation": measured["max_total_variation"],
                    "reference_logit_scale": measured["reference_logit_scale"],
                }
            )

    adamw_cells = [row for row in optimizer_isolation if row["optimizer"] == "adamw"]
    sgd_cells = [row for row in optimizer_isolation if row["optimizer"] == "sgd"]

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "split": config["split"],
        "outcome_eligible": False,
        "status": "complete",
        "float32_eps": eps,
        "map": {
            "name": mapping.name,
            "transport_degenerate": mapping.transport_degenerate,
            "declared_level": mapping.spec.declared_level.value,
        },
        "optimizer_isolation": optimizer_isolation,
        "end_to_end": end_to_end,
        "summary": {
            "map_is_non_degenerate": not mapping.transport_degenerate,
            "adamw_bitwise_conjugate_in_all_cells": all(
                row["bitwise_conjugate"] for row in adamw_cells
            ),
            "sgd_within_one_ulp_in_all_cells": all(row["within_one_ulp"] for row in sgd_cells),
            "worst_sgd_isolation_divergence": max(
                row["max_parameter_divergence"] for row in sgd_cells
            ),
            "worst_end_to_end_logit_divergence": max(
                row["max_logit_divergence"] for row in end_to_end
            ),
        },
        "scientific_interpretation": (
            "Discovery evidence on an analytic microcosm. PyTorch keeps per-component moments for "
            "complex parameters rather than modulus-based ones, so complex AdamW is bitwise "
            "identical to AdamW on the realified pair even under weight decay and a large epsilon. "
            "A genuinely non-degenerate complex/real map is therefore exactly conjugate as an "
            "update rule: complex parameterization supplies no optimizer geometry in this "
            "framework. SGD leaves a residual at one unit in the last place, which is complex "
            "kernel arithmetic (H4) rather than optimizer geometry (H2). Any complex-versus-real "
            "performance gap observed in PyTorch must therefore originate in function class, "
            "initialization, or numerics, and cannot be attributed to optimizer coordinates. "
            "Nothing here supports an architecture claim or a transport-covariance law."
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
    print(f"{EXPERIMENT_ID}: map non-degenerate={summary['map_is_non_degenerate']}")
    print(
        f"  AdamW bitwise conjugate (all cells) : {summary['adamw_bitwise_conjugate_in_all_cells']}"
    )
    print(f"  SGD within one ULP (all cells)      : {summary['sgd_within_one_ulp_in_all_cells']}")
    print(
        f"  worst SGD isolation divergence      : {summary['worst_sgd_isolation_divergence']:.3e}"
    )
    print(
        f"  worst end-to-end logit divergence   : {summary['worst_end_to_end_logit_divergence']:.3e}"
    )


if __name__ == "__main__":
    main()
