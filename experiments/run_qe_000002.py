"""QE-000002: hidden-unit permutation zero-defect positive control.

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Not outcome-eligible: this experiment
validates the measurement instrument, it does not estimate an architecture effect.

The control has two required directions. Under correctly permuted optimizer state the paired
predictive divergence must stay at floating-point rounding scale. Under deliberately un-permuted
state the defect must be detectable. An instrument that cannot fail the second direction cannot be
trusted on the first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence import HiddenUnitPermutationMap, paired_training_divergence
from qneuro.equivalence.microcosms import TwoLayerMLP, fixed_batch

EXPERIMENT_ID = "QE-000002"
SCHEMA_VERSION = "1.0.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "hidden_unit_permutation_zero_defect_control",
    "description": (
        "Exact discrete symmetry control for the equivalence compiler. Establishes that correctly "
        "transported optimizer state yields a rounding-scale defect and that a broken transport is "
        "detected."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "instrument_control",
    "outcome_eligible": False,
    "family": "permutation_symmetry",
    "model": {"in_features": 6, "hidden": 12, "out_features": 4},
    "training": {
        "optimizers": ["sgd", "adamw"],
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "warmup_steps": 15,
        "measured_steps": 25,
        "batch_size": 32,
        "device": "cpu",
    },
    "seeds": {"model": [0, 1, 2], "permutation_offset": 1, "stream": 11},
    "thresholds": {
        "transported_tolerance_eps_multiple": 100.0,
        "minimum_detection_ratio": 1000.0,
        "minimum_broken_divergence": 1e-4,
    },
}


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_config = config["model"]
    training = config["training"]
    thresholds = config["thresholds"]
    eps = float(torch.finfo(torch.float32).eps)

    records: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []

    for model_seed in config["seeds"]["model"]:
        torch.manual_seed(int(model_seed))
        source = TwoLayerMLP(
            model_config["in_features"], model_config["hidden"], model_config["out_features"]
        )
        mapping = HiddenUnitPermutationMap.random(
            source, seed=int(model_seed) + int(config["seeds"]["permutation_offset"])
        )
        target = mapping.build_target(source)

        certificate = mapping.certify(
            source,
            target,
            batch=fixed_batch(
                seed=int(model_seed) + 7,
                features=model_config["in_features"],
                batch_size=training["batch_size"],
                classes=model_config["out_features"],
            ),
        )
        certificates.append({"model_seed": int(model_seed), **certificate.as_dict()})

        for optimizer_name in training["optimizers"]:
            measured = {
                transported: paired_training_divergence(
                    source,
                    target,
                    mapping,
                    optimizer_name=optimizer_name,
                    warmup_steps=int(training["warmup_steps"]),
                    measured_steps=int(training["measured_steps"]),
                    transport_optimizer_state=transported,
                    seed=int(config["seeds"]["stream"]),
                    learning_rate=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                )
                for transported in (True, False)
            }
            transported_divergence = measured[True]["max_logit_divergence"]
            broken_divergence = measured[False]["max_logit_divergence"]
            tolerance = (
                float(thresholds["transported_tolerance_eps_multiple"])
                * eps
                * max(measured[True]["reference_logit_scale"], 1.0)
            )
            ratio = broken_divergence / max(transported_divergence, eps * 1e-3)
            records.append(
                {
                    "model_seed": int(model_seed),
                    "optimizer": optimizer_name,
                    "transported_max_logit_divergence": transported_divergence,
                    "transported_max_total_variation": measured[True]["max_total_variation"],
                    "broken_max_logit_divergence": broken_divergence,
                    "broken_max_total_variation": measured[False]["max_total_variation"],
                    "tolerance": tolerance,
                    "detection_ratio": ratio,
                    "transported_within_tolerance": bool(transported_divergence <= tolerance),
                    "broken_detected": bool(
                        ratio >= float(thresholds["minimum_detection_ratio"])
                        and broken_divergence >= float(thresholds["minimum_broken_divergence"])
                    ),
                }
            )

    positive_control_passes = all(record["transported_within_tolerance"] for record in records)
    negative_control_passes = all(record["broken_detected"] for record in records)
    gate_passes = positive_control_passes and negative_control_passes

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "outcome_eligible": False,
        "status": "complete",
        "float32_eps": eps,
        "records": records,
        "certificates": certificates,
        "summary": {
            "positive_control_passes": positive_control_passes,
            "negative_control_passes": negative_control_passes,
            "gate_passes": gate_passes,
            "worst_transported_divergence": max(
                record["transported_max_logit_divergence"] for record in records
            ),
            "smallest_detection_ratio": min(record["detection_ratio"] for record in records),
            "all_maps_non_degenerate": all(
                not certificate["transport_degenerate"] for certificate in certificates
            ),
        },
        "scientific_interpretation": (
            "Instrument validation only. A passing gate shows that the equivalence framework can "
            "distinguish a correctly transported exact symmetry from a broken one. It establishes "
            "nothing about any architecture, and it cannot support any transport-covariance claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results")
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    result = run(config)

    directory = args.output / EXPERIMENT_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (directory / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = result["summary"]
    print(f"{EXPERIMENT_ID}: gate_passes={summary['gate_passes']}")
    print(f"  worst transported divergence : {summary['worst_transported_divergence']:.3e}")
    print(f"  smallest detection ratio     : {summary['smallest_detection_ratio']:.3e}")
    if not summary["gate_passes"]:
        raise SystemExit("QE-000002 gate failed; the measurement system requires repair.")


if __name__ == "__main__":
    main()
