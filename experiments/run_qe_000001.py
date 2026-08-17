"""QE-000001: complex / exact-real forward certificate and pole-reachability measurement.

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Instrument, not outcome-eligible.

Two obligations, both created by `docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md`:

1. emit a certificate declaring **E2/E3 on a stated domain**, never E0/E1 globally, and carrying
   the `transport_degenerate` flag;
2. measure how close the pre-activation `delta` actually comes to a pole of complex `tanh`, which
   could not be answered retrospectively because the falsification phase saved no activations.

The reachability figure below is a probe, not a settled answer. It measures this configuration on
this data, and a wider sweep is required before the excluded region can be called unreachable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence.complex_real import (
    CRITICAL_POLE_RADIUS,
    ComplexToExactRealMap,
    record_complex_tanh_inputs,
)
from qneuro.models.equivalent import ExactRealBlockOperatorState
from qneuro.models.operators import ComplexOperatorState

EXPERIMENT_ID = "QE-000001"
SCHEMA_VERSION = "1.0.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "complex_exact_real_certificate_and_pole_reachability",
    "description": (
        "Formalizes the historical complex/exact-real mapping as a flagged transport-degenerate "
        "map, certifies it at E2 on a declared domain, and measures how close training drives the "
        "pre-activation to a pole of complex tanh."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "amendment_document": "docs/EQUIVALENCE_SCIENCE_AMENDMENT_001.md",
    "profile": "instrument",
    "outcome_eligible": False,
    "family": "complex_real",
    "model": {
        "num_tokens": 16,
        "pad_token": 15,
        "state_dim": 8,
        "rank": 2,
        "num_classes": 5,
        "step_size": 0.35,
    },
    "training": {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "steps": 60,
        "batch_size": 24,
        "sequence_length": 12,
        "device": "cpu",
    },
    "seeds": {"model": [0, 1, 2], "data": 1234},
    "declared_domain": {
        "excluded": "min_k |delta - i(2k+1)pi/2| <= rho_c",
        "critical_radius": CRITICAL_POLE_RADIUS,
        "measurement": "bisection over 16 approach angles at 1e-3 relative tolerance",
    },
    "protocol_deviations": [
        (
            "Reachability is probed on random token streams, not on the historical NeuroWorld or "
            "independent generators, because the falsification-phase runs saved no activations."
        ),
        (
            "A single configuration is probed; this cannot establish that the excluded region is "
            "globally unreachable."
        ),
    ],
}


def make_batch(config: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    model = config["model"]
    training = config["training"]
    generator = torch.Generator().manual_seed(seed)
    rows, length = int(training["batch_size"]), int(training["sequence_length"])
    mask = torch.ones(rows, length, dtype=torch.bool)
    mask[:, -3:] = torch.rand(rows, 3, generator=generator) > 0.4
    return {
        "tokens": torch.randint(0, model["num_tokens"] - 1, (rows, length), generator=generator),
        "mask": mask,
        "vector": torch.rand(rows, 6, generator=generator),
        "label": torch.randint(0, model["num_classes"], (rows,), generator=generator),
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_config = {k: v for k, v in config["model"].items()}
    training = config["training"]
    mapping = ComplexToExactRealMap()
    radius = CRITICAL_POLE_RADIUS["float32"]

    certificates: list[dict[str, Any]] = []
    reachability: list[dict[str, Any]] = []

    for model_seed in config["seeds"]["model"]:
        torch.manual_seed(int(model_seed))
        complex_model = ComplexOperatorState(**model_config)
        real_model = ExactRealBlockOperatorState(**model_config)
        real_model.copy_from_complex(complex_model)

        certificate = mapping.certify(
            complex_model, real_model, make_batch(config, int(config["seeds"]["data"]) + model_seed)
        )
        certificates.append({"model_seed": int(model_seed), **certificate.as_dict()})

        # Reachability probe: train and watch the pre-activation across every step.
        optimizer = torch.optim.AdamW(
            complex_model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        minimum_distance = math.inf
        step_of_minimum = -1
        for step in range(int(training["steps"])):
            batch = make_batch(config, int(config["seeds"]["data"]) + 1000 * model_seed + step)
            optimizer.zero_grad(set_to_none=True)
            with record_complex_tanh_inputs() as observed:
                logits = complex_model(**batch)
            loss = torch.nn.functional.cross_entropy(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(complex_model.parameters(), max_norm=5.0)
            optimizer.step()
            if observed["minimum_pole_distance"] < minimum_distance:
                minimum_distance = observed["minimum_pole_distance"]
                step_of_minimum = step

        reachability.append(
            {
                "model_seed": int(model_seed),
                "steps": int(training["steps"]),
                "minimum_pole_distance": minimum_distance,
                "step_of_minimum": step_of_minimum,
                "critical_radius": radius,
                "margin_multiples_of_radius": minimum_distance / radius,
                "entered_excluded_region": bool(minimum_distance <= radius),
            }
        )

    worst_margin = min(record["margin_multiples_of_radius"] for record in reachability)
    any_entered = any(record["entered_excluded_region"] for record in reachability)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "outcome_eligible": False,
        "status": "complete",
        "certificates": certificates,
        "pole_reachability": reachability,
        "summary": {
            "declared_level": "E2",
            "transport_degenerate": True,
            "worst_parameter_identity_residual": max(
                certificate["residuals"]["parameter_identity"] for certificate in certificates
            ),
            "worst_logit_residual": max(
                certificate["residuals"]["max_logit"] for certificate in certificates
            ),
            "worst_gradient_residual": max(
                certificate["residuals"]["max_gradient"] for certificate in certificates
            ),
            "minimum_pole_distance_observed": min(
                record["minimum_pole_distance"] for record in reachability
            ),
            "worst_margin_multiples_of_radius": worst_margin,
            "entered_excluded_region": any_entered,
        },
        "scientific_interpretation": (
            "Instrument result. The parameter map is the identity, so this pair is "
            "transport-degenerate and can bear on H4 (numerical implementation) only. The "
            "certificate declares E2 on a stated domain; E0 and E1 are refused because the two "
            "complex-tanh implementations diverge near the poles. The reachability figure is a "
            "probe of one configuration on random token streams and cannot establish that the "
            "excluded region is globally unreachable."
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
    (directory / "certificate.json").write_text(
        json.dumps(result["certificates"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = result["summary"]
    print(
        f"{EXPERIMENT_ID}: declared_level={summary['declared_level']} "
        f"transport_degenerate={summary['transport_degenerate']}"
    )
    print(f"  parameter identity residual : {summary['worst_parameter_identity_residual']:.3e}")
    print(f"  logit residual              : {summary['worst_logit_residual']:.3e}")
    print(f"  gradient residual           : {summary['worst_gradient_residual']:.3e}")
    print(f"  min pole distance observed  : {summary['minimum_pole_distance_observed']:.3e}")
    print(f"  margin (multiples of rho_c) : {summary['worst_margin_multiples_of_radius']:.1f}x")
    print(f"  entered excluded region     : {summary['entered_excluded_region']}")


if __name__ == "__main__":
    main()
