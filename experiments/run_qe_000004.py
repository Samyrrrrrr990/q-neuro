"""QE-000004: dense versus factorized linear maps.

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Discovery split.

This family tests whether the framework generalizes beyond exact symmetries. The forward map
``(U, V) -> UV`` is exact, but it is non-injective and admits **no** optimizer-state transport,
because factor-space descent induces a state-dependent preconditioner on the product rather than a
coordinate change of dense descent.

The experiment therefore measures two things:

1. the refusals: inverse, gradient transport, and optimizer transport must all decline;
2. the induced preconditioner, which is what the implicit bias literature predicts should make the
   two parameterizations train differently despite realizing the same function.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence import FactorizedToDenseMap
from qneuro.equivalence.factorization import (
    DenseLinear,
    FactorizedLinear,
    align_factorized_to_dense,
)
from qneuro.equivalence.microcosms import batch_stream

EXPERIMENT_ID = "QE-000004"
SCHEMA_VERSION = "1.0.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "dense_versus_factorized_linear",
    "description": (
        "Exact non-injective map with no optimizer transport. Measures the induced preconditioner "
        "and the divergence of the realized weight trajectories under matched training."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "discovery",
    "split": "discovery",
    "outcome_eligible": False,
    "family": "factorization",
    "model": {"in_features": 6, "out_features": 4, "ranks": [4, 2]},
    "training": {
        "optimizers": ["sgd", "adamw"],
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "steps": 40,
        "batch_size": 32,
        "device": "cpu",
    },
    "seeds": {"model": [0, 1, 2], "stream": 9},
    "protocol_deviations": [
        "Analytic microcosm only; no task generator and no architecture claim.",
        (
            "Discovery split: these cells may inform estimator design and may not be reused for "
            "confirmation."
        ),
    ],
}


def _refusal_audit(mapping: FactorizedToDenseMap) -> dict[str, Any]:
    """Record that the framework declines what it cannot do, rather than faking it."""

    audit: dict[str, Any] = {"supports_optimizer_transport": mapping.supports_optimizer_transport}
    for label, call in (
        ("inverse", lambda: mapping.unmap_parameters({"weight": torch.eye(2)})),
        ("gradient_transport", lambda: mapping.map_gradients({"left": torch.eye(2)})),
        ("optimizer_state_transport", lambda: mapping.map_optimizer_state({})),
    ):
        try:
            call()
        except NotImplementedError as error:
            audit[label] = {"refused": True, "reason": str(error).split(".")[0]}
        else:  # pragma: no cover - a silent success here is a framework regression
            audit[label] = {"refused": False, "reason": "map did not refuse"}
    return audit


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_config = config["model"]
    training = config["training"]
    mapping = FactorizedToDenseMap()

    records: list[dict[str, Any]] = []
    for rank in model_config["ranks"]:
        for model_seed in config["seeds"]["model"]:
            torch.manual_seed(int(model_seed))
            dense = DenseLinear(model_config["in_features"], model_config["out_features"])
            factorized = FactorizedLinear(
                model_config["in_features"], model_config["out_features"], rank=rank
            )
            align_factorized_to_dense(dense, factorized)

            alignment_residual = float(
                (factorized.effective_weight().detach() - dense.weight.detach()).abs().max()
            )
            full_rank = rank >= min(model_config["in_features"], model_config["out_features"])

            for optimizer_name in training["optimizers"]:
                dense_model = copy.deepcopy(dense)
                factorized_model = copy.deepcopy(factorized)
                batches = batch_stream(
                    int(config["seeds"]["stream"]),
                    model_config["in_features"],
                    int(training["steps"]),
                    batch_size=int(training["batch_size"]),
                    classes=model_config["out_features"],
                )
                builder = torch.optim.SGD if optimizer_name == "sgd" else torch.optim.AdamW
                dense_optimizer = builder(
                    dense_model.parameters(),
                    lr=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                )
                factorized_optimizer = builder(
                    factorized_model.parameters(),
                    lr=float(training["learning_rate"]),
                    weight_decay=float(training["weight_decay"]),
                )

                cosine_first_step = None
                max_logit_divergence = 0.0
                for index, batch in enumerate(batches):
                    with torch.no_grad():
                        max_logit_divergence = max(
                            max_logit_divergence,
                            float(
                                (dense_model(batch["x"]) - factorized_model(batch["x"])).abs().max()
                            ),
                        )
                    dense_optimizer.zero_grad(set_to_none=True)
                    torch.nn.functional.cross_entropy(
                        dense_model(batch["x"]), batch["y"]
                    ).backward()
                    dense_gradient = dense_model.weight.grad.detach().clone()

                    if index == 0:
                        induced = mapping.induced_preconditioned_gradient(
                            factorized_model.left.detach(),
                            factorized_model.right.detach(),
                            dense_gradient,
                        )
                        cosine_first_step = float(
                            torch.nn.functional.cosine_similarity(
                                induced.flatten(), dense_gradient.flatten(), dim=0
                            )
                        )
                    dense_optimizer.step()

                    factorized_optimizer.zero_grad(set_to_none=True)
                    torch.nn.functional.cross_entropy(
                        factorized_model(batch["x"]), batch["y"]
                    ).backward()
                    factorized_optimizer.step()

                weight_divergence = float(
                    (dense_model.weight.detach() - factorized_model.effective_weight().detach())
                    .abs()
                    .max()
                )
                records.append(
                    {
                        "rank": int(rank),
                        "full_rank": full_rank,
                        "model_seed": int(model_seed),
                        "optimizer": optimizer_name,
                        "alignment_residual": alignment_residual,
                        "induced_gradient_cosine_first_step": cosine_first_step,
                        "max_logit_divergence": max_logit_divergence,
                        "effective_weight_divergence": weight_divergence,
                    }
                )

    full_rank_records = [record for record in records if record["full_rank"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "split": config["split"],
        "outcome_eligible": False,
        "status": "complete",
        "records": records,
        "refusal_audit": _refusal_audit(mapping),
        "summary": {
            "cells": len(records),
            "worst_full_rank_alignment_residual": max(
                record["alignment_residual"] for record in full_rank_records
            ),
            "smallest_induced_gradient_cosine": min(
                record["induced_gradient_cosine_first_step"] for record in full_rank_records
            ),
            "largest_induced_gradient_cosine": max(
                record["induced_gradient_cosine_first_step"] for record in full_rank_records
            ),
            "worst_effective_weight_divergence": max(
                record["effective_weight_divergence"] for record in full_rank_records
            ),
            "all_transport_operations_refused": all(
                entry["refused"]
                for key, entry in _refusal_audit(mapping).items()
                if isinstance(entry, dict)
            ),
        },
        "scientific_interpretation": (
            "Discovery evidence on an analytic microcosm. Two parameterizations that realize the "
            "same predictor at initialization separate under matched training because factor-space "
            "descent preconditions the effective weight. No optimizer-state transport exists for "
            "this map, and the framework refuses rather than approximating one. Nothing here "
            "supports an architecture claim or a transport-covariance law."
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
    print(f"  full-rank alignment residual  : {summary['worst_full_rank_alignment_residual']:.3e}")
    print(
        "  induced gradient cosine range : "
        f"[{summary['smallest_induced_gradient_cosine']:.4f}, "
        f"{summary['largest_induced_gradient_cosine']:.4f}]"
    )
    print(f"  worst effective W divergence  : {summary['worst_effective_weight_divergence']:.3e}")
    print(f"  all transport ops refused     : {summary['all_transport_operations_refused']}")


if __name__ == "__main__":
    main()
