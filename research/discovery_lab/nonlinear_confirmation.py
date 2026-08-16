"""DISCOVERY-001 step 5: the untouched nonlinear confirmation.

One attempt. Every threshold, grid point, criterion, and model definition is **read from**
`research/discovery_lab/frozen/DISCOVERY-001-P1.json`, whose hash was recorded before this file
generated any evidence. Nothing here restates a number that the frozen record already fixes, so the
code cannot silently drift from the prediction it is testing.

The prediction under test:

    rho = eta * lambda_max(H_target) / 2   evaluated at the mapped initialization
    target diverges  <=>  rho > 1

`H_target = S^-1 H_source S^-1` holds exactly for any twice-differentiable loss, so this is the same
quantity as the linear microcosm's `eta * lambda_max(H) / (2 s^2)`, not a refit of it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from qneuro.equivalence.microcosms import HomogeneousMLP, fixed_batch
from qneuro.equivalence.scaling import HomogeneousScalingMap

FROZEN_PATH = Path(__file__).resolve().parent / "frozen" / "DISCOVERY-001-P1.json"


def load_frozen() -> dict[str, Any]:
    """Load the frozen prediction and verify its hash before using any of it."""

    payload = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    prediction = payload["prediction"]
    recomputed = hashlib.sha256(
        json.dumps(prediction, indent=2, sort_keys=True).encode()
    ).hexdigest()
    if recomputed != payload["sha256_of_prediction"]:
        raise ValueError(
            "frozen prediction hash mismatch: the record was modified after freezing. "
            f"expected {payload['sha256_of_prediction']}, recomputed {recomputed}"
        )
    return prediction


def largest_hessian_eigenvalue(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    iterations: int = 200,
    tolerance: float = 1e-9,
) -> float:
    """Largest signed eigenvalue of the loss Hessian, by power iteration on Hessian-vector products.

    Shifted power iteration: iterating on ``H + cI`` for a large positive ``c`` converges to the
    most positive eigenvalue rather than the largest in magnitude, which is what governs blow-up
    and what the linear case used.
    """

    parameters = [p for p in model.parameters() if p.requires_grad]
    loss = torch.nn.functional.cross_entropy(model(batch["x"]), batch["y"])
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)

    def hessian_vector(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
        dot = sum((g * v).sum() for g, v in zip(gradients, vectors, strict=True))
        return [h.detach() for h in torch.autograd.grad(dot, parameters, retain_graph=True)]

    # Crude upper bound on |lambda| to use as the spectral shift.
    probe = [torch.randn_like(p) for p in parameters]
    norm = math.sqrt(sum(float((v * v).sum()) for v in probe))
    probe = [v / norm for v in probe]
    for _ in range(30):
        product = hessian_vector(probe)
        magnitude = math.sqrt(sum(float((v * v).sum()) for v in product))
        if magnitude <= 0.0:
            break
        probe = [v / magnitude for v in product]
    shift = 2.0 * magnitude + 1.0

    vector = [torch.randn_like(p) for p in parameters]
    norm = math.sqrt(sum(float((v * v).sum()) for v in vector))
    vector = [v / norm for v in vector]
    eigenvalue = 0.0
    for _ in range(iterations):
        product = hessian_vector(vector)
        shifted = [hv + shift * v for hv, v in zip(product, vector, strict=True)]
        magnitude = math.sqrt(sum(float((v * v).sum()) for v in shifted))
        if magnitude <= 0.0:
            return 0.0
        vector = [v / magnitude for v in shifted]
        previous, eigenvalue = eigenvalue, magnitude - shift
        if abs(eigenvalue - previous) <= tolerance * max(abs(eigenvalue), 1.0):
            break
    return eigenvalue


def _run(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    learning_rate: float,
    steps: int,
    optimizer_name: str,
    growth_threshold: float,
) -> dict[str, Any]:
    """Full-batch training; report whether the parameter norm is still growing at the end."""

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint = (2 * steps) // 3 - 1
    midpoint = float("nan")
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(batch["x"]), batch["y"])
        if not torch.isfinite(loss):
            return {"diverged": True, "growth_ratio": float("inf"), "final_norm": float("inf")}
        loss.backward()
        optimizer.step()
        norm = float(
            torch.linalg.vector_norm(torch.cat([p.detach().flatten() for p in model.parameters()]))
        )
        if not math.isfinite(norm):
            return {"diverged": True, "growth_ratio": float("inf"), "final_norm": float("inf")}
        if step == checkpoint:
            midpoint = norm

    final = norm
    if not math.isfinite(final) or not math.isfinite(midpoint):
        growth = float("inf")
    else:
        growth = final / midpoint if midpoint > 0.0 else 0.0
    return {
        "diverged": bool(not math.isfinite(growth) or growth > growth_threshold),
        "growth_ratio": growth,
        "final_norm": final,
    }


def confirm() -> dict[str, Any]:
    """Run the frozen grid once and score it against the frozen thresholds."""

    prediction = load_frozen()
    grid = prediction["frozen_grid"]
    criterion = prediction["divergence_criterion"]
    band = prediction["critical_band"]
    thresholds = prediction["pass_thresholds"]
    transition = float(prediction["predicted_transition"])
    growth_threshold = float(criterion["growth_ratio_threshold"])

    torch.set_default_dtype(torch.float64)
    records: list[dict[str, Any]] = []

    for width in grid["widths"]:
        for seed in grid["model_seeds"]:
            batch = fixed_batch(
                seed=seed + 5000,
                features=int(grid["in_features"]),
                batch_size=int(grid["batch_size"]),
                classes=int(grid["out_features"]),
            )
            batch = {"x": batch["x"].double(), "y": batch["y"]}
            for scale in grid["scales"]:
                mapping = HomogeneousScalingMap(float(scale))
                for learning_rate in grid["learning_rates"]:
                    torch.manual_seed(seed)
                    source = HomogeneousMLP(
                        int(grid["in_features"]), int(width), int(grid["out_features"])
                    ).double()
                    target = mapping.build_target(source)

                    eigenvalue = largest_hessian_eigenvalue(target, batch)
                    rho = float(learning_rate) * eigenvalue / 2.0

                    for optimizer_name in ("sgd", "adam"):
                        torch.manual_seed(seed)
                        fresh_source = HomogeneousMLP(
                            int(grid["in_features"]), int(width), int(grid["out_features"])
                        ).double()
                        fresh_target = mapping.build_target(fresh_source)
                        outcome = _run(
                            fresh_target,
                            batch,
                            float(learning_rate),
                            int(grid["steps"]),
                            optimizer_name,
                            growth_threshold,
                        )
                        in_band = float(band["low"]) < rho < float(band["high"])
                        records.append(
                            {
                                "width": int(width),
                                "seed": int(seed),
                                "scale": float(scale),
                                "learning_rate": float(learning_rate),
                                "optimizer": optimizer_name,
                                "lambda_max_target": eigenvalue,
                                "rho": rho,
                                "in_critical_band": in_band,
                                "predicted_diverged": bool(rho > transition),
                                **outcome,
                            }
                        )

    sgd = [r for r in records if r["optimizer"] == "sgd"]
    scored = [r for r in sgd if not r["in_critical_band"]]
    false_alarms = [r for r in scored if r["rho"] <= float(band["low"]) and r["diverged"]]
    misses = [r for r in scored if r["rho"] >= float(band["high"]) and not r["diverged"]]

    adam = [r for r in records if r["optimizer"] == "adam"]
    sgd_rate = sum(r["diverged"] for r in sgd) / len(sgd) if sgd else 0.0
    adam_rate = sum(r["diverged"] for r in adam) / len(adam) if adam else 0.0

    passes = len(false_alarms) <= int(thresholds["false_alarms_max"]) and len(misses) <= int(
        thresholds["misses_max"]
    )
    differential_passes = adam_rate < 0.2 * sgd_rate if sgd_rate > 0 else False

    return {
        "prediction_id": prediction["prediction_id"],
        "prediction_sha256": json.loads(FROZEN_PATH.read_text(encoding="utf-8"))[
            "sha256_of_prediction"
        ],
        "cells": len(records),
        "sgd_cells": len(sgd),
        "sgd_scored_cells": len(scored),
        "cells_in_critical_band": len(sgd) - len(scored),
        "false_alarms": len(false_alarms),
        "misses": len(misses),
        "sgd_divergence_rate": sgd_rate,
        "adam_divergence_rate": adam_rate,
        "primary_prediction_passes": passes,
        "differential_prediction_passes": differential_passes,
        "false_alarm_detail": false_alarms[:10],
        "miss_detail": misses[:10],
        "records": records,
    }
