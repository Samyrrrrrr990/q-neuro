"""Per-step transport traces: the raw material the estimator study consumes.

A paired run started from a mapped initialization has ``e_0 = 0``, so the first step's divergence is
the first defect. After that the two trajectories have drifted and the *one-step* defect is no
longer what you observe — you observe the accumulation. Measuring `delta_k` at later steps therefore
requires re-coupling: snapshot both systems, set the target to the mapped source, take one step on
each, read the disagreement, restore.

That is what this module does. Everything is measured, nothing is inferred from a closed form,
because the point of the estimator study is to find out which measurable quantity predicts the
final gap.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qneuro.equivalence.maps import ParameterMap
from qneuro.equivalence.microcosms import batch_stream
from qneuro.equivalence.transport import (
    _model_shape,
    _step,
    build_optimizer,
    load_named_optimizer_state,
    named_optimizer_state,
)


def _predictions(model: nn.Module, probe: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(probe).detach().clone()


def _snapshot(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }


def _restore(
    model: nn.Module, optimizer: torch.optim.Optimizer, snapshot: Mapping[str, Any]
) -> None:
    model.load_state_dict(snapshot["model"])
    optimizer.load_state_dict(snapshot["optimizer"])


def paired_transport_trace(
    source: nn.Module,
    target: nn.Module,
    mapping: ParameterMap,
    *,
    optimizer_name: str = "adamw",
    steps: int = 40,
    transport_optimizer_state: bool = True,
    transport_learning_rate: bool = False,
    seed: int = 0,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> dict[str, Any]:
    """Run a coupled pair and record per-step defect, divergence, and amplification.

    Returns series in predictive space, which is the coordinate-free side of the comparison, plus
    the scalar features and target used by the estimator study.
    """

    work_source = copy.deepcopy(source)
    work_target = copy.deepcopy(target)
    features = _model_shape(work_source, "in_features", "first")
    classes = _model_shape(work_source, "out_features", "second")
    batches = batch_stream(seed, features, steps, classes=classes)
    probe = batch_stream(seed + 991, features, 1, classes=classes)[0]["x"]

    source_optimizer = build_optimizer(optimizer_name, work_source, learning_rate, weight_decay)
    target_optimizer = build_optimizer(
        optimizer_name,
        work_target,
        learning_rate,
        weight_decay,
        learning_rate_scales=(
            mapping.learning_rate_scales(optimizer_name) if transport_learning_rate else None
        ),
    )

    # Start coupled: mapped parameters, and mapped optimizer state where the map supports it.
    mapped = mapping.map_parameters(dict(work_source.named_parameters()))
    with torch.no_grad():
        for name, parameter in work_target.named_parameters():
            parameter.copy_(mapped[name])

    defects: list[float] = []
    divergences: list[float] = []
    gradient_norms: list[float] = []
    losses: list[float] = []

    for index, batch in enumerate(batches):
        divergences.append(
            float((_predictions(work_source, probe) - _predictions(work_target, probe)).abs().max())
        )

        # --- one-step defect, measured from a re-coupled state ---
        source_snapshot = _snapshot(work_source, source_optimizer)
        target_snapshot = _snapshot(work_target, target_optimizer)
        recoupled = mapping.map_parameters(dict(work_source.named_parameters()))
        with torch.no_grad():
            for name, parameter in work_target.named_parameters():
                parameter.copy_(recoupled[name])
        if transport_optimizer_state and mapping.supports_optimizer_transport:
            load_named_optimizer_state(
                work_target,
                target_optimizer,
                mapping.map_optimizer_state(named_optimizer_state(work_source, source_optimizer)),
            )
        _step(work_source, source_optimizer, batch)
        _step(work_target, target_optimizer, batch)
        defects.append(
            float((_predictions(work_source, probe) - _predictions(work_target, probe)).abs().max())
        )
        _restore(work_source, source_optimizer, source_snapshot)
        _restore(work_target, target_optimizer, target_snapshot)

        # --- the actual coupled step ---
        source_optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(work_source(batch["x"]), batch["y"])
        loss.backward()
        losses.append(float(loss.detach()))
        gradient_norms.append(
            float(
                torch.linalg.vector_norm(
                    torch.cat(
                        [
                            p.grad.detach().flatten().abs()
                            for p in work_source.parameters()
                            if p.grad is not None
                        ]
                    )
                )
            )
        )
        source_optimizer.step()
        _step(work_target, target_optimizer, batch)
        del index

    final_divergence = float(
        (_predictions(work_source, probe) - _predictions(work_target, probe)).abs().max()
    )

    # Measured local amplification: how much the existing divergence grew this step. Only defined
    # where the previous divergence is above the rounding floor.
    floor = float(torch.finfo(torch.float32).eps)
    amplifications = [
        divergences[k + 1] / divergences[k]
        for k in range(len(divergences) - 1)
        if divergences[k] > floor
    ]
    mean_amplification = sum(amplifications) / len(amplifications) if amplifications else 1.0

    cumulative_defect = sum(defects)
    amplified_defect = 0.0
    for k, defect in enumerate(defects):
        amplified_defect += defect * (mean_amplification ** max(len(defects) - 1 - k, 0))

    return {
        "defects": defects,
        "divergences": divergences,
        "final_divergence": final_divergence,
        "one_step_defect": defects[0] if defects else 0.0,
        "cumulative_defect": cumulative_defect,
        "amplified_defect": amplified_defect,
        "mean_amplification": mean_amplification,
        "one_step_predictive_divergence": divergences[1] if len(divergences) > 1 else 0.0,
        "total_gradient_norm": sum(gradient_norms),
        "loss_decrease": (losses[0] - losses[-1]) if losses else 0.0,
        "learning_rate": learning_rate,
        "parameter_count": float(sum(p.numel() for p in work_source.parameters())),
        "steps": float(steps),
    }
