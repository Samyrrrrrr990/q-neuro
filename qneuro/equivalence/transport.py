"""Paired training under a transported map.

This is the harness behind QE-000002. It warms one member of a pair up so that the optimizer
accumulates real state, transports at a chosen level, then runs both members on an identical batch
stream and reports how far their predictions separate.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from qneuro.equivalence.defects import predictive_divergence
from qneuro.equivalence.maps import ParameterMap
from qneuro.equivalence.microcosms import batch_stream


def build_optimizer(
    name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    learning_rate_scales: Mapping[str, float] | None = None,
) -> torch.optim.Optimizer:
    """Build an optimizer with one parameter group per named parameter.

    Per-parameter groups exist so a map can transport the learning rate, which is the only way some
    update rules become conjugate across a scaling orbit.
    """

    scales = dict(learning_rate_scales or {})
    groups: list[dict[str, Any]] = [
        {"params": [parameter], "lr": learning_rate * scales.get(name, 1.0)}
        for name, parameter in model.named_parameters()
    ]
    if name == "sgd":
        return torch.optim.SGD(groups, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name!r}")


def named_optimizer_state(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, Any]]:
    """Re-key optimizer state from parameter objects to parameter names."""

    return {
        name: dict(optimizer.state[parameter])
        for name, parameter in model.named_parameters()
        if parameter in optimizer.state
    }


def load_named_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, parameter in model.named_parameters():
        entries = state.get(name)
        if entries is None:
            continue
        optimizer.state[parameter] = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in entries.items()
        }


def _model_shape(model: nn.Module, attribute: str, submodule: str) -> int:
    """Read an input/output width from the model, or from its named submodule."""

    value = getattr(model, attribute, None)
    if value is None:
        value = getattr(getattr(model, submodule), attribute)
    return int(value)


def _step(
    model: nn.Module, optimizer: torch.optim.Optimizer, batch: Mapping[str, torch.Tensor]
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(batch["x"]), batch["y"])
    loss.backward()
    optimizer.step()


def paired_training_divergence(
    source: nn.Module,
    target: nn.Module,
    mapping: ParameterMap,
    *,
    optimizer_name: str = "adamw",
    warmup_steps: int = 15,
    measured_steps: int = 25,
    transport_optimizer_state: bool = True,
    transport_learning_rate: bool = False,
    seed: int = 0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict[str, float]:
    """Measure predictive divergence across a map during paired training.

    The caller's modules are never mutated. ``transport_optimizer_state=False`` is the deliberate
    negative control: it leaves the target optimizer in its fresh state so that the instrument's
    ability to detect a broken transport can itself be tested.

    ``transport_learning_rate`` climbs one further rung, applying the per-parameter learning-rate
    factors the map declares. For some families this is the difference between a conjugate and a
    non-conjugate update map, and isolating it is the point of the measurement.
    """

    work_source = copy.deepcopy(source)
    work_target = copy.deepcopy(target)

    batches = batch_stream(
        seed,
        _model_shape(work_source, "in_features", "first"),
        warmup_steps + measured_steps,
        classes=_model_shape(work_source, "out_features", "second"),
    )

    source_optimizer = build_optimizer(optimizer_name, work_source, learning_rate, weight_decay)
    for batch in batches[:warmup_steps]:
        _step(work_source, source_optimizer, batch)

    # Transport parameters exactly at the hand-off point.
    mapped_parameters = mapping.map_parameters(dict(work_source.named_parameters()))
    with torch.no_grad():
        for name, parameter in work_target.named_parameters():
            parameter.copy_(mapped_parameters[name])

    target_optimizer = build_optimizer(
        optimizer_name,
        work_target,
        learning_rate,
        weight_decay,
        learning_rate_scales=(
            mapping.learning_rate_scales(optimizer_name) if transport_learning_rate else None
        ),
    )
    if transport_optimizer_state:
        load_named_optimizer_state(
            work_target,
            target_optimizer,
            mapping.map_optimizer_state(named_optimizer_state(work_source, source_optimizer)),
        )

    max_logit = 0.0
    max_total_variation = 0.0
    reference_scale = 1.0
    for batch in batches[warmup_steps:]:
        with torch.no_grad():
            source_logits = work_source(batch["x"])
            target_logits = work_target(batch["x"])
        divergence = predictive_divergence(source_logits, target_logits)
        max_logit = max(max_logit, divergence["max_logit"])
        max_total_variation = max(max_total_variation, divergence["max_total_variation"])
        reference_scale = max(reference_scale, float(source_logits.abs().max()))
        _step(work_source, source_optimizer, batch)
        _step(work_target, target_optimizer, batch)

    return {
        "max_logit_divergence": max_logit,
        "max_total_variation": max_total_variation,
        "reference_logit_scale": reference_scale,
        "warmup_steps": float(warmup_steps),
        "measured_steps": float(measured_steps),
        "optimizer_state_transported": float(transport_optimizer_state),
        "learning_rate_transported": float(transport_learning_rate),
    }
