"""Construct models with transparent real-scalar parameter budgeting."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from neuroworld import NeuroWorld
from qneuro.models import ComplexOperatorState, EvidenceMLP, RealOperatorState, TinyTransformer


def parameter_count(model: nn.Module) -> int:
    """Count trainable real scalars; complex entries count as two when native complex is used."""

    count = 0
    for parameter in model.parameters():
        multiplier = 2 if parameter.is_complex() else 1
        count += multiplier * parameter.numel()
    return count


def _nearest_width(
    builder: Callable[[int], nn.Module], candidates: list[int], budget: int
) -> tuple[nn.Module, int]:
    models = [(builder(width), width) for width in candidates]
    return min(models, key=lambda item: abs(parameter_count(item[0]) - budget))


def build_model(
    name: str,
    parameter_budget: int,
    rank: int,
    max_length: int,
    step_size: float,
) -> tuple[nn.Module, dict[str, int | float | str]]:
    if name == "mlp":
        model, width = _nearest_width(
            lambda value: EvidenceMLP(82, value, NeuroWorld.num_diagnoses),
            list(range(8, 513)),
            parameter_budget,
        )
    elif name == "transformer":
        model, width = _nearest_width(
            lambda value: TinyTransformer(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                NeuroWorld.num_diagnoses,
                value,
                max_length,
            ),
            list(range(8, 129, 4)),
            parameter_budget,
        )
    elif name == "real_operator":
        model, width = _nearest_width(
            lambda value: RealOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 257)),
            parameter_budget,
        )
    elif name == "complex_operator":
        model, width = _nearest_width(
            lambda value: ComplexOperatorState(
                NeuroWorld.num_tokens,
                NeuroWorld.pad_token,
                value,
                rank,
                NeuroWorld.num_diagnoses,
                step_size,
            ),
            list(range(4, 129)),
            parameter_budget,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return model, {
        "name": name,
        "width": width,
        "rank": rank if "operator" in name else 0,
        "parameter_count": parameter_count(model),
        "parameter_budget": parameter_budget,
    }
