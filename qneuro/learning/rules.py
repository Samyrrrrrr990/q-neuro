"""Controlled global, phase-coded, and local learning rules.

These routines are intentionally small research prototypes. Phase Gradient Optimization (PGO)
still uses autograd to obtain task gradients; its experimental contribution is how gradients are
combined in paired real/imaginary parameter planes. The local and centroid rules never call
``backward`` and update only quantities available at the current state transition.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from qneuro.models import ComplexOperatorState
from qneuro.models.operators import _bounded_norm_complex


class AuxiliaryTrainingModel(nn.Module):
    """Attach disposable mechanism/localization heads to a diagnostic model."""

    def __init__(self, base: ComplexOperatorState):
        super().__init__()
        self.base = base
        representation_dim = 2 * base.state_dim
        self.mechanism_head = nn.Linear(representation_dim, 5)
        self.localization_head = nn.Linear(representation_dim, 4)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.base(**batch)

    def task_logits(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = self.base.evolve(batch["tokens"], batch["mask"], batch.get("vector"))
        representation = torch.cat([state.real, state.imag], dim=-1)
        return {
            "diagnosis": self.base.measure(state),
            "mechanism": self.mechanism_head(representation),
            "localization": self.localization_head(representation),
        }


def multi_objective_losses(
    model: AuxiliaryTrainingModel,
    batch: dict[str, torch.Tensor],
    auxiliary_weight: float,
) -> dict[str, torch.Tensor]:
    """Return diagnosis and two simulator-factor losses without fabricating order labels."""

    logits = model.task_logits(batch)
    labels = batch["label"].long()
    losses = {"diagnosis": F.cross_entropy(logits["diagnosis"], labels)}
    factorial = labels >= 8
    if bool(factorial.any()):
        local_index = labels[factorial] - 8
        losses["mechanism"] = auxiliary_weight * F.cross_entropy(
            logits["mechanism"][factorial], local_index % 5
        )
        losses["localization"] = auxiliary_weight * F.cross_entropy(
            logits["localization"][factorial], (local_index // 3) % 4
        )
    return losses


def _named_trainable_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def _task_gradients(
    losses: dict[str, torch.Tensor], parameters: list[nn.Parameter]
) -> tuple[list[str], list[list[torch.Tensor]]]:
    task_names = list(losses)
    output: list[list[torch.Tensor]] = []
    for index, name in enumerate(task_names):
        raw = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=index < len(task_names) - 1,
            allow_unused=True,
        )
        output.append(
            [
                torch.zeros_like(parameter) if value is None else value
                for parameter, value in zip(parameters, raw, strict=True)
            ]
        )
    return task_names, output


def _global_dot(first: list[torch.Tensor], second: list[torch.Tensor]) -> torch.Tensor:
    return sum(
        (left.float() * right.float()).sum() for left, right in zip(first, second, strict=True)
    )


def _cosine(first: list[torch.Tensor], second: list[torch.Tensor]) -> float:
    dot = _global_dot(first, second)
    norm = torch.sqrt(
        _global_dot(first, first).clamp_min(1e-20) * _global_dot(second, second).clamp_min(1e-20)
    )
    return float((dot / norm).clamp(-1.0, 1.0).detach().cpu())


def apply_pcgrad(losses: dict[str, torch.Tensor], model: nn.Module) -> dict[str, float]:
    """Project conflicting task gradients and install their mean on ``parameter.grad``."""

    named = _named_trainable_parameters(model)
    parameters = [parameter for _, parameter in named]
    task_names, task_gradients = _task_gradients(losses, parameters)
    diagnosis = task_gradients[0]
    diagnostics = {
        f"diagnosis_cosine_{name}": _cosine(diagnosis, gradients)
        for name, gradients in zip(task_names[1:], task_gradients[1:], strict=True)
    }
    projected: list[list[torch.Tensor]] = []
    for index, gradients in enumerate(task_gradients):
        current = [value.clone() for value in gradients]
        for other_index, other in enumerate(task_gradients):
            if index == other_index:
                continue
            dot = _global_dot(current, other)
            if float(dot.detach()) < 0.0:
                scale = dot / _global_dot(other, other).clamp_min(1e-20)
                current = [
                    value - scale.to(value.dtype) * reference
                    for value, reference in zip(current, other, strict=True)
                ]
        projected.append(current)
    for parameter_index, parameter in enumerate(parameters):
        parameter.grad = torch.stack([gradients[parameter_index] for gradients in projected]).mean(
            dim=0
        )
    return diagnostics


def apply_phase_gradient(losses: dict[str, torch.Tensor], model: nn.Module) -> dict[str, float]:
    """Combine task gradients through adaptive rotations in complex parameter planes.

    Diagnosis defines phase zero. For each auxiliary task, half the angular distance implied by
    its cosine with the diagnosis gradient is used as a phase. Agreement therefore stays aligned,
    while a fully opposed update is moved into quadrature instead of cancelling diagnosis. This is
    meaningful only for parameters with explicit real/imaginary pairs; unpaired parameters receive
    cosine-attenuated task sums.
    """

    named = _named_trainable_parameters(model)
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    name_to_index = {name: index for index, name in enumerate(names)}
    task_names, task_gradients = _task_gradients(losses, parameters)
    diagnosis = task_gradients[0]
    cosines = [1.0] + [_cosine(diagnosis, values) for values in task_gradients[1:]]
    phases = [0.0] + [0.5 * math.acos(max(-1.0, min(1.0, value))) for value in cosines[1:]]
    handled: set[int] = set()
    for index, (name, parameter) in enumerate(named):
        if index in handled:
            continue
        if name.endswith("_real"):
            imaginary_name = f"{name[:-5]}_imag"
            imaginary_index = name_to_index.get(imaginary_name)
        else:
            imaginary_index = None
        if imaginary_index is not None and parameters[imaginary_index].shape == parameter.shape:
            real_update = torch.zeros_like(parameter)
            imaginary_update = torch.zeros_like(parameters[imaginary_index])
            for phase, gradients in zip(phases, task_gradients, strict=True):
                cosine = math.cos(phase)
                sine = math.sin(phase)
                real = gradients[index]
                imaginary = gradients[imaginary_index]
                real_update = real_update + cosine * real - sine * imaginary
                imaginary_update = imaginary_update + sine * real + cosine * imaginary
            parameter.grad = real_update / len(task_gradients)
            parameters[imaginary_index].grad = imaginary_update / len(task_gradients)
            handled.update((index, imaginary_index))
        else:
            parameter.grad = sum(
                math.cos(phase) * gradients[index]
                for phase, gradients in zip(phases, task_gradients, strict=True)
            ) / len(task_gradients)
            handled.add(index)
    diagnostics: dict[str, float] = {}
    for task_name, cosine, phase in zip(task_names[1:], cosines[1:], phases[1:], strict=True):
        diagnostics[f"diagnosis_cosine_{task_name}"] = cosine
        diagnostics[f"phase_radians_{task_name}"] = phase
    return diagnostics


def _complex_parameters(model: ComplexOperatorState) -> dict[str, torch.Tensor]:
    return {
        "initial": torch.complex(model.initial_real, model.initial_imag),
        "injection": torch.complex(model.injection_real, model.injection_imag),
        "left": torch.complex(model.left_real, model.left_imag),
        "right": torch.complex(model.right_real, model.right_imag),
        "demographic": torch.complex(model.demographic_real, model.demographic_imag),
    }


def _write_complex(
    target_real: nn.Parameter, target_imag: nn.Parameter, value: torch.Tensor
) -> None:
    target_real.copy_(value.real)
    target_imag.copy_(value.imag)


def _prototype_codes(
    num_classes: int, state_dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    real = torch.randn(num_classes, state_dim, generator=generator, device=device)
    imag = torch.randn(num_classes, state_dim, generator=generator, device=device)
    return _bounded_norm_complex(torch.complex(real, imag))


@torch.no_grad()
def local_plasticity_epoch(
    model: ComplexOperatorState,
    batches: Iterable[dict[str, torch.Tensor]],
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    """Run a supervised transition-local complex delta/Hebbian update without autograd."""

    device = model.initial_real.device
    codes = _prototype_codes(model.readout_real.shape[0], model.state_dim, seed, device)
    _write_complex(model.readout_real, model.readout_imag, codes)
    total_error = 0.0
    total_transitions = 0
    for batch in batches:
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        vector = batch["vector"].to(device)
        labels = batch["label"].to(device).long()
        target = codes[labels]
        values = _complex_parameters(model)
        state = values["initial"].expand(tokens.shape[0], -1)
        state = state + vector[:, -2:].to(values["demographic"].dtype) @ values["demographic"]
        state = _bounded_norm_complex(state)
        for position in range(tokens.shape[1]):
            active = mask[:, position]
            if not bool(active.any()):
                continue
            token = tokens[:, position].clamp_max(model.num_tokens - 1)
            previous = state
            left = values["left"][token]
            right = values["right"][token]
            projection = torch.einsum("bs,bsr->br", previous, right.conj())
            delta = values["injection"][token] + torch.einsum("bsr,br->bs", left, projection)
            candidate = _bounded_norm_complex(previous + model.step_size * torch.tanh(delta))
            state = torch.where(active[:, None], candidate, previous)
            error = torch.where(active[:, None], target - state, torch.zeros_like(state))
            total_error += float(torch.abs(error[active]).square().mean().cpu()) * int(active.sum())
            total_transitions += int(active.sum())

            token_active = token[active]
            error_active = error[active]
            previous_active = previous[active]
            left_active = left[active]
            projection_active = projection[active]
            left_signal = error_active[:, :, None] * projection_active.conj()[:, None, :]
            right_signal = (
                previous_active[:, :, None]
                * torch.einsum("bsr,bs->br", left_active.conj(), error_active).conj()[:, None, :]
            )
            counts = torch.bincount(token_active, minlength=model.num_tokens).clamp_min(1)
            injection_update = torch.zeros_like(values["injection"])
            left_update = torch.zeros_like(values["left"])
            right_update = torch.zeros_like(values["right"])
            injection_update.index_add_(0, token_active, error_active)
            left_update.index_add_(0, token_active, left_signal)
            right_update.index_add_(0, token_active, right_signal)
            injection_update = injection_update / counts[:, None]
            left_update = left_update / counts[:, None, None]
            right_update = right_update / counts[:, None, None]
            injection_update = injection_update / torch.linalg.vector_norm(
                injection_update, dim=-1, keepdim=True
            ).clamp_min(1.0)
            left_update = (
                left_update
                / torch.linalg.vector_norm(left_update.flatten(1), dim=-1, keepdim=True).clamp_min(
                    1.0
                )[:, None]
            )
            right_update = (
                right_update
                / torch.linalg.vector_norm(right_update.flatten(1), dim=-1, keepdim=True).clamp_min(
                    1.0
                )[:, None]
            )
            values["injection"] = values["injection"] + learning_rate * injection_update
            values["left"] = values["left"] + 0.25 * learning_rate * left_update
            values["right"] = values["right"] + 0.25 * learning_rate * right_update
            _write_complex(model.injection_real, model.injection_imag, values["injection"])
            _write_complex(model.left_real, model.left_imag, values["left"])
            _write_complex(model.right_real, model.right_imag, values["right"])

        demographic_update = vector[:, -2:].T.to(error.dtype) @ (target - state) / tokens.shape[0]
        values["demographic"] = values["demographic"] + 0.2 * learning_rate * demographic_update
        _write_complex(model.demographic_real, model.demographic_imag, values["demographic"])
    return {"local_transition_mse": total_error / max(1, total_transitions)}


@torch.no_grad()
def fit_centroid_readout(
    model: ComplexOperatorState,
    batches: Iterable[dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Fit a class-centroid Born readout with frozen dynamics and no differentiation."""

    device = model.initial_real.device
    num_classes = model.readout_real.shape[0]
    sums = torch.zeros(num_classes, model.state_dim, dtype=torch.complex64, device=device)
    counts = torch.zeros(num_classes, device=device)
    for batch in batches:
        labels = batch["label"].to(device).long()
        state = model.evolve(
            batch["tokens"].to(device), batch["mask"].to(device), batch["vector"].to(device)
        )
        sums.index_add_(0, labels, state)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=counts.dtype))
    centroids = sums / counts[:, None].clamp_min(1.0)
    centroids = _bounded_norm_complex(centroids)
    _write_complex(model.readout_real, model.readout_imag, centroids)
    return {
        "centroid_min_count": float(counts.min().cpu()),
        "centroid_mean_norm": float(torch.linalg.vector_norm(centroids, dim=-1).mean().cpu()),
    }
