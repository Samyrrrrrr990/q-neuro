"""Frozen-representation probes for NeuroWorld's known causal properties."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def factorial_property_labels(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Recover the declared factorial causes for diagnosis labels 8-19."""

    if bool((labels < 8).any()):
        raise ValueError("factorial property labels are defined only for diagnoses 8-19")
    local_index = labels.long() - 8
    return {
        "mechanism": local_index % 5,
        "localization": (local_index // 3) % 4,
        "temporality": (2 * local_index + 1) % 3,
        "context": (3 * local_index + 2) % 4,
    }


class LinearObservableProbe(nn.Module):
    """Affine readout over a frozen real representation."""

    def __init__(self, representation_dim: int, num_classes: int):
        super().__init__()
        self.readout = nn.Linear(representation_dim, num_classes)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.readout(representation)


class HermitianObservableProbe(nn.Module):
    """Learn class observables A_c and return real expectations z^H A_c z."""

    def __init__(self, state_dim: int, num_classes: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.raw_real = nn.Parameter(torch.empty(num_classes, state_dim, state_dim))
        self.raw_imag = nn.Parameter(torch.empty(num_classes, state_dim, state_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.raw_real, std=0.03)
        nn.init.normal_(self.raw_imag, std=0.03)

    def matrices(self) -> torch.Tensor:
        real = 0.5 * (self.raw_real + self.raw_real.transpose(-2, -1))
        imag = 0.5 * (self.raw_imag - self.raw_imag.transpose(-2, -1))
        return torch.complex(real, imag)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        if representation.shape[-1] != 2 * self.state_dim:
            raise ValueError("representation must concatenate real and imaginary state channels")
        state = torch.complex(
            representation[:, : self.state_dim], representation[:, self.state_dim :]
        )
        expectation = torch.einsum("bi,cij,bj->bc", state.conj(), self.matrices(), state)
        return expectation.real + self.bias


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    target_probability = probabilities.gather(1, targets[:, None]).squeeze(1).clamp_min(1e-12)
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(targets).float().mean()),
        "nll": float(-torch.log(target_probability).mean()),
    }


def fit_probe(
    builder: Callable[[], nn.Module],
    train_representation: torch.Tensor,
    train_targets: torch.Tensor,
    test_representation: torch.Tensor,
    test_targets: torch.Tensor,
    seed: int,
    epochs: int = 250,
    learning_rate: float = 0.03,
    weight_decay: float = 1e-3,
    standardize: bool = True,
) -> tuple[nn.Module, dict[str, float]]:
    """Fit a small probe with training-only standardization and checkpoint selection."""

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(train_representation.shape[0], generator=generator)
    validation_count = max(1, train_representation.shape[0] // 5)
    validation_index = permutation[:validation_count]
    fit_index = permutation[validation_count:]
    if standardize:
        mean = train_representation[fit_index].mean(dim=0, keepdim=True)
        scale = (
            train_representation[fit_index].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-5)
        )
        train = (train_representation[fit_index] - mean) / scale
        validation = (train_representation[validation_index] - mean) / scale
        test = (test_representation - mean) / scale
    else:
        train = train_representation[fit_index]
        validation = train_representation[validation_index]
        test = test_representation
    fit_targets = train_targets[fit_index]
    validation_targets = train_targets[validation_index]
    probe = builder()
    optimizer = torch.optim.AdamW(probe.parameters(), learning_rate, weight_decay=weight_decay)
    best_validation_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = probe(train)
        loss = torch.nn.functional.cross_entropy(logits, fit_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            validation_nll = float(
                torch.nn.functional.cross_entropy(probe(validation), validation_targets)
            )
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_state = {
                key: value.detach().cpu().clone() for key, value in probe.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("probe fitting did not produce a checkpoint")
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        metrics = _metrics(probe(test), test_targets)
    metrics["validation_nll"] = best_validation_nll
    return probe, metrics
