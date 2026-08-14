"""Real-arithmetic equivalents and mechanism-stealing controls."""

from __future__ import annotations

import math

import torch
from torch import nn

from qneuro.models.operators import ComplexOperatorState


def _complex_tanh_real_pair(
    real: torch.Tensor, imag: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate tanh(real + i imag) using only real tensor operations."""

    denominator = torch.cosh(2.0 * real) + torch.cos(2.0 * imag)
    denominator = denominator.clamp_min(torch.finfo(real.dtype).eps)
    return torch.sinh(2.0 * real) / denominator, torch.sin(2.0 * imag) / denominator


class ExactRealBlockOperatorState(nn.Module):
    """Real-tensor implementation of the complex operator's constrained function class.

    Parameters remain split into the same real and imaginary coordinates as ComplexOperatorState,
    but every transition and measurement is evaluated with explicit real block algebra. This is an
    exact representational control up to floating-point evaluation order, not an unrestricted real
    baseline.
    """

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        state_dim: int,
        rank: int,
        num_classes: int,
        step_size: float = 0.35,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.pad_token = int(pad_token)
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.initial_real = nn.Parameter(torch.empty(state_dim))
        self.initial_imag = nn.Parameter(torch.empty(state_dim))
        self.injection_real = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.injection_imag = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.left_real = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.left_imag = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.right_real = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.right_imag = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.demographic_real = nn.Parameter(torch.empty(2, state_dim))
        self.demographic_imag = nn.Parameter(torch.empty(2, state_dim))
        self.readout_real = nn.Parameter(torch.empty(num_classes, state_dim))
        self.readout_imag = nn.Parameter(torch.empty(num_classes, state_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.initial_real,
            self.initial_imag,
            self.injection_real,
            self.injection_imag,
            self.left_real,
            self.left_imag,
            self.right_real,
            self.right_imag,
            self.demographic_real,
            self.demographic_imag,
        ):
            nn.init.normal_(parameter, std=0.045)
        nn.init.normal_(self.readout_real, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.readout_imag, std=1.0 / math.sqrt(self.state_dim))

    @torch.no_grad()
    def copy_from_complex(self, source: ComplexOperatorState) -> None:
        """Copy a shape-compatible complex model into the explicit real representation."""

        if (source.num_tokens, source.state_dim, source.rank) != (
            self.num_tokens,
            self.state_dim,
            self.rank,
        ):
            raise ValueError("source and target operator shapes differ")
        source_parameters = dict(source.named_parameters())
        for name, parameter in self.named_parameters():
            parameter.copy_(source_parameters[name])

    def _normalize(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.sqrt(torch.sum(real.square() + imag.square(), dim=-1, keepdim=True)).clamp_min(
            self.eps
        )
        scale = math.sqrt(self.state_dim) / norm
        return real * scale, imag * scale

    def evolve_pair(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = tokens.shape
        real = self.initial_real.expand(batch_size, -1)
        imag = self.initial_imag.expand(batch_size, -1)
        if vector is not None:
            context = vector[:, -2:]
            real = real + context @ self.demographic_real
            imag = imag + context @ self.demographic_imag
        real, imag = self._normalize(real, imag)
        for position in range(sequence_length):
            active = mask[:, position, None]
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            left_real, left_imag = self.left_real[token], self.left_imag[token]
            right_real, right_imag = self.right_real[token], self.right_imag[token]
            projection_real = torch.einsum("bs,bsr->br", real, right_real) + torch.einsum(
                "bs,bsr->br", imag, right_imag
            )
            projection_imag = torch.einsum("bs,bsr->br", imag, right_real) - torch.einsum(
                "bs,bsr->br", real, right_imag
            )
            delta_real = (
                self.injection_real[token]
                + torch.einsum("bsr,br->bs", left_real, projection_real)
                - torch.einsum("bsr,br->bs", left_imag, projection_imag)
            )
            delta_imag = (
                self.injection_imag[token]
                + torch.einsum("bsr,br->bs", left_real, projection_imag)
                + torch.einsum("bsr,br->bs", left_imag, projection_real)
            )
            update_real, update_imag = _complex_tanh_real_pair(delta_real, delta_imag)
            candidate_real, candidate_imag = self._normalize(
                real + self.step_size * update_real,
                imag + self.step_size * update_imag,
            )
            real = torch.where(active, candidate_real, real)
            imag = torch.where(active, candidate_imag, imag)
        return real, imag

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        real, imag = self.evolve_pair(tokens, mask, vector)
        return torch.cat([real, imag], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        real, imag = self.evolve_pair(tokens, mask, vector)
        amplitude_real = real @ self.readout_real.T + imag @ self.readout_imag.T
        amplitude_imag = imag @ self.readout_real.T - real @ self.readout_imag.T
        return torch.log(amplitude_real.square() + amplitude_imag.square() + self.eps)
