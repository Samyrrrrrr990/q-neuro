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


class RealRotationBlockOperator(nn.Module):
    """Real paired state with token-specific SO(2) blocks and affine evidence injection."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        pair_dim: int,
        num_classes: int,
        step_size: float = 0.35,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.pad_token = int(pad_token)
        self.pair_dim = int(pair_dim)
        self.state_dim = 2 * self.pair_dim
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.initial = nn.Parameter(torch.empty(pair_dim, 2))
        self.injection = nn.Parameter(torch.empty(num_tokens, pair_dim, 2))
        self.angle = nn.Parameter(torch.empty(num_tokens, pair_dim))
        self.gate = nn.Parameter(torch.empty(num_tokens, pair_dim))
        self.demographic_projection = nn.Linear(2, self.state_dim, bias=False)
        self.readout = nn.Linear(self.state_dim, 2 * num_classes)
        self.num_classes = int(num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.initial, std=0.05)
        nn.init.normal_(self.injection, std=0.05)
        nn.init.normal_(self.angle, std=0.08)
        nn.init.zeros_(self.gate)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def _normalize(self, state: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(state.flatten(1), dim=-1, keepdim=True).clamp_min(self.eps)
        return state * (math.sqrt(self.pair_dim) / norm)[:, None]

    def evolve(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None = None
    ) -> torch.Tensor:
        state = self.initial[None].expand(tokens.shape[0], -1, -1)
        if vector is not None:
            state = state + self.demographic_projection(vector[:, -2:]).reshape(
                tokens.shape[0], self.pair_dim, 2
            )
        state = self._normalize(state)
        for position in range(tokens.shape[1]):
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            theta = self.angle[token]
            cosine, sine = torch.cos(theta), torch.sin(theta)
            first, second = state[..., 0], state[..., 1]
            rotated = torch.stack(
                [cosine * first - sine * second, sine * first + cosine * second], dim=-1
            )
            gate = torch.sigmoid(self.gate[token])[..., None]
            candidate = self._normalize(
                state
                + self.step_size * torch.tanh(gate * (rotated - state) + self.injection[token])
            )
            state = torch.where(mask[:, position, None, None], candidate, state)
        return state

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return self.evolve(tokens, mask, vector).flatten(1)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        amplitude = self.readout(self.encode(**batch)).reshape(-1, self.num_classes, 2)
        return torch.log(amplitude.square().sum(dim=-1) + self.eps)


class RealPolarOperatorState(nn.Module):
    """Explicit real magnitude/angle state testing whether phase-like memory is sufficient."""

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
        self.num_classes = int(num_classes)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.initial_log_magnitude = nn.Parameter(torch.empty(state_dim))
        self.initial_phase = nn.Parameter(torch.empty(state_dim))
        self.radial_injection = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.phase_injection = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.radial_left = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.radial_right = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.phase_left = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.phase_right = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.demographic_magnitude = nn.Parameter(torch.empty(2, state_dim))
        self.demographic_phase = nn.Parameter(torch.empty(2, state_dim))
        self.readout_real = nn.Parameter(torch.empty(num_classes, state_dim))
        self.readout_imag = nn.Parameter(torch.empty(num_classes, state_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.05)
        nn.init.normal_(self.readout_real, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.readout_imag, std=1.0 / math.sqrt(self.state_dim))

    def evolve_polar(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_magnitude = self.initial_log_magnitude[None].expand(tokens.shape[0], -1)
        phase = self.initial_phase[None].expand(tokens.shape[0], -1)
        if vector is not None:
            log_magnitude = log_magnitude + vector[:, -2:] @ self.demographic_magnitude
            phase = phase + vector[:, -2:] @ self.demographic_phase
        for position in range(tokens.shape[1]):
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            radial_projection = torch.einsum("bs,bsr->br", log_magnitude, self.radial_right[token])
            phase_projection = torch.einsum("bs,bsr->br", torch.sin(phase), self.phase_right[token])
            radial_delta = self.radial_injection[token] + torch.einsum(
                "bsr,br->bs", self.radial_left[token], radial_projection
            )
            phase_delta = self.phase_injection[token] + torch.einsum(
                "bsr,br->bs", self.phase_left[token], phase_projection
            )
            candidate_magnitude = log_magnitude + self.step_size * torch.tanh(radial_delta)
            candidate_phase = phase + math.pi * self.step_size * torch.tanh(phase_delta)
            active = mask[:, position, None]
            log_magnitude = torch.where(active, candidate_magnitude, log_magnitude)
            phase = torch.where(active, candidate_phase, phase)
        magnitude = torch.nn.functional.softplus(log_magnitude) + self.eps
        magnitude = magnitude * (
            math.sqrt(self.state_dim)
            / torch.linalg.vector_norm(magnitude, dim=-1, keepdim=True).clamp_min(self.eps)
        )
        return magnitude, phase

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        magnitude, phase = self.evolve_polar(tokens, mask, vector)
        return torch.cat([magnitude, torch.sin(phase), torch.cos(phase)], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        magnitude, phase = self.evolve_polar(tokens, mask, vector)
        real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
        amplitude_real = real @ self.readout_real.T + imag @ self.readout_imag.T
        amplitude_imag = imag @ self.readout_real.T - real @ self.readout_imag.T
        return torch.log(amplitude_real.square() + amplitude_imag.square() + self.eps)
