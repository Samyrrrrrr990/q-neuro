"""Real and complex low-rank evidence-operator state models."""

from __future__ import annotations

import math

import torch
from torch import nn


def _bounded_norm_real(state: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target = math.sqrt(state.shape[-1])
    norm = torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(eps)
    return target * state / norm


def _bounded_norm_complex(state: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target = math.sqrt(state.shape[-1])
    norm = torch.sqrt(torch.sum(torch.abs(state).square(), dim=-1, keepdim=True)).clamp_min(eps)
    return target * state / norm


class RealOperatorState(nn.Module):
    """Ordered state evolution under token-specific low-rank real operators."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        state_dim: int,
        rank: int,
        num_classes: int,
        step_size: float = 0.35,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.pad_token = pad_token
        self.state_dim = state_dim
        self.rank = rank
        self.step_size = float(step_size)
        self.initial_state = nn.Parameter(torch.empty(state_dim))
        self.injection = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.left = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.right = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        self.demographic_projection = nn.Linear(2, state_dim, bias=False)
        self.readout = nn.Linear(state_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.initial_state, std=0.05)
        nn.init.normal_(self.injection, std=0.06)
        nn.init.normal_(self.left, std=0.06)
        nn.init.normal_(self.right, std=0.06)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def evolve(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = tokens.shape
        state = self.initial_state.expand(batch_size, -1)
        if vector is not None:
            state = state + self.demographic_projection(vector[:, -2:])
        state = _bounded_norm_real(state)
        for position in range(sequence_length):
            active = mask[:, position]
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            injection = self.injection[token]
            left = self.left[token]
            right = self.right[token]
            projection = torch.einsum("bs,bsr->br", state, right)
            delta = injection + torch.einsum("bsr,br->bs", left, projection)
            candidate = _bounded_norm_real(state + self.step_size * torch.tanh(delta))
            state = torch.where(active.unsqueeze(-1), candidate, state)
        return state

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return self.readout(self.evolve(tokens, mask, vector))

    @torch.no_grad()
    def commutator_norm(self, token_a: int, token_b: int) -> float:
        identity = torch.eye(self.state_dim, device=self.left.device, dtype=self.left.dtype)
        operator_a = identity + self.step_size * self.left[token_a] @ self.right[token_a].T
        operator_b = identity + self.step_size * self.left[token_b] @ self.right[token_b].T
        commutator = operator_a @ operator_b - operator_b @ operator_a
        return float(torch.linalg.matrix_norm(commutator).cpu())


class ComplexOperatorState(nn.Module):
    """Complex operator model whose phase affects evolution and Born-style measurement."""

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
        self.num_tokens = num_tokens
        self.pad_token = pad_token
        self.state_dim = state_dim
        self.rank = rank
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

    @staticmethod
    def _complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.complex(real, imag)

    def evolve(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length = tokens.shape
        initial = self._complex(self.initial_real, self.initial_imag)
        state = initial.expand(batch_size, -1)
        if vector is not None:
            demographic = self._complex(self.demographic_real, self.demographic_imag)
            state = state + vector[:, -2:].to(demographic.dtype) @ demographic
        state = _bounded_norm_complex(state)
        for position in range(sequence_length):
            active = mask[:, position]
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            injection = self._complex(self.injection_real[token], self.injection_imag[token])
            left = self._complex(self.left_real[token], self.left_imag[token])
            right = self._complex(self.right_real[token], self.right_imag[token])
            projection = torch.einsum("bs,bsr->br", state, right.conj())
            delta = injection + torch.einsum("bsr,br->bs", left, projection)
            candidate = _bounded_norm_complex(state + self.step_size * torch.tanh(delta))
            state = torch.where(active.unsqueeze(-1), candidate, state)
        return state

    def measure(self, state: torch.Tensor, phase_mode: str = "learned") -> torch.Tensor:
        if phase_mode == "zero":
            state = torch.complex(torch.abs(state), torch.zeros_like(state.real))
        elif phase_mode == "randomized":
            phases = 2.0 * math.pi * torch.rand_like(state.real)
            state = state * torch.complex(torch.cos(phases), torch.sin(phases))
        elif phase_mode != "learned":
            raise ValueError(f"unknown phase_mode: {phase_mode}")
        readout = self._complex(self.readout_real, self.readout_imag)
        amplitude = torch.einsum("bs,ds->bd", state, readout.conj())
        return torch.log(torch.abs(amplitude).square() + self.eps)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        phase_mode: str = "learned",
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return self.measure(self.evolve(tokens, mask, vector), phase_mode=phase_mode)

    @torch.no_grad()
    def probabilities(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.softmax(self.forward(tokens=tokens, mask=mask, vector=vector), dim=-1)

    @torch.no_grad()
    def commutator_norm(self, token_a: int, token_b: int) -> float:
        dtype = torch.complex64
        identity = torch.eye(self.state_dim, device=self.left_real.device, dtype=dtype)
        left_a = self._complex(self.left_real[token_a], self.left_imag[token_a])
        right_a = self._complex(self.right_real[token_a], self.right_imag[token_a])
        left_b = self._complex(self.left_real[token_b], self.left_imag[token_b])
        right_b = self._complex(self.right_real[token_b], self.right_imag[token_b])
        operator_a = identity + self.step_size * left_a @ right_a.conj().T
        operator_b = identity + self.step_size * left_b @ right_b.conj().T
        commutator = operator_a @ operator_b - operator_b @ operator_a
        return float(torch.linalg.matrix_norm(commutator).real.cpu())
