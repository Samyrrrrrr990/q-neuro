"""Causal mechanism controls for complex and noncommutative state updates."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from qneuro.models.baselines import DenseRealMatrixRecurrence
from qneuro.models.operators import (
    ComplexOperatorState,
    RealOperatorState,
    _bounded_norm_complex,
)


class CommutingComplexOperatorState(nn.Module):
    """Complex diagonal linear operators that commute in one fixed coordinate basis."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        state_dim: int,
        num_classes: int,
        step_size: float = 0.35,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.pad_token = int(pad_token)
        self.state_dim = int(state_dim)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.initial_real = nn.Parameter(torch.empty(state_dim))
        self.initial_imag = nn.Parameter(torch.empty(state_dim))
        self.injection_real = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.injection_imag = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.diagonal_real = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.diagonal_imag = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.demographic_real = nn.Parameter(torch.empty(2, state_dim))
        self.demographic_imag = nn.Parameter(torch.empty(2, state_dim))
        self.readout_real = nn.Parameter(torch.empty(num_classes, state_dim))
        self.readout_imag = nn.Parameter(torch.empty(num_classes, state_dim))
        self.reset_parameters()

    @staticmethod
    def _complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.complex(real, imag)

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.045)
        nn.init.normal_(self.readout_real, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.readout_imag, std=1.0 / math.sqrt(self.state_dim))

    def evolve(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self._complex(self.initial_real, self.initial_imag)[None].expand(
            tokens.shape[0], -1
        )
        if vector is not None:
            demographic = self._complex(self.demographic_real, self.demographic_imag)
            state = state + vector[:, -2:].to(demographic.dtype) @ demographic
        state = _bounded_norm_complex(state)
        for position in range(tokens.shape[1]):
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            diagonal = self._complex(self.diagonal_real[token], self.diagonal_imag[token])
            injection = self._complex(self.injection_real[token], self.injection_imag[token])
            candidate = _bounded_norm_complex(
                state + self.step_size * torch.tanh(injection + diagonal * state)
            )
            state = torch.where(mask[:, position, None], candidate, state)
        return state

    def encode(self, **batch: torch.Tensor) -> torch.Tensor:
        state = self.evolve(batch["tokens"], batch["mask"], batch.get("vector"))
        return torch.cat([state.real, state.imag], dim=-1)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        state = self.evolve(batch["tokens"], batch["mask"], batch.get("vector"))
        readout = self._complex(self.readout_real, self.readout_imag)
        amplitude = torch.einsum("bs,ds->bd", state, readout.conj())
        return torch.log(torch.abs(amplitude).square() + self.eps)

    @torch.no_grad()
    def commutator_norm(self, token_a: int, token_b: int) -> float:
        del token_a, token_b
        return 0.0


class CommutatorPenaltyComplexOperator(ComplexOperatorState):
    """Complex operator with a differentiable penalty on order-marker commutators."""

    def __init__(self, *args: object, penalty_strength: float = 1e-3, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.penalty_strength = float(penalty_strength)

    def _linear_operator(self, token: int) -> torch.Tensor:
        identity = torch.eye(
            self.state_dim,
            device=self.left_real.device,
            dtype=torch.complex64,
        )
        left = self._complex(self.left_real[token], self.left_imag[token])
        right = self._complex(self.right_real[token], self.right_imag[token])
        return identity + self.step_size * left @ right.conj().T

    def auxiliary_loss(self) -> torch.Tensor:
        penalties: list[torch.Tensor] = []
        for token_a in range(0, 8, 2):
            token_b = token_a + 1
            operator_a = self._linear_operator(token_a)
            operator_b = self._linear_operator(token_b)
            commutator = operator_a @ operator_b - operator_b @ operator_a
            scale = torch.linalg.matrix_norm(operator_a) * torch.linalg.matrix_norm(operator_b)
            penalties.append(
                torch.linalg.matrix_norm(commutator).square() / scale.square().clamp_min(1e-8)
            )
        return self.penalty_strength * torch.stack(penalties).real.mean()


class NoncommutativeRealOperator(DenseRealMatrixRecurrence):
    """Unrestricted real transition with a bounded incentive for marker noncommutativity."""

    def __init__(self, *args: object, incentive_strength: float = 1e-4, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.incentive_strength = float(incentive_strength)

    def auxiliary_loss(self) -> torch.Tensor:
        scores: list[torch.Tensor] = []
        for token_a in range(0, 8, 2):
            token_b = token_a + 1
            operator_a = self.transition[token_a]
            operator_b = self.transition[token_b]
            commutator = operator_a @ operator_b - operator_b @ operator_a
            numerator = torch.linalg.matrix_norm(commutator).square()
            denominator = (
                torch.linalg.matrix_norm(operator_a).square()
                * torch.linalg.matrix_norm(operator_b).square()
            )
            scores.append(numerator / (numerator + denominator + 1e-8))
        return -self.incentive_strength * torch.stack(scores).mean()

    @torch.no_grad()
    def commutator_norm(self, token_a: int, token_b: int) -> float:
        first = self.transition[token_a]
        second = self.transition[token_b]
        return float(torch.linalg.matrix_norm(first @ second - second @ first).cpu())


class PhaseDestroyedTrainingOperator(ComplexOperatorState):
    """Reset phase to zero after every update during both training and inference."""

    def _intervene_state(self, state: torch.Tensor, position: int) -> torch.Tensor:
        del position
        return torch.complex(torch.abs(state), torch.zeros_like(state.real))


class MagnitudeDestroyedOperator(ComplexOperatorState):
    """Keep only component-wise phase after every state update."""

    def _intervene_state(self, state: torch.Tensor, position: int) -> torch.Tensor:
        del position
        unit = state / torch.abs(state).clamp_min(self.eps)
        return unit


class NoConjugationComplexOperator(ComplexOperatorState):
    """Remove complex conjugation from transition projection and measurement."""

    def evolve(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        state = self._complex(self.initial_real, self.initial_imag)[None].expand(
            tokens.shape[0], -1
        )
        if vector is not None:
            demographic = self._complex(self.demographic_real, self.demographic_imag)
            state = state + vector[:, -2:].to(demographic.dtype) @ demographic
        state = _bounded_norm_complex(state)
        for position in range(tokens.shape[1]):
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            injection = self._complex(self.injection_real[token], self.injection_imag[token])
            left = self._complex(self.left_real[token], self.left_imag[token])
            right = self._complex(self.right_real[token], self.right_imag[token])
            projection = torch.einsum("bs,bsr->br", state, right)
            delta = injection + torch.einsum("bsr,br->bs", left, projection)
            candidate = _bounded_norm_complex(state + self.step_size * torch.tanh(delta))
            state = torch.where(mask[:, position, None], candidate, state)
        return state

    def measure(self, state: torch.Tensor, phase_mode: str = "learned") -> torch.Tensor:
        if phase_mode != "learned":
            return super().measure(state, phase_mode=phase_mode)
        readout = self._complex(self.readout_real, self.readout_imag)
        amplitude = torch.einsum("bs,ds->bd", state, readout)
        return torch.log(torch.abs(amplitude).square() + self.eps)


class FixedRandomComplexOperator(ComplexOperatorState):
    """Freeze randomly initialized state dynamics and learn only the measurement."""

    def __init__(self, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("readout_"))


class FrozenReadoutComplexOperator(ComplexOperatorState):
    """Learn dynamics while holding the random complex readout fixed."""

    def __init__(self, *args: object, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.readout_real.requires_grad_(False)
        self.readout_imag.requires_grad_(False)


class AmbiguityAwareRealOperator(RealOperatorState):
    """Real operator with a positive-evidence (Dirichlet-mean) prediction head."""

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        evidence = F.softplus(self.readout(self.evolve(tokens, mask, vector))) + 1.0
        return torch.log(evidence / evidence.sum(dim=-1, keepdim=True))


class AmbiguityAwareComplexOperator(ComplexOperatorState):
    """Complex dynamics with a positive-evidence (Dirichlet-mean) prediction head."""

    def measure(self, state: torch.Tensor, phase_mode: str = "learned") -> torch.Tensor:
        logits = super().measure(state, phase_mode=phase_mode)
        evidence = F.softplus(logits) + 1.0
        return torch.log(evidence / evidence.sum(dim=-1, keepdim=True))


class FixedTwoStateAttractor(nn.Module):
    """Parallel real hypothesis states combined as an equal-weight predictive mixture."""

    def __init__(self, num_tokens: int, state_dim: int, num_classes: int, step_size: float = 0.35):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.step_size = float(step_size)
        self.initial = nn.Parameter(torch.empty(2, state_dim))
        self.embedding = nn.Parameter(torch.empty(num_tokens, 2, state_dim))
        self.gate = nn.Parameter(torch.empty(num_tokens, 2, state_dim))
        self.demographic = nn.Linear(2, 2 * state_dim, bias=False)
        self.readout = nn.Parameter(torch.empty(2, num_classes, state_dim))
        self.bias = nn.Parameter(torch.zeros(2, num_classes))
        nn.init.normal_(self.initial, std=0.05)
        nn.init.normal_(self.embedding, std=0.06)
        nn.init.normal_(self.gate, std=0.04)
        nn.init.normal_(self.readout, std=1.0 / math.sqrt(state_dim))

    def evolve(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None = None
    ) -> torch.Tensor:
        state = self.initial[None].expand(tokens.shape[0], -1, -1)
        if vector is not None:
            state = state + self.demographic(vector[:, -2:]).reshape(-1, 2, self.state_dim)
        for position in range(tokens.shape[1]):
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            gate = torch.sigmoid(self.gate[token])
            candidate = torch.tanh(state + self.step_size * gate * self.embedding[token])
            state = torch.where(mask[:, position, None, None], candidate, state)
        return state

    def encode(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.evolve(batch["tokens"], batch["mask"], batch.get("vector")).flatten(1)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        state = self.evolve(batch["tokens"], batch["mask"], batch.get("vector"))
        component_logits = torch.einsum("bks,kcs->bkc", state, self.readout) + self.bias[None]
        component_log_probabilities = F.log_softmax(component_logits, dim=-1)
        return torch.logsumexp(component_log_probabilities, dim=1) - math.log(2.0)
