"""Mechanism-level baselines and Q-Neuro dynamics used after Experiment Zero."""

from __future__ import annotations

import math
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from qneuro.models.operators import _bounded_norm_complex, _bounded_norm_real


class LogisticEvidence(nn.Module):
    """Linear same-evidence control with explicit missingness indicators."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.readout = nn.Linear(input_dim, num_classes)

    def encode(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return vector

    def forward(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.readout(vector)


class ComplexEvidenceMLP(nn.Module):
    """Feed-forward complex control with phase-sensitive amplitude measurement."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.input_real = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.input_imag = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.bias_real = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_imag = nn.Parameter(torch.zeros(hidden_dim))
        self.readout_real = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.readout_imag = nn.Parameter(torch.empty(num_classes, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for value in (self.input_real, self.input_imag):
            nn.init.xavier_uniform_(value)
        for value in (self.readout_real, self.readout_imag):
            nn.init.normal_(value, std=1.0 / math.sqrt(value.shape[-1]))

    @staticmethod
    def _complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.complex(real, imag)

    def complex_state(self, vector: torch.Tensor) -> torch.Tensor:
        real = F.linear(vector, self.input_real, self.bias_real)
        imag = F.linear(vector, self.input_imag, self.bias_imag)
        return self._complex(torch.tanh(real), torch.tanh(imag))

    def encode(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        state = self.complex_state(vector)
        return torch.cat([state.real, state.imag], dim=-1)

    def forward(
        self, vector: torch.Tensor, phase_mode: str = "learned", **_: torch.Tensor
    ) -> torch.Tensor:
        state = self.complex_state(vector)
        if phase_mode == "zero":
            state = torch.complex(torch.abs(state), torch.zeros_like(state.real))
        elif phase_mode == "randomized":
            phase = 2.0 * math.pi * torch.rand_like(state.real)
            state = state * torch.complex(torch.cos(phase), torch.sin(phase))
        elif phase_mode != "learned":
            raise ValueError(f"unknown phase mode: {phase_mode}")
        readout = self._complex(self.readout_real, self.readout_imag)
        amplitude = torch.einsum("bh,dh->bd", state, readout.conj())
        return torch.log(torch.abs(amplitude).square() + self.eps)


class DiagonalStateSpace(nn.Module):
    """Compact diagonal recurrent state-space baseline."""

    def __init__(self, num_tokens: int, state_dim: int, num_classes: int):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.token_input = nn.Embedding(num_tokens + 1, state_dim, padding_idx=num_tokens)
        self.logit_decay = nn.Parameter(torch.zeros(state_dim))
        self.initial_state = nn.Parameter(torch.zeros(state_dim))
        self.demographic_projection = nn.Linear(2, state_dim, bias=False)
        self.readout = nn.Linear(state_dim, num_classes)
        nn.init.normal_(self.token_input.weight, std=0.08)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        state = self.initial_state.expand(tokens.shape[0], -1)
        if vector is not None:
            state = state + self.demographic_projection(vector[:, -2:])
        decay = torch.sigmoid(self.logit_decay)
        for position in range(tokens.shape[1]):
            candidate = torch.tanh(decay * state + self.token_input(tokens[:, position]))
            state = torch.where(mask[:, position, None], candidate, state)
        return state

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))


class ModernHopfieldMemory(nn.Module):
    """Single-retrieval modern-Hopfield-style associative diagnosis baseline."""

    def __init__(self, num_tokens: int, state_dim: int, num_classes: int):
        super().__init__()
        self.state_dim = int(state_dim)
        self.token_embedding = nn.Embedding(num_tokens + 1, state_dim, padding_idx=num_tokens)
        self.disease_queries = nn.Parameter(torch.empty(num_classes, state_dim))
        self.context_projection = nn.Linear(2, state_dim, bias=False)
        self.retrieval_projection = nn.Linear(state_dim, state_dim)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.normal_(self.token_embedding.weight, std=0.08)
        nn.init.normal_(self.disease_queries, std=1.0 / math.sqrt(state_dim))

    def retrieve(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None
    ) -> torch.Tensor:
        evidence = self.token_embedding(tokens)
        scores = torch.einsum("btd,cd->bct", evidence, self.disease_queries) / math.sqrt(
            self.state_dim
        )
        scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        retrieved = torch.einsum("bct,btd->bcd", weights, evidence)
        state = retrieved + self.disease_queries[None]
        if vector is not None:
            state = state + self.context_projection(vector[:, -2:])[:, None, :]
        return torch.tanh(self.retrieval_projection(state))

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return self.retrieve(tokens, mask, vector).mean(dim=1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        state = self.retrieve(tokens, mask, vector)
        return torch.einsum("bcd,cd->bc", state, self.disease_queries) + self.bias


class EvidenceGraphNetwork(nn.Module):
    """Shared-message GNN over the declared NeuroWorld factor graph."""

    def __init__(self, state_dim: int, num_classes: int, steps: int = 3):
        super().__init__()
        self.steps = int(steps)
        self.finding_embedding = nn.Parameter(torch.empty(40, state_dim))
        self.status_embedding = nn.Embedding(3, state_dim)
        self.self_update = nn.Linear(state_dim, state_dim)
        self.message_update = nn.Linear(state_dim, state_dim, bias=False)
        self.context_projection = nn.Linear(2, state_dim, bias=False)
        self.readout = nn.Linear(state_dim, num_classes)
        self.register_buffer("adjacency", self._factor_adjacency())
        nn.init.normal_(self.finding_embedding, std=0.08)

    @staticmethod
    def _factor_adjacency() -> torch.Tensor:
        groups = (range(8), range(8, 18), range(18, 26), range(26, 32), range(32, 40))
        adjacency = torch.eye(40)
        for group in groups:
            indices = torch.tensor(list(group))
            adjacency[indices[:, None], indices[None, :]] = 1.0
        for offset in range(8):
            indices = torch.tensor(
                [offset, 8 + offset % 10, 18 + offset, 26 + offset % 6, 32 + offset]
            )
            adjacency[indices[:, None], indices[None, :]] = 1.0
        return adjacency / adjacency.sum(dim=-1, keepdim=True)

    def encode(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        values = vector[:, :40]
        observed = vector[:, 40:80].bool()
        status = torch.where(observed, torch.where(values > 0, 2, 1), 0).long()
        state = self.finding_embedding[None] + self.status_embedding(status)
        for _ in range(self.steps):
            message = torch.einsum("ij,bjd->bid", self.adjacency, state)
            state = F.gelu(self.self_update(state) + self.message_update(message))
        pooled = state.mean(dim=1) + self.context_projection(vector[:, -2:])
        return pooled

    def forward(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(vector))


class CoupledTensorState(nn.Module):
    """Low-rank multiplicative control for inseparable evidence-factor interactions."""

    def __init__(self, input_dim: int, factor_dim: int, num_classes: int):
        super().__init__()
        self.left = nn.Linear(input_dim, factor_dim)
        self.right = nn.Linear(input_dim, factor_dim)
        self.diagonal = nn.Linear(input_dim, factor_dim)
        self.readout = nn.Linear(2 * factor_dim, num_classes)

    def encode(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        left = torch.tanh(self.left(vector))
        right = torch.tanh(self.right(vector))
        coupled = left * right
        return torch.cat([coupled, torch.tanh(self.diagonal(vector))], dim=-1)

    def forward(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(vector))


class EnergyAttractorState(nn.Module):
    """Evidence-forced descent toward learned disease attractors."""

    def __init__(
        self,
        num_tokens: int,
        state_dim: int,
        num_classes: int,
        steps: int = 6,
        step_size: float = 0.35,
        adaptive: bool = False,
        ponder_cost: float = 0.01,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.num_classes = int(num_classes)
        self.steps = int(steps)
        self.step_size = float(step_size)
        self.adaptive = bool(adaptive)
        self.ponder_cost = float(ponder_cost)
        self.initial_state = nn.Parameter(torch.empty(state_dim))
        self.evidence_embedding = nn.Embedding(num_tokens + 1, state_dim, padding_idx=num_tokens)
        self.demographic_projection = nn.Linear(2, state_dim, bias=False)
        self.attractors = nn.Parameter(torch.empty(num_classes, state_dim))
        self.force_projection = nn.Linear(state_dim, state_dim, bias=False)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        if adaptive:
            self.halt_head = nn.Linear(3, 1)
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, -1.5)
        self._last_expected_steps: torch.Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.initial_state, std=0.05)
        nn.init.normal_(self.evidence_embedding.weight, std=0.07)
        nn.init.normal_(self.attractors, std=0.10)

    def _force(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None
    ) -> torch.Tensor:
        evidence = self.evidence_embedding(tokens) * mask[..., None]
        force = evidence.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        force = self.force_projection(force)
        if vector is not None:
            force = force + self.demographic_projection(vector[:, -2:])
        return force

    def _logits(self, state: torch.Tensor) -> torch.Tensor:
        temperature = F.softplus(self.log_temperature) + 0.05
        return -torch.cdist(state, self.attractors).square() / temperature

    def trajectory(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        force = self._force(tokens, mask, vector)
        state = _bounded_norm_real(self.initial_state[None] + force)
        states = [state]
        logits = [self._logits(state)]
        temperature = F.softplus(self.log_temperature) + 0.05
        for _ in range(self.steps - 1):
            distance = torch.cdist(state, self.attractors).square()
            weights = torch.softmax(-distance / temperature, dim=-1)
            target = weights @ self.attractors
            candidate = state + self.step_size * torch.tanh(force + target - state)
            state = _bounded_norm_real(candidate)
            states.append(state)
            logits.append(self._logits(state))
        return states, logits

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        states, _ = self.trajectory(tokens, mask, vector)
        return states[-1]

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        states, logits = self.trajectory(tokens, mask, vector)
        if not self.adaptive:
            self._last_expected_steps = torch.full(
                (tokens.shape[0],), float(self.steps), device=tokens.device
            )
            return logits[-1]
        remaining = torch.ones(tokens.shape[0], device=tokens.device)
        output = torch.zeros_like(logits[0])
        expected_steps = torch.zeros_like(remaining)
        previous = states[0]
        for index, (state, step_logits) in enumerate(zip(states, logits, strict=True)):
            velocity = torch.linalg.vector_norm(state - previous, dim=-1) / math.sqrt(
                self.state_dim
            )
            probability = torch.softmax(step_logits, dim=-1)
            entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(dim=-1)
            separation = probability.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
            if index == self.steps - 1:
                allocation = remaining
            else:
                halt_probability = torch.sigmoid(
                    self.halt_head(torch.stack([velocity, entropy, separation], dim=-1)).squeeze(-1)
                )
                allocation = remaining * halt_probability
            output = output + allocation[:, None] * step_logits
            expected_steps = expected_steps + allocation * float(index + 1)
            remaining = remaining - allocation
            previous = state
        self._last_expected_steps = expected_steps
        return output

    def auxiliary_loss(self) -> torch.Tensor:
        if not self.adaptive or self._last_expected_steps is None:
            return torch.zeros((), device=self.initial_state.device)
        return self.ponder_cost * self._last_expected_steps.mean() / self.steps

    @torch.no_grad()
    def trajectory_diagnostics(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        states, logits = self.trajectory(tokens, mask, vector)
        velocities = torch.stack(
            [
                torch.zeros(states[0].shape[0], device=states[0].device),
                *[
                    torch.linalg.vector_norm(current - previous, dim=-1) / math.sqrt(self.state_dim)
                    for previous, current in pairwise(states)
                ],
            ],
            dim=1,
        )
        entropy = torch.stack(
            [
                -(probability * torch.log(probability.clamp_min(1e-12))).sum(dim=-1)
                for probability in (torch.softmax(value, dim=-1) for value in logits)
            ],
            dim=1,
        )
        return {"velocity": velocities, "entropy": entropy}


class HamiltonianDissipativeState(nn.Module):
    """Complex low-rank Hermitian rotation with optional learned dissipation."""

    def __init__(
        self,
        num_tokens: int,
        state_dim: int,
        rank: int,
        num_classes: int,
        step_size: float,
        coherent: bool,
        dissipative: bool,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not coherent and not dissipative:
            raise ValueError("at least one dynamical component must be enabled")
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.step_size = float(step_size)
        self.coherent = bool(coherent)
        self.dissipative = bool(dissipative)
        self.eps = float(eps)
        self.initial_real = nn.Parameter(torch.empty(state_dim))
        self.initial_imag = nn.Parameter(torch.empty(state_dim))
        self.injection_real = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.injection_imag = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.demographic_real = nn.Parameter(torch.empty(2, state_dim))
        self.demographic_imag = nn.Parameter(torch.empty(2, state_dim))
        if coherent:
            self.hamiltonian_diagonal = nn.Parameter(torch.empty(num_tokens, state_dim))
            self.coupling_real = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
            self.coupling_imag = nn.Parameter(torch.empty(num_tokens, state_dim, rank))
        if dissipative:
            self.log_damping = nn.Parameter(torch.full((num_tokens, state_dim), -2.0))
        self.readout_real = nn.Parameter(torch.empty(num_classes, state_dim))
        self.readout_imag = nn.Parameter(torch.empty(num_classes, state_dim))
        self.reset_parameters()

    @staticmethod
    def _complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.complex(real, imag)

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "log_damping" in name:
                continue
            nn.init.normal_(parameter, std=0.05)
        nn.init.normal_(self.readout_real, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.readout_imag, std=1.0 / math.sqrt(self.state_dim))

    def evolve(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None
    ) -> torch.Tensor:
        state = self._complex(self.initial_real, self.initial_imag).expand(tokens.shape[0], -1)
        if vector is not None:
            demographic = self._complex(self.demographic_real, self.demographic_imag)
            state = state + vector[:, -2:].to(demographic.dtype) @ demographic
        state = _bounded_norm_complex(state)
        for position in range(tokens.shape[1]):
            active = mask[:, position]
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            delta = self._complex(self.injection_real[token], self.injection_imag[token])
            if self.coherent:
                coupling = self._complex(self.coupling_real[token], self.coupling_imag[token])
                projected = torch.einsum("bsd,bs->bd", coupling.conj(), state)
                hamiltonian_state = self.hamiltonian_diagonal[token] * state + torch.einsum(
                    "bsd,bd->bs", coupling, projected
                )
                delta = delta - 1j * hamiltonian_state
            if self.dissipative:
                delta = delta - F.softplus(self.log_damping[token]) * state
            candidate = _bounded_norm_complex(state + self.step_size * delta)
            state = torch.where(active[:, None], candidate, state)
        return state

    def measure(self, state: torch.Tensor) -> torch.Tensor:
        readout = self._complex(self.readout_real, self.readout_imag)
        amplitude = torch.einsum("bs,ds->bd", state, readout.conj())
        return torch.log(torch.abs(amplitude).square() + self.eps)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        state = self.evolve(tokens, mask, vector)
        return torch.cat([state.real, state.imag], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        return self.measure(self.evolve(tokens, mask, vector))


class DiagnosticDensityDynamics(nn.Module):
    """Low-rank density-factor evolution preserving Hermiticity, PSD, and unit trace."""

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        factor_rank: int,
        operator_rank: int,
        step_size: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.num_classes = int(num_classes)
        self.factor_rank = int(factor_rank)
        self.operator_rank = int(operator_rank)
        self.step_size = float(step_size)
        self.eps = float(eps)
        shape = (num_classes, factor_rank)
        self.initial_real = nn.Parameter(torch.empty(shape))
        self.initial_imag = nn.Parameter(torch.empty(shape))
        self.context_real = nn.Parameter(torch.empty(2, *shape))
        self.context_imag = nn.Parameter(torch.empty(2, *shape))
        self.injection_real = nn.Parameter(torch.empty(num_tokens, *shape))
        self.injection_imag = nn.Parameter(torch.empty(num_tokens, *shape))
        self.hamiltonian_diagonal = nn.Parameter(torch.empty(num_tokens, num_classes))
        coupling_shape = (num_tokens, num_classes, operator_rank)
        self.coupling_real = nn.Parameter(torch.empty(coupling_shape))
        self.coupling_imag = nn.Parameter(torch.empty(coupling_shape))
        self.log_damping = nn.Parameter(torch.full((num_tokens, num_classes), -2.0))
        self.reset_parameters()

    @staticmethod
    def _complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.complex(real, imag)

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if "log_damping" in name:
                continue
            nn.init.normal_(parameter, std=0.05)

    @staticmethod
    def _normalize_factor(factor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = torch.sqrt(torch.abs(factor).square().sum(dim=(-2, -1), keepdim=True)).clamp_min(eps)
        return factor / norm

    def evolve_factor(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None
    ) -> torch.Tensor:
        factor = self._complex(self.initial_real, self.initial_imag).expand(tokens.shape[0], -1, -1)
        if vector is not None:
            context = self._complex(self.context_real, self.context_imag)
            factor = factor + torch.einsum("bc,cdk->bdk", vector[:, -2:].to(context.dtype), context)
        factor = self._normalize_factor(factor)
        for position in range(tokens.shape[1]):
            active = mask[:, position]
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            coupling = self._complex(self.coupling_real[token], self.coupling_imag[token])
            projected = torch.einsum("bdr,bdk->brk", coupling.conj(), factor)
            hamiltonian_factor = self.hamiltonian_diagonal[token][
                ..., None
            ] * factor + torch.einsum("bdr,brk->bdk", coupling, projected)
            injection = self._complex(self.injection_real[token], self.injection_imag[token])
            damping = F.softplus(self.log_damping[token])[..., None]
            delta = -1j * hamiltonian_factor - damping * factor + injection
            candidate = self._normalize_factor(factor + self.step_size * delta)
            factor = torch.where(active[:, None, None], candidate, factor)
        return factor

    @staticmethod
    def density_matrix(factor: torch.Tensor) -> torch.Tensor:
        density = torch.einsum("bdk,bek->bde", factor, factor.conj())
        trace = density.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return density / trace[:, None]

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        density = self.density_matrix(self.evolve_factor(tokens, mask, vector))
        return torch.cat([density.real.flatten(1), density.imag.flatten(1)], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        factor = self.evolve_factor(tokens, mask, vector)
        probability = torch.abs(factor).square().sum(dim=-1).clamp_min(self.eps)
        return torch.log(probability)

    @torch.no_grad()
    def density_diagnostics(
        self, tokens: torch.Tensor, mask: torch.Tensor, vector: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        density = self.density_matrix(self.evolve_factor(tokens, mask, vector))
        diagonal = torch.diagonal(density, dim1=-2, dim2=-1)
        off_diagonal = density - torch.diag_embed(diagonal)
        return {
            "trace": diagonal.real.sum(dim=-1),
            "hermiticity_error": torch.linalg.matrix_norm(
                density - density.conj().transpose(-2, -1)
            ),
            "off_diagonal_coherence": torch.linalg.matrix_norm(off_diagonal),
        }
