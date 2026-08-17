"""Conventional parameter-budgeted baselines."""

from __future__ import annotations

import torch
from torch import nn


class EvidenceMLP(nn.Module):
    """One-hidden-layer baseline over values, missingness, and demographics."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.readout = nn.Linear(hidden_dim, num_classes)

    def encode(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.activation(self.input_layer(vector))

    def forward(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(vector))


class TinyTransformer(nn.Module):
    """Small ordered-sequence baseline with masked mean pooling."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        num_classes: int,
        model_dim: int,
        max_length: int,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
    ):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.pad_token = pad_token
        self.token_embedding = nn.Embedding(num_tokens + 1, model_dim, padding_idx=pad_token)
        self.position_embedding = nn.Parameter(torch.zeros(max_length, model_dim))
        self.demographic_projection = nn.Linear(2, model_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 2 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1, enable_nested_tensor=False
        )
        self.readout = nn.Linear(model_dim, num_classes)
        nn.init.normal_(self.position_embedding, std=0.02)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.position_embedding.shape[0]:
            raise ValueError("sequence exceeds configured max_length")
        context = self.demographic_projection(vector[:, -2:]).unsqueeze(1)
        hidden = self.token_embedding(tokens) + self.position_embedding[:length] + context
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        return self.readout(self.encode(tokens, mask, vector, **kwargs))


class TinyGRU(nn.Module):
    """Validation-tunable recurrent baseline with demographic initial context."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        num_classes: int,
        hidden_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embedding_dim, padding_idx=pad_token)
        self.demographic_projection = nn.Linear(2, hidden_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        initial = torch.tanh(self.demographic_projection(vector[:, -2:])).unsqueeze(0)
        _, final = self.gru(packed, initial)
        return final.squeeze(0)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor:
        return self.readout(self.encode(tokens, mask, vector, **kwargs))


class TinyRNN(nn.Module):
    """Vanilla tanh recurrence with the same evidence and demographic inputs."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        num_classes: int,
        hidden_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embedding_dim, padding_idx=pad_token)
        self.demographic_projection = nn.Linear(2, hidden_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, nonlinearity="tanh", batch_first=True)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(tokens), lengths, batch_first=True, enforce_sorted=False
        )
        initial = torch.tanh(self.demographic_projection(vector[:, -2:])).unsqueeze(0)
        _, final = self.rnn(packed, initial)
        return final.squeeze(0)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))


class TinyLSTM(nn.Module):
    """LSTM control with independent demographic hidden and cell initialization."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        num_classes: int,
        hidden_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embedding_dim, padding_idx=pad_token)
        self.hidden_projection = nn.Linear(2, hidden_dim)
        self.cell_projection = nn.Linear(2, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(tokens), lengths, batch_first=True, enforce_sorted=False
        )
        context = vector[:, -2:]
        hidden = torch.tanh(self.hidden_projection(context)).unsqueeze(0)
        cell = torch.tanh(self.cell_projection(context)).unsqueeze(0)
        _, (final, _) = self.lstm(packed, (hidden, cell))
        return final.squeeze(0)

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))


class CausalTransformer(nn.Module):
    """Compact causal Transformer using the last observed token for classification."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        num_classes: int,
        model_dim: int,
        max_length: int,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
    ):
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.token_embedding = nn.Embedding(num_tokens + 1, model_dim, padding_idx=pad_token)
        self.position_embedding = nn.Parameter(torch.zeros(max_length, model_dim))
        self.demographic_projection = nn.Linear(2, model_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim or 2 * model_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.readout = nn.Linear(model_dim, num_classes)
        nn.init.normal_(self.position_embedding, std=0.02)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.position_embedding.shape[0]:
            raise ValueError("sequence exceeds configured max_length")
        hidden = (
            self.token_embedding(tokens)
            + self.position_embedding[:length]
            + self.demographic_projection(vector[:, -2:]).unsqueeze(1)
        )
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~mask,
            is_causal=True,
        )
        final_index = mask.sum(dim=1).clamp_min(1) - 1
        return encoded[torch.arange(tokens.shape[0], device=tokens.device), final_index]

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))


class ResidualGatedRecurrent(nn.Module):
    """Single-gate residual recurrence controlling for stable incremental updates."""

    def __init__(
        self,
        num_tokens: int,
        pad_token: int,
        num_classes: int,
        hidden_dim: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, embedding_dim, padding_idx=pad_token)
        self.demographic_projection = nn.Linear(2, hidden_dim)
        self.input_gate = nn.Linear(embedding_dim, hidden_dim)
        self.state_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.input_candidate = nn.Linear(embedding_dim, hidden_dim)
        self.state_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.readout = nn.Linear(hidden_dim, num_classes)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.tanh(self.demographic_projection(vector[:, -2:]))
        embedded = self.embedding(tokens)
        for position in range(tokens.shape[1]):
            value = embedded[:, position]
            gate = torch.sigmoid(self.input_gate(value) + self.state_gate(state))
            candidate = torch.tanh(self.input_candidate(value) + self.state_candidate(state))
            updated = state + gate * (candidate - state)
            state = torch.where(mask[:, position, None], updated, state)
        return state

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))


class DenseRealMatrixRecurrence(nn.Module):
    """Token-conditioned unrestricted dense real state transition."""

    def __init__(
        self,
        num_tokens: int,
        num_classes: int,
        state_dim: int,
        step_size: float,
        readout_dim: int | None = None,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.state_dim = int(state_dim)
        self.step_size = float(step_size)
        self.initial_state = nn.Parameter(torch.empty(state_dim))
        self.injection = nn.Parameter(torch.empty(num_tokens, state_dim))
        self.transition = nn.Parameter(torch.empty(num_tokens, state_dim, state_dim))
        self.demographic_projection = nn.Linear(2, state_dim, bias=False)
        self.readout = (
            nn.Sequential(
                nn.Linear(state_dim, readout_dim),
                nn.GELU(),
                nn.Linear(readout_dim, num_classes),
            )
            if readout_dim is not None
            else nn.Linear(state_dim, num_classes)
        )
        nn.init.normal_(self.initial_state, std=0.05)
        nn.init.normal_(self.injection, std=0.06)
        nn.init.normal_(self.transition, std=0.03)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        state = self.initial_state[None] + self.demographic_projection(vector[:, -2:])
        for position in range(tokens.shape[1]):
            token = tokens[:, position].clamp_max(self.num_tokens - 1)
            transformed = torch.einsum("bij,bj->bi", self.transition[token], state)
            candidate = torch.tanh(state + self.step_size * (transformed + self.injection[token]))
            state = torch.where(mask[:, position, None], candidate, state)
        return state

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))


class OrthogonalRealRecurrence(nn.Module):
    """Shared norm-preserving real recurrence with a learned orthogonal transition."""

    def __init__(self, num_tokens: int, pad_token: int, num_classes: int, state_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens + 1, state_dim, padding_idx=pad_token)
        self.skew_source = nn.Parameter(torch.empty(state_dim, state_dim))
        self.demographic_projection = nn.Linear(2, state_dim)
        self.readout = nn.Linear(state_dim, num_classes)
        nn.init.normal_(self.embedding.weight, std=0.06)
        nn.init.normal_(self.skew_source, std=0.02)

    def encode(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        vector: torch.Tensor,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.tanh(self.demographic_projection(vector[:, -2:]))
        skew = self.skew_source - self.skew_source.T
        transition = torch.linalg.matrix_exp(skew)
        embedded = self.embedding(tokens)
        for position in range(tokens.shape[1]):
            candidate = torch.tanh(state @ transition.T + embedded[:, position])
            state = torch.where(mask[:, position, None], candidate, state)
        return state

    def forward(self, **batch: torch.Tensor) -> torch.Tensor:
        return self.readout(self.encode(**batch))
