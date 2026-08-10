"""Conventional parameter-budgeted baselines."""

from __future__ import annotations

import torch
from torch import nn


class EvidenceMLP(nn.Module):
    """One-hidden-layer baseline over values, missingness, and demographics."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, vector: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        return self.network(vector)


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
            dim_feedforward=2 * model_dim,
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

    def forward(
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
        return self.readout(pooled)


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

    def forward(
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
        return self.readout(final.squeeze(0))
