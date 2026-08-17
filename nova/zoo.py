"""Baseline architectures. These are the things Nova has to beat, so they are built to be strong.

Every model shares one interface — `(B, L)` int tokens in, `(B, L, VOCAB)` logits out — and is
causal, so no model can see the future.

**On strawmen.** Length extrapolation is trivially impossible for a transformer with learned
absolute position embeddings, so using only that would be dishonest. RoPE and ALiBi are included
because both are designed to extrapolate, and ALiBi in particular was introduced for exactly this
property. `transformer_learned` is kept as a *reference point* to show the axis is real, never as
the comparison a claim rests on.

**On the SSM.** A diagonal state-space layer is the S4D/Mamba-family core without input-dependent
selectivity. It is implemented as an explicit recurrence rather than an FFT scan because at these
lengths the recurrence is exact, cheap, and easier to verify.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from nova.tasks import VOCAB


def _causal_mask(length: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class RotaryAttention(nn.Module):
    """Multi-head causal attention with rotary positions, or ALiBi slopes, or learned absolutes."""

    def __init__(self, d: int, heads: int, position: str):
        super().__init__()
        self.d, self.heads, self.position = d, heads, position
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        if position == "alibi":
            slopes = 2.0 ** (-8.0 * torch.arange(1, heads + 1) / heads)
            self.register_buffer("slopes", slopes.view(1, heads, 1, 1), persistent=False)

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        length, dim = x.shape[-2], x.shape[-1]
        half = dim // 2
        frequency = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device).float() / half))
        angle = torch.arange(length, device=x.device).float().unsqueeze(1) * frequency
        cos, sin = angle.cos(), angle.sin()
        left, right = x[..., :half], x[..., half:]
        return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if self.position == "rope":
            q, k = self._rotate(q), self._rotate(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.position == "alibi":
            distance = torch.arange(length, device=h.device)
            scores = scores - self.slopes * (distance.view(1, 1, 1, -1) - distance.view(1, 1, -1, 1)).abs()
        scores = scores.masked_fill(_causal_mask(length, h.device), float("-inf"))
        attended = torch.softmax(scores, dim=-1) @ v
        return self.out(attended.transpose(1, 2).reshape(batch, length, self.d))


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, position: str):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = RotaryAttention(d, heads, position)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        h = h + self.attention(self.norm1(h))
        return h + self.mlp(self.norm2(h))


class Transformer(nn.Module):
    def __init__(self, d: int = 64, depth: int = 3, heads: int = 4, position: str = "rope"):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.position = position
        if position == "learned":
            self.absolute = nn.Embedding(512, d)
        self.blocks = nn.ModuleList(TransformerBlock(d, heads, position) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.embed(x)
        if self.position == "learned":
            h = h + self.absolute(torch.arange(x.shape[1], device=x.device)).unsqueeze(0)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


class GatedRNN(nn.Module):
    """GRU or LSTM. Recurrent state is length-agnostic by construction, so these extrapolate if
    they learn the right update at all."""

    def __init__(self, d: int = 64, depth: int = 2, cell: str = "gru"):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        module = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = module(d, d, num_layers=depth, batch_first=True)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        return self.head(self.rnn(self.embed(x))[0])


class DiagonalSSM(nn.Module):
    """S4D-style diagonal state space: h_t = a * h_{t-1} + b * u_t, with a stable by construction.

    No input-dependent selectivity, which is the Mamba addition; that is a deliberate choice so the
    baseline family is the well-understood one.
    """

    def __init__(self, d: int = 64, depth: int = 3, state: int = 16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.depth, self.d, self.state = depth, d, state
        # Logits of the decay, spread so sigmoid covers fast and slow timescales at init. A first
        # version took log() of a negative linspace and produced NaNs, which the causality probe
        # caught immediately -- the model was not non-causal, it was not finite.
        self.decay_logit = nn.ParameterList(
            nn.Parameter(torch.linspace(-3.0, 3.0, state).repeat(d, 1)) for _ in range(depth)
        )
        self.inp = nn.ModuleList(nn.Linear(d, d * state) for _ in range(depth))
        self.outp = nn.ModuleList(nn.Linear(d * state, d) for _ in range(depth))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(depth)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(2 * depth))
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        batch, length = x.shape
        h = self.embed(x)
        for layer in range(self.depth):
            decay = torch.sigmoid(self.decay_logit[layer]).view(1, self.d, self.state)
            u = self.inp[layer](self.norms[2 * layer](h)).view(batch, length, self.d, self.state)
            state = torch.zeros(batch, self.d, self.state, device=x.device)
            outputs = []
            for step in range(length):
                state = decay * state + u[:, step]
                outputs.append(state)
            stacked = torch.stack(outputs, 1).reshape(batch, length, self.d * self.state)
            h = h + self.outp[layer](stacked)
            h = h + self.mlp[layer](self.norms[2 * layer + 1](h))
        return self.head(h)


class CausalMLP(nn.Module):
    """A per-position MLP over the token and a causal mean of the prefix. A weak control: it has no
    mechanism for order-sensitive computation, so anything that fails to beat it is not computing."""

    def __init__(self, d: int = 64, depth: int = 3):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.LayerNorm(d)]
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(d, VOCAB)
        self.depth = depth

    def forward(self, x):
        h = self.embed(x)
        for index in range(self.depth):
            prefix = h.cumsum(1) / torch.arange(
                1, x.shape[1] + 1, device=x.device
            ).view(1, -1, 1)
            block = self.layers[4 * index : 4 * index + 4]
            z = block[0](torch.cat([h, prefix], dim=-1))
            z = block[2](block[1](z))
            h = block[3](h + z)
        return self.head(h)


def parameter_count(name: str, **kwargs) -> int:
    return sum(p.numel() for p in BASELINES[name](**kwargs).parameters())


def match_parameters(name: str, target: int, *, depth: int = 3, tolerance: float = 0.15, **kwargs):
    """Pick the width whose parameter count lands closest to `target`.

    Matching parameters is the weakest of the fairness conditions and the easiest to get wrong by
    accident: the families here differ by 8x at equal width, so comparing them at a fixed `d` would
    be comparing model sizes, not architectures.
    """

    best, best_gap = None, float("inf")
    # Step by 8 so that d / heads is even for the 4-head models: rotary embeddings split the head
    # dimension in half and an odd head dimension makes the two halves different sizes.
    for width in range(8, 257, 8):
        try:
            count = parameter_count(name, d=width, depth=depth, **kwargs)
        except (ValueError, RuntimeError):
            continue
        gap = abs(count - target) / target
        if gap < best_gap:
            best, best_gap = (width, count), gap
    if best is None or best_gap > tolerance:
        raise ValueError(f"cannot match {name} to {target} parameters within {tolerance:.0%}")
    return {"d": best[0], "depth": depth}, best[1]


BASELINES: dict[str, Callable[..., nn.Module]] = {
    "transformer_rope": lambda **kw: Transformer(position="rope", **kw),
    "transformer_alibi": lambda **kw: Transformer(position="alibi", **kw),
    "transformer_learned": lambda **kw: Transformer(position="learned", **kw),
    "gru": lambda **kw: GatedRNN(cell="gru", **kw),
    "lstm": lambda **kw: GatedRNN(cell="lstm", **kw),
    "ssm_diag": lambda **kw: DiagonalSSM(**kw),
    "causal_mlp": lambda **kw: CausalMLP(**kw),
}


class LinearAttention(nn.Module):
    """Causal linear attention in its recurrent form: a fast-weight matrix updated by outer products.

    This family is the *obvious* answer to the gap the baseline sweep exposed -- recurrent state that
    is also content-addressable, so it should track state like an RNN and retrieve like attention.
    It is decades old (Schmidhuber's fast-weight programmers, 1992) and modern under several names
    (linear transformers, RetNet, RWKV). It is included as a BASELINE precisely so that Nova cannot
    accidentally rediscover it.

        S_t = decay * S_{t-1} + phi(k_t) v_t^T
        y_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)
    """

    def __init__(self, d: int, heads: int, decay: bool):
        super().__init__()
        self.d, self.heads, self.head_dim = d, heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.use_decay = decay
        if decay:
            self.decay_logit = nn.Parameter(torch.linspace(1.0, 4.0, heads))

    def forward(self, h):
        batch, length, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        q, k = torch.nn.functional.elu(q) + 1.0, torch.nn.functional.elu(k) + 1.0
        gamma = (
            torch.sigmoid(self.decay_logit).view(1, self.heads, 1, 1)
            if self.use_decay else torch.ones(1, self.heads, 1, 1, device=h.device)
        )
        state = torch.zeros(batch, self.heads, self.head_dim, self.head_dim, device=h.device)
        norm = torch.zeros(batch, self.heads, self.head_dim, device=h.device)
        outputs = []
        for step in range(length):
            state = gamma * state + k[:, :, step].unsqueeze(-1) * v[:, :, step].unsqueeze(-2)
            norm = gamma.squeeze(-1) * norm + k[:, :, step]
            numerator = (q[:, :, step].unsqueeze(-2) @ state).squeeze(-2)
            denominator = (q[:, :, step] * norm).sum(-1, keepdim=True).clamp_min(1e-4)
            outputs.append(numerator / denominator)
        attended = torch.stack(outputs, 2)
        return self.out(attended.transpose(1, 2).reshape(batch, length, self.d))


class LinearAttentionModel(nn.Module):
    def __init__(self, d: int = 64, depth: int = 3, heads: int = 4, decay: bool = True):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [nn.LayerNorm(d), LinearAttention(d, heads, decay), nn.LayerNorm(d),
                 nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))]
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        h = self.embed(x)
        for norm1, attention, norm2, mlp in self.blocks:
            h = h + attention(norm1(h))
            h = h + mlp(norm2(h))
        return self.head(self.norm(h))


class SelectiveSSM(nn.Module):
    """Mamba-family core: the decay is a function of the input, so the state can choose to forget.

    Selectivity is the addition that separates Mamba from S4D, and it is what lets a state-space
    model do content-dependent gating. Included as a baseline for the same reason as linear
    attention: it is the established answer, not a Nova candidate.
    """

    def __init__(self, d: int = 64, depth: int = 3, state: int = 16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.depth, self.d, self.state = depth, d, state
        self.gate = nn.ModuleList(nn.Linear(d, state) for _ in range(depth))
        self.inp = nn.ModuleList(nn.Linear(d, d * state) for _ in range(depth))
        self.outp = nn.ModuleList(nn.Linear(d * state, d) for _ in range(depth))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(depth)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(2 * depth))
        self.head = nn.Linear(d, VOCAB)

    def forward(self, x):
        batch, length = x.shape
        h = self.embed(x)
        for layer in range(self.depth):
            normed = self.norms[2 * layer](h)
            decay = torch.sigmoid(self.gate[layer](normed))
            u = self.inp[layer](normed).view(batch, length, self.d, self.state)
            state = torch.zeros(batch, self.d, self.state, device=x.device)
            outputs = []
            for step in range(length):
                state = decay[:, step].unsqueeze(1) * state + u[:, step]
                outputs.append(state)
            stacked = torch.stack(outputs, 1).reshape(batch, length, self.d * self.state)
            h = h + self.outp[layer](stacked)
            h = h + self.mlp[layer](self.norms[2 * layer + 1](h))
        return self.head(h)


BASELINES["linear_attention"] = lambda **kw: LinearAttentionModel(decay=False, **kw)
BASELINES["retentive"] = lambda **kw: LinearAttentionModel(decay=True, **kw)
BASELINES["ssm_selective"] = lambda **kw: SelectiveSSM(**kw)
