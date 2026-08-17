"""Nova candidates. Each isolates ONE change from a baseline, so a difference is attributable.

The candidate space is derived from the measured frontier rather than from taste. On the five
shortcut-clean tasks at 4x the trained length, the baseline sweep found:

    parity_scan  chance 0.501   best 1.000  (LSTM/GRU)
    mod_sum      chance 0.145   best 0.992  (LSTM)
    copy         chance 0.126   best 0.268  (linear attention)
    reverse      chance 0.126   best 0.336  (LSTM)
    needle       chance 0.131   best 0.764  (transformer)

Two gaps, both real:

1. **No architecture does both.** Recurrent models implement the state automaton exactly and
   extrapolate perfectly on it, but retrieve badly. Attention retrieves but its state-tracking
   collapses. Linear attention and selective state-space models -- the established "both" answer --
   are mediocre at each: 0.55 / 0.30 / 0.58. They did not close it.
2. **Ordered memory does not extrapolate at all.** Copy and reverse sit near chance for every
   architecture tested.

**Hypothesis H-DILUTION.** Softmax attention is not length-invariant. Adding non-matching keys
shifts the output because they take probability mass, so a read learned at length 16 is a different
operation at length 64. If dilution is the cause, a retrieval whose output does not depend on the
number of non-matching candidates should extrapolate where softmax does not.

Three normalisers implement that, and each is a one-line change from the baseline:

* ``max``       -- divide by the maximum score instead of the sum. Invariant to added low-score keys.
* ``threshold`` -- zero any score below a learned margin below the max, then renormalise.
* ``logl``      -- keep softmax but scale the temperature by ``log(L)``, which is the direct
  correction for mass dilution rather than a change of operation.

`softmax` is retained in the same class as the internal control, so the comparison is one flag.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn

from nova.tasks import VOCAB
from nova.zoo import _causal_mask


class InvariantAttention(nn.Module):
    """Causal attention whose normaliser is selectable. `softmax` reproduces the baseline exactly."""

    def __init__(self, d: int, heads: int, normaliser: str = "max", rope: bool = True):
        super().__init__()
        self.d, self.heads, self.head_dim = d, heads, d // heads
        self.normaliser, self.rope = normaliser, rope
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        if normaliser == "threshold":
            self.margin = nn.Parameter(torch.tensor(2.0))
        # `softmax_rms` is the confound control: ordinary softmax, but with the SAME post-read RMS
        # normalisation that the unnormalised readers need. Without it, any advantage of `max` or
        # `threshold` could be the extra normalisation rather than the length invariance.

    def _rotate(self, x):
        length, dim = x.shape[-2], x.shape[-1]
        if dim % 2:
            return x  # rotary needs an even head dimension; leave odd ones unrotated
        half = dim // 2
        frequency = 1.0 / (10000.0 ** (torch.arange(0, half, device=x.device).float() / half))
        angle = torch.arange(length, device=x.device).float().unsqueeze(1) * frequency
        cos, sin = angle.cos(), angle.sin()
        left, right = x[..., :half], x[..., half:]
        return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)

    def forward(self, h):
        batch, length, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        if self.rope:
            q, k = self._rotate(q), self._rotate(k)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = _causal_mask(length, h.device)
        scores = scores.masked_fill(mask, float("-inf"))

        if self.normaliser in ("softmax", "softmax_rms"):
            weights = torch.softmax(scores, dim=-1)
        elif self.normaliser == "logl":
            # Mass dilution grows like log of the candidate count; sharpen by exactly that much.
            positions = torch.arange(1, length + 1, device=h.device).float().view(1, 1, -1, 1)
            weights = torch.softmax(scores * positions.log().clamp_min(1.0), dim=-1)
        elif self.normaliser == "max":
            # Divide by the maximum instead of the sum, and DO NOT renormalise afterwards: the
            # weight of a high-scoring key must not depend on how many low-scoring keys exist.
            #
            # A first version divided by the sum at the end, which is algebraically exactly softmax
            # and produced identical numbers to the control. The bug was invisible in the code and
            # obvious in the results, which is why the internal control is a flag on the same class.
            shifted = scores - scores.amax(dim=-1, keepdim=True)
            weights = shifted.exp().masked_fill(mask, 0.0)
        elif self.normaliser == "threshold":
            # Hard-gate everything more than a learned margin below the best match, then read
            # without a sum normaliser. Keys that lose by more than the margin contribute nothing at
            # all, so their number is irrelevant.
            top = scores.amax(dim=-1, keepdim=True)
            keep = (scores >= (top - self.margin.abs())) & ~mask
            weights = (scores - top).exp() * keep.float()
        else:
            raise ValueError(f"unknown normaliser {self.normaliser!r}")

        attended = torch.nan_to_num(weights) @ v
        if self.normaliser in ("max", "threshold", "softmax_rms"):
            # Unnormalised reads have a magnitude that grows with the number of MATCHING keys, which
            # is information rather than dilution, but it has to be tamed before the residual.
            attended = attended * torch.rsqrt(
                attended.pow(2).mean(-1, keepdim=True) + 1e-6
            )
        return self.out(attended.transpose(1, 2).reshape(batch, length, self.d))


class InvariantTransformer(nn.Module):
    def __init__(self, d: int = 64, depth: int = 3, heads: int = 4, normaliser: str = "max"):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [nn.LayerNorm(d), InvariantAttention(d, heads, normaliser), nn.LayerNorm(d),
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


class RecurrentRetrieval(nn.Module):
    """A gated recurrent core plus a retrieval head, with the normaliser selectable.

    The measured complementarity says recurrence tracks state and attention retrieves. This puts
    both in one model so the question becomes whether a single architecture can hold both
    capabilities, and whether the retrieval half needs length invariance to survive extrapolation.

    An LSTM augmented with softmax attention is prior art (attention-augmented RNNs, and every
    encoder-decoder before transformers), so `normaliser="softmax"` is the control rather than a
    candidate.
    """

    def __init__(self, d: int = 64, depth: int = 2, heads: int = 4, normaliser: str = "max",
                 attention_dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.rnn = nn.LSTM(d, d, num_layers=depth, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = InvariantAttention(d, heads, normaliser, rope=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.head = nn.Linear(d, VOCAB)
        #: Probability of dropping the whole attention branch for a training batch. If the
        #: interference account is right, handicapping the non-extrapolating route should force the
        #: recurrence to be learned. Zero reproduces the plain hybrid exactly.
        self.attention_dropout = attention_dropout
        #: Test-time branch ablation, for the mechanistic intervention. `None` keeps both.
        self.ablate: str | None = None

    def forward(self, x):
        h = self.embed(x)
        recurrent = self.rnn(h)[0]
        if self.ablate != "recurrence":
            h = h + recurrent
        if self.ablate != "attention":
            use = True
            if self.training and self.attention_dropout > 0.0:
                use = bool(torch.rand(()) > self.attention_dropout)
            if use:
                h = h + self.attention(self.norm1(h))
        h = h + self.mlp(self.norm2(h))
        return self.head(h)


def make_candidates() -> dict[str, Callable[..., nn.Module]]:
    """The Tier 0/1 sweep. Every entry differs from a named control by exactly one flag."""

    space: dict[str, Callable[..., nn.Module]] = {}
    for normaliser in ("softmax", "softmax_rms", "logl", "max", "threshold"):
        space[f"attn_{normaliser}"] = (
            lambda normaliser=normaliser, **kw: InvariantTransformer(normaliser=normaliser, **kw)
        )
        space[f"rnn_attn_{normaliser}"] = (
            lambda normaliser=normaliser, **kw: RecurrentRetrieval(normaliser=normaliser, **kw)
        )
    # Handicapping the shortcut route: the H-INTERFERENCE intervention.
    for probability in (0.25, 0.5, 0.75):
        space[f"rnn_attn_drop{int(probability * 100)}"] = (
            lambda p=probability, **kw: RecurrentRetrieval(
                normaliser="softmax", attention_dropout=p, **kw
            )
        )
    return space


CANDIDATES = make_candidates()


class LateFusion(nn.Module):
    """Two routes with NO shared parameters, combined only by a gate at the output.

    Tests whether the measured capability competition is caused by parameter sharing. If disjoint
    parameters resolve it, the competition is a capacity-allocation problem. If it persists, the
    competition is in the objective -- the training distribution simply does not reward the
    extrapolating solution -- and no amount of separation will help.

    The gate is per-position and content-dependent, so the model can route state-tracking positions
    to the recurrence and retrieval positions to attention if that is what it wants.
    """

    def __init__(self, d: int = 48, depth: int = 2, heads: int = 4, gated: bool = True):
        super().__init__()
        self.recurrent_embed = nn.Embedding(VOCAB, d)
        self.attention_embed = nn.Embedding(VOCAB, d)
        self.rnn = nn.LSTM(d, d, num_layers=depth, batch_first=True)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                [nn.LayerNorm(d), InvariantAttention(d, heads, "softmax"), nn.LayerNorm(d),
                 nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))]
            )
            for _ in range(depth)
        )
        self.gated = gated
        self.gate = nn.Linear(2 * d, 1)
        self.head_recurrent = nn.Linear(d, VOCAB)
        self.head_attention = nn.Linear(d, VOCAB)

    def forward(self, x):
        r = self.recurrent_embed(x)
        r = r + self.rnn(r)[0]
        a = self.attention_embed(x)
        for norm1, attention, norm2, mlp in self.blocks:
            a = a + attention(norm1(a))
            a = a + mlp(norm2(a))
        logits_r, logits_a = self.head_recurrent(r), self.head_attention(a)
        if not self.gated:
            return logits_r + logits_a
        weight = torch.sigmoid(self.gate(torch.cat([r, a], dim=-1)))
        return weight * logits_r + (1 - weight) * logits_a


class LoopedTransformer(nn.Module):
    """One weight-shared block applied `loops` times. Tests whether more internal computation helps.

    Universal-Transformer-style depth recurrence is prior art; it is here to answer the directive's
    question of whether extra internal steps buy capability rather than repeated execution, measured
    by sweeping `loops` at inference.
    """

    def __init__(self, d: int = 64, depth: int = 1, heads: int = 4, loops: int = 4):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attention = InvariantAttention(d, heads, "softmax")
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.loops = loops

    def forward(self, x):
        h = self.embed(x)
        for _ in range(self.loops):
            h = h + self.attention(self.norm1(h))
            h = h + self.mlp(self.norm2(h))
        return self.head(self.norm(h))


CANDIDATES["late_fusion_gated"] = lambda **kw: LateFusion(gated=True, **kw)
CANDIDATES["late_fusion_sum"] = lambda **kw: LateFusion(gated=False, **kw)
for _loops in (2, 4, 8):
    CANDIDATES[f"looped{_loops}"] = (
        lambda loops=_loops, **kw: LoopedTransformer(loops=loops, **kw)
    )


class CursorMemory(nn.Module):
    """An explicit read pointer that moves by a PREDICTED RELATIVE increment over a memory of inputs.

    Copy and reverse are the one gap where every architecture tested sits near chance at 4x length
    (best 0.38 against a chance of 0.126). The reason is positional: emitting item i of the payload
    at output position i needs an index that keeps working when the payload gets longer, and both
    absolute positions and content-addressed reads fail at that.

    A cursor moves by a *relative* step each timestep, so the operation it implements is the same at
    any length by construction. The read is a soft window around the cursor so the whole thing stays
    differentiable.

    This is Neural Turing Machine / DNC territory (Graves et al. 2014, 2016) and location-based
    addressing is exactly their contribution. No novelty is claimed for the mechanism; the question
    is whether it closes a gap that nothing else closes.
    """

    def __init__(self, d: int = 64, depth: int = 1, heads: int = 4, sharpness: float = 4.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.rnn = nn.LSTM(d, d, num_layers=depth, batch_first=True)
        # Three moves -- back one, stay, forward one -- plus a reset-to-start gate, which is what
        # `reverse` needs and `copy` does not.
        self.move = nn.Linear(d, 3)
        self.reset = nn.Linear(d, 1)
        self.norm = nn.LayerNorm(2 * d)
        self.head = nn.Linear(2 * d, VOCAB)
        self.sharpness = sharpness

    def forward(self, x):
        batch, length = x.shape
        memory = self.embed(x)
        control = self.rnn(memory)[0]
        positions = torch.arange(length, device=x.device).float()
        cursor = torch.zeros(batch, device=x.device)
        reads = []
        for step in range(length):
            state = control[:, step]
            shift = torch.softmax(self.move(state), dim=-1)
            delta = shift[:, 2] - shift[:, 0]
            gate = torch.sigmoid(self.reset(state)).squeeze(-1)
            cursor = (1 - gate) * (cursor + delta)
            # Soft window over memory, centred on the cursor. Causal: never read ahead of `step`.
            distance = positions.view(1, -1) - cursor.view(-1, 1)
            weights = torch.softmax(-self.sharpness * distance.abs(), dim=-1)
            weights = weights * (positions.view(1, -1) <= step).float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            reads.append((weights.unsqueeze(-1) * memory).sum(1))
        return self.head(self.norm(torch.cat([control, torch.stack(reads, 1)], dim=-1)))


CANDIDATES["cursor"] = lambda **kw: CursorMemory(**kw)
CANDIDATES["cursor_sharp"] = lambda **kw: CursorMemory(sharpness=12.0, **kw)


class CursorAttention(nn.Module):
    """All three routes at once: recurrence, a location cursor, and content-addressed attention.

    The final test of whether the measured capability competition is fundamental. Each route leads a
    different column of the frontier -- recurrence leads state tracking, the cursor leads ordered
    memory, attention leads retrieval -- and no pair has yet produced a model that leads all three.
    If the capabilities compose, this should approach the per-task best everywhere. If they compete,
    it will lose somewhere, and the competition is a property of the objective rather than of any
    particular pairing.

    Every component is prior art (LSTM; NTM location addressing; attention-augmented RNNs). Nothing
    here is claimed as a mechanism.
    """

    def __init__(self, d: int = 48, depth: int = 2, heads: int = 4, sharpness: float = 4.0):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.rnn = nn.LSTM(d, d, num_layers=depth, batch_first=True)
        self.move = nn.Linear(d, 3)
        self.reset = nn.Linear(d, 1)
        self.sharpness = sharpness
        self.norm_a = nn.LayerNorm(d)
        self.attention = InvariantAttention(d, heads, "max", rope=False)
        self.norm_out = nn.LayerNorm(3 * d)
        self.head = nn.Linear(3 * d, VOCAB)

    def forward(self, x):
        batch, length = x.shape
        memory = self.embed(x)
        control = self.rnn(memory)[0]
        positions = torch.arange(length, device=x.device).float()
        cursor = torch.zeros(batch, device=x.device)
        reads = []
        for step in range(length):
            state = control[:, step]
            shift = torch.softmax(self.move(state), dim=-1)
            gate = torch.sigmoid(self.reset(state)).squeeze(-1)
            cursor = (1 - gate) * (cursor + shift[:, 2] - shift[:, 0])
            distance = positions.view(1, -1) - cursor.view(-1, 1)
            weights = torch.softmax(-self.sharpness * distance.abs(), dim=-1)
            weights = weights * (positions.view(1, -1) <= step).float()
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
            reads.append((weights.unsqueeze(-1) * memory).sum(1))
        retrieved = self.attention(self.norm_a(control))
        joined = torch.cat([control, torch.stack(reads, 1), retrieved], dim=-1)
        return self.head(self.norm_out(joined))


CANDIDATES["cursor_attn"] = lambda **kw: CursorAttention(**kw)
