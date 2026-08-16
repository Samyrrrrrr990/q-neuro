"""Gate 1/3/9: real-data early activity classification, five halting policies, one shared core.

Lane B. Implements exactly the protocol frozen in `QNEURO3-HAR-P1` (sha256 70e92005...). Nothing
here may be changed after that prediction is opened.

UCI Human Activity Recognition Using Smartphones. The 128-step inertial window arrives in 16 chunks
of 8 timesteps; the model classifies the activity as early as it can. The split is the dataset's own
canonical subject-disjoint partition, so it is a genuine distribution shift by person and I did not
choose it.

Five arms over one identical recurrent core:

* ``fixed``      -- consume all 16 chunks, classify from the final state.
* ``confidence`` -- exit when max softmax exceeds a threshold tuned on validation only. The
  strongest natural baseline: it needs no teacher and no extra training.
* ``supervised`` -- the Q-Neuro 3.0 mechanism. Halt on a supervised predicate, classify at the halt
  step. The halt target comes from a teacher's earliest confident-and-correct chunk, which is
  **early-exit distillation and is prior art**; it also doubles this arm's training cost.
* ``act``        -- Graves (2016) adaptive computation time: halting unit, ponder cost, output is
  the halting-weighted mean over steps.
* ``pondernet``  -- Banino et al. (2021): per-step Bernoulli halting, expected task loss under the
  halting distribution, KL to a geometric prior.
"""

from __future__ import annotations

import pathlib
from typing import Any

import torch
from torch import nn

CHUNK = 8
CHUNKS = 16
CHANNELS = 9
CLASSES = 6
WIDTH = 96
VALIDATION_SUBJECTS = (1, 3, 5, 6)


def load(directory: str | pathlib.Path) -> dict[str, dict[str, torch.Tensor]]:
    """Train / validation / test tensors, standardised with TRAIN statistics only."""

    import numpy as np

    directory = pathlib.Path(directory)
    raw = {
        split: np.load(directory / f"har_{split}.npz") for split in ("train", "test")
    }
    train_x = torch.from_numpy(raw["train"]["x"])
    mean = train_x.mean(dim=(0, 2), keepdim=True)
    std = train_x.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)

    def pack(x, y, subject):
        x = (torch.from_numpy(x) - mean) / std
        # (N, channels, 128) -> (N, 16 chunks, 9*8 features)
        chunks = x.unfold(2, CHUNK, CHUNK).permute(0, 2, 1, 3).reshape(x.shape[0], CHUNKS, -1)
        return {
            "x": chunks.contiguous(),
            "y": torch.from_numpy(y) - 1,
            "subject": torch.from_numpy(subject),
        }

    full = pack(raw["train"]["x"], raw["train"]["y"], raw["train"]["subject"])
    held = torch.isin(full["subject"], torch.tensor(VALIDATION_SUBJECTS))
    return {
        "train": {k: v[~held] for k, v in full.items()},
        "validation": {k: v[held] for k, v in full.items()},
        "test": pack(raw["test"]["x"], raw["test"]["y"], raw["test"]["subject"]),
    }


class Core(nn.Module):
    """One chunk per step. Identical for every arm, so differences are the halting policy."""

    def __init__(self, d: int = WIDTH):
        super().__init__()
        self.embed = nn.Linear(CHANNELS * CHUNK, d)
        self.step = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.classify = nn.Linear(d, CLASSES)

    def start(self, batch: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch.shape[0], self.embed.out_features, device=batch.device)

    def advance(self, h: torch.Tensor, chunk: torch.Tensor) -> torch.Tensor:
        h = h + self.step(torch.cat([h, self.embed(chunk)], dim=-1))
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)


class Halting(nn.Module):
    """All five policies over one `Core`, selected by `arm`."""

    ARMS = ("fixed", "confidence", "supervised", "act", "pondernet")

    def __init__(self, arm: str, d: int = WIDTH, chunks: int = CHUNKS):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm, self.chunks = arm, chunks
        self.core = Core(d)
        self.halt = nn.Linear(d, 1)
        if arm in ("supervised", "act", "pondernet"):
            # Bias towards not halting at the start, so the core becomes useful before halting has
            # anything to trade against. Cycle 1 measured that ponder collapse is otherwise the
            # dominant failure mode of learned halting.
            nn.init.constant_(self.halt.bias, -2.0)

    def sweep(self, x: torch.Tensor):
        """Per-chunk logits and halting probabilities. Every arm reads these; only the rule differs."""

        h = self.core.start(x)
        logits, probs = [], []
        for index in range(self.chunks):
            h = self.core.advance(h, x[:, index])
            logits.append(self.core.classify(h))
            probs.append(torch.sigmoid(self.halt(h)).squeeze(-1))
        return torch.stack(logits, 1), torch.stack(probs, 1).clamp(1e-6, 1 - 1e-6)

    @staticmethod
    def first_arrival(p: torch.Tensor):
        log_not = torch.log1p(-p)
        cum = torch.cat([torch.zeros_like(log_not[:, :1]), log_not[:, :-1].cumsum(1)], dim=1)
        return torch.log(p) + cum

    def decide(self, logits: torch.Tensor, p: torch.Tensor, *, threshold: float = 0.5):
        """Which chunk each example stops at, and the answer read there."""

        if self.arm == "fixed":
            step = torch.full((logits.shape[0],), self.chunks, dtype=torch.long)
            return logits[:, -1], step
        if self.arm == "confidence":
            confident = logits.softmax(-1).max(-1).values > threshold
        else:
            confident = p > 0.5
        any_fired = confident.any(dim=1)
        first = torch.where(
            any_fired, confident.float().argmax(dim=1),
            torch.full_like(any_fired, self.chunks - 1, dtype=torch.long),
        )
        picked = logits.gather(1, first.view(-1, 1, 1).expand(-1, 1, CLASSES)).squeeze(1)
        return picked, first + 1


def teacher_targets(model: Halting, x: torch.Tensor, y: torch.Tensor, *, confidence: float = 0.9):
    """Earliest chunk where the teacher is correct AND confident; else the last chunk.

    This is early-exit distillation, declared as prior art in the frozen record.
    """

    with torch.no_grad():
        logits, _ = model.sweep(x)
        probability = logits.softmax(-1)
        correct = probability.argmax(-1) == y.unsqueeze(1)
        good = correct & (probability.max(-1).values > confidence)
        any_good = good.any(dim=1)
        first = torch.where(
            any_good, good.float().argmax(dim=1),
            torch.full_like(any_good, CHUNKS - 1, dtype=torch.long),
        )
    return first + 1


def train_arm(
    arm: str, data: dict, *, seed: int = 0, epochs: int = 12, batch_size: int = 128,
    learning_rate: float = 2e-3, ponder: float = 0.01, teacher: Halting | None = None,
) -> Halting:
    torch.manual_seed(seed)
    model = Halting(arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cross_entropy = torch.nn.functional.cross_entropy
    x_all, y_all = data["train"]["x"], data["train"]["y"]
    n = x_all.shape[0]
    generator = torch.Generator().manual_seed(seed + 777)
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            x, y = x_all[index], y_all[index]
            optimizer.zero_grad(set_to_none=True)
            logits, p = model.sweep(x)

            if arm in ("fixed", "confidence"):
                # Deep supervision at every chunk: the confidence baseline needs calibrated
                # per-chunk heads to exit on, and giving it less would be a strawman.
                loss = cross_entropy(
                    logits.reshape(-1, CLASSES), y.repeat_interleave(CHUNKS)
                )
            elif arm == "supervised":
                target = teacher_targets(teacher, x, y)
                log_first = model.first_arrival(p)
                at_truth = logits.gather(
                    1, (target - 1).view(-1, 1, 1).expand(-1, 1, CLASSES)
                ).squeeze(1)
                loss = -log_first.gather(1, (target - 1).unsqueeze(1)).mean()
                loss = loss + cross_entropy(at_truth, y)
            elif arm == "act":
                remaining = torch.ones(x.shape[0])
                mixed = torch.zeros(x.shape[0], CLASSES)
                expected = torch.zeros(x.shape[0])
                for step in range(CHUNKS):
                    q = p[:, step] if step < CHUNKS - 1 else torch.ones_like(p[:, step])
                    weight = remaining * q
                    mixed = mixed + weight.unsqueeze(-1) * logits[:, step].softmax(-1)
                    expected = expected + weight * (step + 1)
                    remaining = remaining * (1 - q)
                loss = torch.nn.functional.nll_loss(
                    torch.log(mixed + 1e-9), y
                ) + ponder * expected.mean()
            else:  # pondernet
                log_first = model.first_arrival(p)
                weights = log_first.exp()
                per_step = torch.stack(
                    [cross_entropy(logits[:, s], y, reduction="none") for s in range(CHUNKS)], 1
                )
                expected_loss = (weights * per_step).sum(1).mean()
                geometric = torch.tensor(
                    [(1 - 0.2) ** k * 0.2 for k in range(CHUNKS)]
                )
                geometric = geometric / geometric.sum()
                normalised = weights / weights.sum(1, keepdim=True).clamp_min(1e-9)
                kl = (
                    normalised * (normalised.clamp_min(1e-9).log() - geometric.log())
                ).sum(1).mean()
                loss = expected_loss + ponder * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


@torch.no_grad()
def evaluate(model: Halting, split: dict, *, threshold: float = 0.5) -> dict[str, float]:
    logits, p = model.sweep(split["x"])
    picked, step = model.decide(logits, p, threshold=threshold)
    return {
        "accuracy": float((picked.argmax(-1) == split["y"]).float().mean()),
        "mean_chunks": float(step.float().mean()),
        "p95_chunks": float(step.float().quantile(0.95)),
    }


def confidence_curve(model: Halting, split: dict) -> list[dict[str, float]]:
    """The baseline's own accuracy/compute frontier, swept over its exit threshold."""

    return [
        {"threshold": t, **evaluate(model, split, threshold=t)}
        for t in (0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.999, 1.01)
    ]


def parameter_count(arm: str) -> int:
    return sum(p.numel() for p in Halting(arm).parameters())


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("accuracy", "mean_chunks")
    return {
        k: sum(r[k] for r in results) / len(results) for k in keys
    } | {"seeds": [r.get("seed") for r in results]}
