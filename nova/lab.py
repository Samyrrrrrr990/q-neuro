"""Nova Discovery Lab: tiered screening, budget matching, and a registry that remembers failures.

The dominant failure mode of the previous programme was spending weeks developing beautiful
theories about bad architectures. The lab exists to prevent that: candidates are killed in minutes
unless they show a large signal, and nothing gets a careful analysis until it has earned one.

Tiers, in increasing cost:

* **Tier 0 — viability.** Does it train, stay finite, and beat the majority-class baseline on one
  task? Seconds. Most candidates die here.
* **Tier 1 — signal.** All eight tasks, two seeds, evaluated in-distribution *and* at 2x and 4x the
  trained length. The question is whether a large capability gap exists, not whether a score moved.
* **Tier 2+** are run by hand once something survives, and Sentinel is not spent on anything below
  Tier 2.

Everything is written to a JSONL registry so a mechanism cannot be silently rediscovered after
being killed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nova import tasks

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "research" / "nova" / "registry.jsonl"

#: Matched across every candidate and baseline. Changing these invalidates cross-run comparisons,
#: so they are constants rather than arguments.
STEPS, BATCH, LEARNING_RATE = 800, 64, 3e-3
PARAMETER_TARGET = 120_000


def accuracy(model: nn.Module, task: str, length: int, *, seed: int, batch: int = 256) -> float:
    """Fraction of scored positions predicted exactly. Whole-sequence credit is not given, because
    partial credit is what distinguishes "learned the procedure" from "learned the prefix"."""

    sample = tasks.make(task, batch, length, seed=seed)
    with torch.no_grad():
        logits = model(sample["x"])
    predicted = logits.argmax(-1)
    scored = sample["mask"]
    if not bool(scored.any()):
        return float("nan")
    return float((predicted[scored] == sample["y"][scored]).float().mean())


def train(
    build: Callable[[], nn.Module], task: str, *, seed: int, steps: int = STEPS,
    batch: int = BATCH, learning_rate: float = LEARNING_RATE,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.1
    )
    started = time.perf_counter()
    losses = []
    for step in range(steps):
        sample = tasks.train_batch(task, batch, seed=seed * 100_000 + step)
        optimizer.zero_grad(set_to_none=True)
        logits = model(sample["x"])
        scored = sample["mask"]
        if not bool(scored.any()):
            continue
        loss = torch.nn.functional.cross_entropy(logits[scored], sample["y"][scored])
        if not torch.isfinite(loss):
            return model, {"diverged": True, "step": step, "seconds": time.perf_counter() - started}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        losses.append(float(loss.detach()))
    return model, {
        "diverged": False,
        "final_loss": sum(losses[-50:]) / max(1, len(losses[-50:])),
        "seconds": time.perf_counter() - started,
    }


def tier0(build: Callable[[], nn.Module], *, task: str = "parity_scan", seed: int = 0) -> dict:
    """Minutes. Trains, stays finite, beats the majority class. Kill aggressively."""

    model, info = train(build, task, seed=seed, steps=250)
    if info["diverged"]:
        return {"tier": 0, "pass": False, "reason": "diverged", **info}
    score = accuracy(model, task, tasks.TRAIN_MAX, seed=999)
    chance = tasks.chance_level(task)
    return {
        "tier": 0,
        "pass": bool(score > chance + 0.10),
        "reason": "beat chance" if score > chance + 0.10 else "did not beat chance",
        "accuracy": score,
        "chance": chance,
        "seconds": info["seconds"],
    }


def tier1(
    name: str, build: Callable[[], nn.Module], *, seeds: tuple[int, ...] = (0, 1),
    task_list: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """All tasks, all evaluation lengths. The output is a capability matrix, not a scalar."""

    task_list = task_list or tuple(tasks.TASKS)
    results: dict[str, Any] = {"name": name, "tasks": {}}
    total_seconds = 0.0
    for task in task_list:
        per_seed = []
        for seed in seeds:
            model, info = train(build, task, seed=seed)
            total_seconds += info["seconds"]
            if info["diverged"]:
                per_seed.append({"seed": seed, "diverged": True})
                continue
            per_seed.append(
                {
                    "seed": seed,
                    "diverged": False,
                    "final_loss": info["final_loss"],
                    **{
                        f"acc@{length}": accuracy(model, task, length, seed=1000 + length)
                        for length in tasks.EVAL_LENGTHS
                    },
                }
            )
        good = [r for r in per_seed if not r["diverged"]]
        results["tasks"][task] = {
            "chance": tasks.chance_level(task),
            "runs": per_seed,
            **{
                f"mean@{length}": (
                    sum(r[f"acc@{length}"] for r in good) / len(good) if good else float("nan")
                )
                for length in tasks.EVAL_LENGTHS
            },
        }
    results["seconds"] = total_seconds
    results["parameters"] = sum(p.numel() for p in build().parameters())
    return results


def summarise(result: dict[str, Any]) -> dict[str, float]:
    """Two scalars that matter: can it learn the procedure, and does the procedure survive length.

    `in_distribution` averages accuracy at the trained length. `extrapolation_ratio` is accuracy at
    4x the trained length divided by accuracy in distribution -- a model that memorised the training
    lengths scores near zero on it however good its in-distribution number is.
    """

    per_task = result["tasks"]
    inside = [v[f"mean@{tasks.TRAIN_MAX}"] for v in per_task.values()]
    outside = [v["mean@64"] for v in per_task.values()]
    above_chance = [
        1.0 if v[f"mean@{tasks.TRAIN_MAX}"] > v["chance"] + 0.10 else 0.0 for v in per_task.values()
    ]
    inside_mean = sum(inside) / len(inside)
    return {
        "in_distribution": inside_mean,
        "extrapolated": sum(outside) / len(outside),
        "extrapolation_ratio": (sum(outside) / len(outside)) / max(inside_mean, 1e-6),
        "tasks_learned": sum(above_chance),
    }


def record(entry: dict[str, Any]) -> None:
    """Append to the registry. Nothing is ever overwritten, so a killed idea stays killed."""

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


def load_registry() -> list[dict[str, Any]]:
    if not REGISTRY.is_file():
        return []
    return [json.loads(line) for line in REGISTRY.read_text(encoding="utf-8").splitlines() if line]


def already_tried(name: str) -> dict[str, Any] | None:
    """Query before proposing a mechanism. Rediscovering a dead idea is the cheapest waste there is."""

    for entry in reversed(load_registry()):
        if entry.get("name") == name:
            return entry
    return None
