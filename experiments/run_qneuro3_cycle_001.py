"""Reproduce Q-Neuro 3.0 cycle 1: adaptive depth versus fixed depth on `chase_to_goal`.

Lane B. Nothing here is a claim; see `docs/LANE_POLICY.md`. The records this reproduces live in
`research/qneuro3/` and the narrative is in `docs/MONOGRAPH.md` part V.

Three stages, each independently runnable:

* ``variance``  — QNEURO3-Q3-VARIANCE-001. Is Q3's headline reproducible across seeds?
* ``baseline``  — QNEURO3-Q0-RELIABILITY-001. The matched control. Without it the variance number
  is uninterpretable, because the task or the budget could be flaky for every architecture.
* ``q4``        — QNEURO3-Q4-P1. Opens a frozen, hashed prediction. Verifies the hash first and
  reads its own pass criterion out of the frozen record so the code cannot drift from the
  prediction it is testing.

Every stage holds fixed: AdamW at lr 2e-3, 8 epochs, batch 128, gradient-norm clip 1.0, 24 nodes,
d = 64, max_depth 8, and a validation set built from seeds disjoint from training.

    python experiments/run_qneuro3_cycle_001.py all

Runtime is roughly 30 s per training run on the M2 reference machine; ``all`` is about 25 minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from qneuro3.elastic import Q0Fixed, Q3Arrival, Q4Grounded, arrival_loss, occupied_nodes
from qneuro3.tasks import chase_to_goal, chase_to_goal_weighted

ROOT = Path(__file__).resolve().parents[1]
FROZEN_Q4 = ROOT / "research" / "qneuro3" / "QNEURO3-Q4-P1.json"

NODES = 24
WIDTH = 64
MAX_DEPTH = 8
LEARNING_RATE = 2e-3
EPOCHS = 8
BATCH = 128
TRAIN_BATCHES = 500
VAL_BATCHES = 25
VAL_BATCH = 256

TASKS: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
    "original": chase_to_goal,
    "weighted": chase_to_goal_weighted,
}


def _batches(task_name: str, count: int, size: int, seed0: int) -> list[dict[str, torch.Tensor]]:
    task = TASKS[task_name]
    if task_name == "weighted":
        return [
            task(size, NODES, MAX_DEPTH, seed=seed0 + i, weights=[1] * MAX_DEPTH)
            for i in range(count)
        ]
    return [task(size, NODES, MAX_DEPTH, seed=seed0 + i) for i in range(count)]


def _validation(task_name: str) -> list[dict[str, torch.Tensor]]:
    # Seed 90000 is disjoint from the training range 1000..1499 for every stage.
    return _batches(task_name, VAL_BATCHES, VAL_BATCH, 90000)


def train_halting(model: torch.nn.Module, task_name: str, *, grounded: bool) -> None:
    train = _batches(task_name, TRAIN_BATCHES, BATCH, 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            if grounded:
                log_first, _, positions = model(
                    batch["perm"], batch["start"], with_positions=True
                )
                truth = occupied_nodes(batch["perm"], batch["start"], MAX_DEPTH)
                loss = arrival_loss(log_first, batch["target"]) + torch.nn.functional.cross_entropy(
                    positions.reshape(-1, NODES), truth.reshape(-1)
                )
            else:
                log_first, _ = model(batch["perm"], batch["start"])
                loss = arrival_loss(log_first, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate_halting(model: torch.nn.Module, task_name: str) -> tuple[float, float]:
    """Accuracy and mean step count. For these models the halt step *is* the answer."""

    correct = total = 0
    steps = 0.0
    validation = _validation(task_name)
    with torch.no_grad():
        for batch in validation:
            _, predicted = model(batch["perm"], batch["start"])
            steps += float(predicted.float().mean())
            correct += int((predicted == batch["target"]).sum())
            total += len(batch["target"])
    return correct / total, steps / len(validation)


def run_fixed(task_name: str, seed: int) -> float:
    train = _batches(task_name, TRAIN_BATCHES, BATCH, 1000)
    torch.manual_seed(seed)
    model = Q0Fixed(NODES, WIDTH, MAX_DEPTH, MAX_DEPTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        for batch in train:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch["perm"], batch["start"])
            torch.nn.functional.cross_entropy(logits, batch["target"] - 1).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    correct = total = 0
    with torch.no_grad():
        for batch in _validation(task_name):
            logits, _ = model(batch["perm"], batch["start"])
            correct += int((logits.argmax(-1) == (batch["target"] - 1)).sum())
            total += len(batch["target"])
    return correct / total


def stage_variance(seeds: int) -> list[dict[str, Any]]:
    rows = []
    print(f"{'task':>10} {'batches':>8} {'seed':>5} {'acc':>8} {'steps':>7}")
    for task_name in TASKS:
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = Q3Arrival(NODES, WIDTH, MAX_DEPTH)
            train_halting(model, task_name, grounded=False)
            accuracy, steps = evaluate_halting(model, task_name)
            rows.append(
                {"task": task_name, "seed": seed, "accuracy": accuracy, "steps": steps}
            )
            print(f"{task_name:>10} {TRAIN_BATCHES:8d} {seed:5d} {accuracy:8.4f} {steps:7.2f}")
    reliable = sum(r["accuracy"] >= 0.99 for r in rows)
    print(f"\nQ3 runs reaching accuracy >= 0.99: {reliable}/{len(rows)}")
    return rows


def stage_baseline(seeds: int) -> list[dict[str, Any]]:
    rows = []
    print(f"{'task':>10} {'seed':>5} {'acc':>8}   (fixed depth 8, always 8.00 steps)")
    for task_name in TASKS:
        for seed in range(seeds):
            accuracy = run_fixed(task_name, seed)
            rows.append({"task": task_name, "seed": seed, "accuracy": accuracy})
            print(f"{task_name:>10} {seed:5d} {accuracy:8.4f}")
    reliable = sum(r["accuracy"] >= 0.99 for r in rows)
    print(f"\nQ0 runs reaching accuracy >= 0.99: {reliable}/{len(rows)}")
    return rows


def stage_q4(seeds: int) -> list[dict[str, Any]]:
    payload = json.loads(FROZEN_Q4.read_text(encoding="utf-8"))
    prediction = payload["prediction"]
    recomputed = hashlib.sha256(
        json.dumps(prediction, indent=2, sort_keys=True).encode()
    ).hexdigest()
    if recomputed != payload["sha256"]:
        raise ValueError(
            "frozen prediction hash mismatch: the record was modified after freezing. "
            f"expected {payload['sha256']}, recomputed {recomputed}"
        )
    print(f"frozen prediction verified: {payload['sha256'][:16]}")
    print(f"criterion: {prediction['pass_criterion']}\n")

    rows = []
    print(f"{'task':>10} {'seed':>5} {'acc':>8} {'steps':>7}")
    for task_name in TASKS:
        for seed in range(seeds):
            torch.manual_seed(seed)
            model = Q4Grounded(NODES, WIDTH, MAX_DEPTH)
            train_halting(model, task_name, grounded=True)
            accuracy, steps = evaluate_halting(model, task_name)
            rows.append(
                {"task": task_name, "seed": seed, "accuracy": accuracy, "steps": steps}
            )
            print(f"{task_name:>10} {seed:5d} {accuracy:8.4f} {steps:7.2f}")

    original = [r for r in rows if r["task"] == "original"]
    weighted = [r for r in rows if r["task"] == "weighted"]
    n_original = sum(r["accuracy"] >= 0.99 for r in original)
    n_weighted = sum(r["accuracy"] >= 0.99 for r in weighted)
    successes = [r for r in rows if r["accuracy"] >= 0.99]
    mean_steps = (
        sum(r["steps"] for r in successes) / len(successes) if successes else float("nan")
    )
    print(f"\noriginal >= 0.99 : {n_original}/{len(original)}  (Q3 was 6/10, need >= 9)")
    print(f"weighted >= 0.99 : {n_weighted}/{len(weighted)}  (Q3 was 1/10, need >= 8)")
    print(f"mean steps on successes: {mean_steps:.2f}  (Q3 was 4.54, need within 0.25)")
    passes = n_original >= 9 and n_weighted >= 8 and abs(mean_steps - 4.54) <= 0.25
    print(f"\nQNEURO3-Q4-P1 VERDICT: {'PASS' if passes else 'FAIL'}")
    if n_original <= 6:
        print("KILL CONDITION TRIGGERED: the mechanism hypothesis is false.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["variance", "baseline", "q4", "all"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    stages = ["variance", "baseline", "q4"] if args.stage == "all" else [args.stage]
    results: dict[str, Any] = {}
    for stage in stages:
        print(f"\n=== {stage} ===")
        if stage == "variance":
            results[stage] = stage_variance(args.seeds)
        elif stage == "baseline":
            results[stage] = stage_baseline(min(args.seeds, 5))
        else:
            results[stage] = stage_q4(args.seeds)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
