"""Standalone reproduction of the one Q-Neuro 3.0 result that survived the full promotion path.

Run with `make reproduce-q3-niche`, or directly:

    PYTHONPATH="$PWD" python experiments/reproduce_q3_niche.py

This script is written to be readable by someone with no context from the rest of the repository.
It re-derives the confirmed result end to end, verifies the frozen prediction's hash from disk
before scoring anything, and prints an explicit PASS/FAIL against criteria that were fixed in
advance. Runtime is roughly 15 minutes on the reference machine.

WHAT IS BEING REPRODUCED
------------------------
`QNEURO3-NICHE-P1` (sha256 7fbcceb8...) is the only frozen prediction in this programme to pass as
written; twelve preceded it and all twelve failed. It states, and the run below checks, that on a
workload with a deep worst case and heavy-tailed difficulty a model that halts on a supervised
predicate:

  N1  matches a full-depth baseline's accuracy (within 0.02),
  N2  attains the OPTIMAL per-example allocation (mean steps within 0.5 of E[distance]),
  N3  is at least 2.5x faster in wall-clock at batch 1, and
  N4  loses that advantage at batch 256 (at most 1.2x) under lockstep execution.

N4 is the clause that matters: the prediction names where the result stops working, and that clause
was derived on a different task family from the one it is tested on here.

WHAT IS NOT BEING CLAIMED
-------------------------
Nothing here is novel. Supervised early exit, per-step attribution and state normalisation are all
prior art (`docs/PRIOR_ART_RUNTIME.md`, `research/qneuro3/QNEURO3-ATTRIBUTION-001.json`). The size
of the saving is `max_depth / E[distance]`, a property of the workload rather than of the
architecture. The contribution is the measured boundary, prospectively confirmed.

A LATER RESULT NARROWS N4
-------------------------
`QNEURO3-SCOPE-CORRECTION-001` records that N4's ceiling belongs to the LOCKSTEP execution policy.
Under active-set compaction the same models recover to 1.95x at batch 256 on this family --
see `--with-compaction` below. That does not affect the frozen prediction, which was about the
policy it measured, and this script reproduces both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "research" / "qneuro3" / "QNEURO3-NICHE-P1.json"

NODES, MAX_DEPTH, TAIL = 32, 24, 0.85
SEEDS = (0, 1, 2)
LEARNING_RATE, EPOCHS, BATCH, TRAIN_BATCHES = 2e-3, 8, 128, 500


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "threads": str(torch.get_num_threads()),
    }


def load_frozen() -> dict:
    """Verify the prediction hash FROM DISK before any evidence is generated.

    A prediction whose hash cannot be re-verified from disk is not frozen; that rule was itself
    learned the hard way (`QNEURO3-RUNTIME-P1`, freeze_procedure_defect).
    """

    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    recomputed = hashlib.sha256(
        json.dumps(payload["prediction"], indent=2, sort_keys=True).encode()
    ).hexdigest()
    if recomputed != payload["sha256"]:
        raise SystemExit(
            f"FROZEN PREDICTION HASH MISMATCH\n  expected {payload['sha256']}\n"
            f"  recomputed {recomputed}\nThe record was modified after freezing; refusing to score."
        )
    return payload


def train(kind: str, seed: int, *, epochs: int, train_batches: int):
    from research.qneuro3.decoupled import ArrivalQuery, SelectQuery, query_chase

    builder = ArrivalQuery if kind == "arrival" else SelectQuery
    data = [
        query_chase(BATCH, NODES, MAX_DEPTH, seed=1000 + i, tail=TAIL)
        for i in range(train_batches)
    ]
    torch.manual_seed(seed)
    model = builder(n_nodes=NODES, max_depth=MAX_DEPTH, normalise=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    cross_entropy = torch.nn.functional.cross_entropy
    for _ in range(epochs):
        for batch in data:
            optimizer.zero_grad(set_to_none=True)
            _, _, per_step, stacked = model(batch)
            index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            at_truth = stacked.gather(1, index).squeeze(1)
            if kind == "arrival":
                loss = -per_step.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
            else:
                loss = cross_entropy(per_step, batch["target"] - 1)
            (loss + cross_entropy(model.name(at_truth), batch["answer"])).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def median_us(function, repeats: int) -> float:
    for _ in range(3):
        function()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        timings.append((time.perf_counter() - start) * 1e6)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="1 seed, short training: smoke only")
    parser.add_argument(
        "--with-compaction", action="store_true",
        help="also measure the active-set compaction policy, which narrows N4",
    )
    args = parser.parse_args()

    from qneuro3.runtime import compacted, lockstep, verify_equivalence
    from research.qneuro3.decoupled import query_chase
    from research.qneuro3.runtime_adapters import QueryAdapter

    payload = load_frozen()
    print(f"frozen prediction verified from disk: {payload['sha256'][:16]}")
    for name, text in payload["prediction"]["predictions"].items():
        print(f"  {name}: {text}")
    print(f"\nenvironment: {json.dumps(environment())}\n")

    seeds = SEEDS[:1] if args.quick else SEEDS
    epochs = 2 if args.quick else EPOCHS
    train_batches = 100 if args.quick else TRAIN_BATCHES

    validation = [
        query_chase(256, NODES, MAX_DEPTH, seed=90000 + i, tail=TAIL) for i in range(25)
    ]
    expected_distance = float(
        torch.cat([b["target"] for b in validation]).float().mean()
    )

    scores: dict[str, list[float]] = {"arrival": [], "select": []}
    steps: list[float] = []
    for kind in ("select", "arrival"):
        for seed in seeds:
            model = train(kind, seed, epochs=epochs, train_batches=train_batches)
            adapter = QueryAdapter(model, MAX_DEPTH) if kind == "arrival" else None
            correct = total = 0
            step_sum = 0.0
            with torch.no_grad():
                for batch in validation:
                    if kind == "arrival":
                        run = lockstep(adapter, batch)
                        predicted, run_steps = run.answers.argmax(-1), run.steps
                    else:
                        logits, _, _, _ = model(batch)
                        predicted = logits.argmax(-1)
                        run_steps = torch.full((len(predicted),), float(MAX_DEPTH))
                    correct += int((predicted == batch["answer"]).sum())
                    total += len(batch["answer"])
                    step_sum += float(run_steps.mean())
            scores[kind].append(correct / total)
            if kind == "arrival":
                steps.append(step_sum / len(validation))
            print(f"  {kind:>8} seed {seed}: accuracy {correct / total:.4f}")

    arrival_model = train("arrival", seeds[0], epochs=epochs, train_batches=train_batches)
    select_model = train("select", seeds[0], epochs=epochs, train_batches=train_batches)
    adapter = QueryAdapter(arrival_model, MAX_DEPTH)

    @torch.no_grad()
    def select_forward(batch):
        core = select_model.core
        context = core.context(batch["perm"], batch["labels"])
        query = core.query(batch["query"])
        state = core.key(batch["start"])
        carried = torch.zeros_like(state)
        scored, kept = [], []
        for _ in range(MAX_DEPTH):
            state, carried = core.advance(state, context, query, carried)
            feature = torch.cat([state, carried, query, carried * query], dim=-1)
            scored.append(select_model.score(feature).squeeze(-1))
            kept.append(state)
        stacked = torch.stack(kept, 1)
        chosen = torch.stack(scored, 1).argmax(1)
        return select_model.name(
            stacked.gather(1, chosen.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])).squeeze(1)
        )

    print("\nwall-clock, matched accuracy (microseconds per example)")
    header = f"{'batch':>6} {'select':>10} {'arrival':>10} {'speedup':>9}"
    if args.with_compaction:
        header += f" {'compacted':>10} {'vs select':>10}"
    print(header)
    latency: dict[int, dict[str, float]] = {}
    for size in (1, 256):
        batch = query_chase(size, NODES, MAX_DEPTH, seed=4242, tail=TAIL)
        repeats = 25 if size == 1 else 7
        if size == 1:
            trials = [
                query_chase(1, NODES, MAX_DEPTH, seed=70000 + i, tail=TAIL) for i in range(repeats)
            ]
            select_us = statistics.mean(median_us(lambda b=b: select_forward(b), 5) for b in trials)
            arrival_us = statistics.mean(
                median_us(lambda b=b: lockstep(adapter, b), 5) for b in trials
            )
        else:
            select_us = median_us(lambda b=batch: select_forward(b), repeats) / size
            arrival_us = median_us(lambda b=batch: lockstep(adapter, b), repeats) / size
        row = {"select": select_us, "arrival": arrival_us, "speedup": select_us / arrival_us}
        line = f"{size:6d} {select_us:10.1f} {arrival_us:10.1f} {row['speedup']:8.2f}x"
        if args.with_compaction:
            verify_equivalence(lockstep(adapter, batch), compacted(adapter, batch))
            compacted_us = median_us(lambda b=batch: compacted(adapter, b), repeats) / size
            row["compacted"] = compacted_us
            row["compacted_vs_select"] = select_us / compacted_us
            line += f" {compacted_us:10.1f} {select_us / compacted_us:9.2f}x"
        latency[size] = row
        print(line)

    arrival_mean = statistics.mean(scores["arrival"])
    select_mean = statistics.mean(scores["select"])
    step_mean = statistics.mean(steps)
    print("\nSCORING against the frozen criteria")
    results = {
        "N1 accuracy within 0.02 of select": abs(arrival_mean - select_mean) <= 0.02,
        "N2 mean steps within 0.5 of E[distance]": abs(step_mean - expected_distance) <= 0.5,
        "N3 batch-1 speedup >= 2.5x": latency[1]["speedup"] >= 2.5,
        "N4 batch-256 speedup <= 1.2x (lockstep)": latency[256]["speedup"] <= 1.2,
    }
    print(f"  N1  arrival {arrival_mean:.4f} vs select {select_mean:.4f}")
    print(f"  N2  steps {step_mean:.2f} vs E[distance] {expected_distance:.2f} of {MAX_DEPTH}")
    print(f"  N3  batch-1 {latency[1]['speedup']:.2f}x")
    print(f"  N4  batch-256 {latency[256]['speedup']:.2f}x")
    if args.with_compaction:
        print(
            f"      under compaction, batch-256 becomes "
            f"{latency[256]['compacted_vs_select']:.2f}x -- the ceiling is a lockstep property"
        )
    if args.quick:
        print(
            "\n  SMOKE TEST ONLY -- mechanics verified (frozen hash, training loop, both execution\n"
            "  policies, equivalence check, scoring path). At 2 epochs over 100 batches the models\n"
            "  do not train, so these numbers are NOT the confirmed result and are not scored.\n"
            "  Run `make reproduce-q3-niche` for the real thing."
        )
        raise SystemExit(0)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(results.values())
    print(f"\nQNEURO3-NICHE-P1 REPRODUCTION: {'PASS' if overall else 'FAIL'}")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
