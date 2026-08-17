"""Gates 2 and 7: the M2 benchmark and the reliability report for Q-Neuro 3.0.

Lane B measurement. Produces the numbers the final scorecard is scored against, and nothing here
interprets them.

Gate 2 asks for median latency, p95, throughput, peak memory and active FLOPs across the full batch
sweep, with the execution policy chosen rather than assumed. Gate 7 asks for seed success rate,
variance, catastrophic-run frequency, calibration, halting correctness and hyperparameter
sensitivity -- because cycle 1 established that a healthy-looking step counter can coexist with an
accuracy collapse, and a method that is fast most of the time and catastrophically wrong the rest
does not pass.

    python experiments/run_qneuro3_m2_benchmark.py sweep        # Gate 2
    python experiments/run_qneuro3_m2_benchmark.py reliability   # Gate 7
    python experiments/run_qneuro3_m2_benchmark.py all --output research/qneuro3/m2_report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
NODES, MAX_DEPTH, TAIL = 32, 24, 0.85
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def environment() -> dict[str, Any]:
    from qneuro3 import hardware

    profile = hardware.detect(quick=True)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "total_memory_gib": profile.total_memory_gib,
        "available_memory_gib": profile.available_memory_gib,
        "physical_cores": profile.physical_cores,
        "mps_available": profile.mps_available,
    }


def _rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes.
    return usage / 1024**2 if usage > 1 << 30 else usage / 1024


def _timed(function, repeats: int) -> tuple[float, float]:
    """Median and p95 microseconds. Both, because a tail is what a latency budget actually buys."""

    for _ in range(3):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    return statistics.median(samples), p95


def train_pair(seed: int = 0, *, epochs: int = 4, train_batches: int = 250):
    from research.qneuro3.decoupled import ArrivalQuery, SelectQuery, query_chase

    data = [
        query_chase(128, NODES, MAX_DEPTH, seed=1000 + i, tail=TAIL) for i in range(train_batches)
    ]
    cross_entropy = torch.nn.functional.cross_entropy
    models = {}
    for name, builder in (("arrival", ArrivalQuery), ("select", SelectQuery)):
        torch.manual_seed(seed)
        model = builder(n_nodes=NODES, max_depth=MAX_DEPTH, normalise=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        for _ in range(epochs):
            for batch in data:
                optimizer.zero_grad(set_to_none=True)
                _, _, per_step, stacked = model(batch)
                index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
                at_truth = stacked.gather(1, index).squeeze(1)
                if name == "arrival":
                    loss = -per_step.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
                else:
                    loss = cross_entropy(per_step, batch["target"] - 1)
                (loss + cross_entropy(model.name(at_truth), batch["answer"])).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        models[name] = model.eval()
    return models


def stage_sweep() -> list[dict[str, Any]]:
    from qneuro3.adaptive import plan
    from qneuro3.runtime import compacted, lockstep, verify_equivalence
    from research.qneuro3.decoupled import query_chase
    from research.qneuro3.runtime_adapters import QueryAdapter

    models = train_pair()
    adapter = QueryAdapter(models["arrival"], MAX_DEPTH)
    select = models["select"]

    @torch.no_grad()
    def select_forward(batch):
        core = select.core
        context = core.context(batch["perm"], batch["labels"])
        query = core.query(batch["query"])
        state = core.key(batch["start"])
        carried = torch.zeros_like(state)
        scored, kept = [], []
        for _ in range(MAX_DEPTH):
            state, carried = core.advance(state, context, query, carried)
            scored.append(
                select.score(
                    torch.cat([state, carried, query, carried * query], dim=-1)
                ).squeeze(-1)
            )
            kept.append(state)
        stacked = torch.stack(kept, 1)
        chosen = torch.stack(scored, 1).argmax(1)
        return select.name(
            stacked.gather(1, chosen.view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])).squeeze(1)
        )

    pmf = TAIL ** torch.arange(1, MAX_DEPTH + 1, dtype=torch.double)
    rows = []
    print(
        f"{'batch':>6} {'policy':>10} {'median us/ex':>13} {'p95 us/ex':>11} "
        f"{'thruput/s':>11} {'rows/ex':>8} {'dRSS MiB':>9} {'chosen':>14}"
    )
    for size in BATCHES:
        batch = query_chase(size, NODES, MAX_DEPTH, seed=4242, tail=TAIL)
        reference = lockstep(adapter, batch)
        verify_equivalence(reference, compacted(adapter, batch))
        repeats = 25 if size <= 16 else 9
        decision = plan(pmf, size, MAX_DEPTH, step_cost_us=2.66)
        for policy, function, rows_per in (
            ("select", lambda b=batch: select_forward(b), float(MAX_DEPTH)),
            ("lockstep", lambda b=batch: lockstep(adapter, b), reference.example_steps / size),
            (
                "compacted",
                lambda b=batch: compacted(adapter, b),
                compacted(adapter, batch).example_steps / size,
            ),
        ):
            before = _rss_mib()
            median, p95 = _timed(function, repeats)
            rows.append(
                {
                    "batch": size,
                    "policy": policy,
                    "median_us_per_example": median / size,
                    "p95_us_per_example": p95 / size,
                    "throughput_per_second": 1e6 / (median / size),
                    "rows_per_example": rows_per,
                    "rss_delta_mib": max(0.0, _rss_mib() - before),
                    "planner_choice": decision.policy,
                }
            )
            print(
                f"{size:6d} {policy:>10} {median / size:13.1f} {p95 / size:11.1f} "
                f"{1e6 / (median / size):11.0f} {rows_per:8.2f} "
                f"{max(0.0, _rss_mib() - before):9.2f} {decision.policy:>14}"
            )
    return rows


def stage_reliability(seeds: int = 10) -> dict[str, Any]:
    from qneuro3.runtime import lockstep
    from research.qneuro3.decoupled import query_chase
    from research.qneuro3.runtime_adapters import QueryAdapter

    validation = [
        query_chase(256, NODES, MAX_DEPTH, seed=90000 + i, tail=TAIL) for i in range(10)
    ]

    def measure(model):
        adapter = QueryAdapter(model, MAX_DEPTH)
        correct = total = halt_correct = 0
        confidence, hits = [], []
        with torch.no_grad():
            for batch in validation:
                run = lockstep(adapter, batch)
                predicted = run.answers.argmax(-1)
                probability = run.answers.softmax(-1).max(-1).values
                correct += int((predicted == batch["answer"]).sum())
                halt_correct += int((run.steps == batch["target"].float()).sum())
                total += len(batch["answer"])
                confidence.append(probability)
                hits.append((predicted == batch["answer"]).float())
        confidence = torch.cat(confidence)
        hits = torch.cat(hits)
        # Expected calibration error, 10 equal-width bins.
        ece = 0.0
        for lower in torch.linspace(0, 0.9, 10):
            mask = (confidence >= lower) & (confidence < lower + 0.1)
            if bool(mask.any()):
                ece += float(mask.float().mean()) * abs(
                    float(confidence[mask].mean()) - float(hits[mask].mean())
                )
        return correct / total, halt_correct / total, ece

    print(f"{'seed':>5} {'accuracy':>9} {'halt acc':>9} {'ECE':>7}")
    accuracies, halts, eces = [], [], []
    for seed in range(seeds):
        model = _train_arrival(seed)
        accuracy, halt, ece = measure(model)
        accuracies.append(accuracy)
        halts.append(halt)
        eces.append(ece)
        print(f"{seed:5d} {accuracy:9.4f} {halt:9.4f} {ece:7.4f}")

    print(f"\n{'lr':>8} {'halt bias':>10} {'accuracy':>9}")
    sensitivity = []
    for learning_rate in (5e-4, 1e-3, 2e-3, 4e-3):
        for bias in (-4.0, -2.0, 0.0):
            model = _train_arrival(0, learning_rate=learning_rate, halt_bias=bias)
            accuracy, halt, _ = measure(model)
            sensitivity.append(
                {"learning_rate": learning_rate, "halt_bias": bias, "accuracy": accuracy,
                 "halt_accuracy": halt}
            )
            print(f"{learning_rate:8.4f} {bias:10.1f} {accuracy:9.4f}")

    reliable = sum(a >= 0.99 for a in accuracies)
    catastrophic = sum(a < 0.5 for a in accuracies)
    return {
        "seeds": seeds,
        "accuracies": accuracies,
        "halt_accuracies": halts,
        "expected_calibration_error": eces,
        "seed_success_rate": reliable / seeds,
        "catastrophic_rate": catastrophic / seeds,
        "accuracy_stdev": statistics.pstdev(accuracies),
        "hyperparameter_sensitivity": sensitivity,
        "hyperparameter_success_rate": sum(s["accuracy"] >= 0.99 for s in sensitivity)
        / len(sensitivity),
    }


def _train_arrival(seed: int, *, learning_rate: float = 2e-3, halt_bias: float = -2.0,
                   epochs: int = 4, train_batches: int = 250):
    from research.qneuro3.decoupled import ArrivalQuery, query_chase

    data = [
        query_chase(128, NODES, MAX_DEPTH, seed=1000 + i, tail=TAIL) for i in range(train_batches)
    ]
    cross_entropy = torch.nn.functional.cross_entropy
    torch.manual_seed(seed)
    model = ArrivalQuery(n_nodes=NODES, max_depth=MAX_DEPTH, normalise=True, halt_bias=halt_bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        for batch in data:
            optimizer.zero_grad(set_to_none=True)
            _, _, log_first, stacked = model(batch)
            index = (batch["target"] - 1).view(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
            at_truth = stacked.gather(1, index).squeeze(1)
            loss = -log_first.gather(1, (batch["target"] - 1).unsqueeze(1)).mean()
            (loss + cross_entropy(model.name(at_truth), batch["answer"])).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sweep", "reliability", "all"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report: dict[str, Any] = {"environment": environment()}
    print(json.dumps(report["environment"], indent=2), "\n")
    if args.stage in ("sweep", "all"):
        print("=== Gate 2: M2 batch sweep ===")
        report["sweep"] = stage_sweep()
    if args.stage in ("reliability", "all"):
        print("\n=== Gate 7: reliability ===")
        report["reliability"] = stage_reliability(args.seeds)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
