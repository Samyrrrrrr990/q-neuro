"""Reproduce Q-Neuro 3.0 cycle 2: supervised predicate halting, and the ceiling on its advantage.

Lane B. Records live in `research/qneuro3/`; the narrative is in `docs/MONOGRAPH.md` part VI.

Cycle 1 closed negative: adaptive depth lost to fixed depth once reliability was counted. Cycle 2
found the cause (an unnormalised recurrent state makes a fixed halting threshold depth-dependent),
fixed it, and then spent most of its effort trying to destroy the result that followed.

Stages, each independently runnable:

* ``reliability``  — the variant sweep that turned an 11-of-24 seed lottery into 6 of 6.
* ``mechanism``    — does normalisation rescue halting in general, or only predicate halting?
* ``attribution``  — where the answer is read from, on the associative-lookup family.
* ``transfer``     — opens frozen QNEURO3-TRANSFER-P1 on the streaming family. FAILS by design.
* ``niche``        — opens frozen QNEURO3-NICHE-P1. The one prediction that passed.
* ``ceiling``      — the analytic batch-maximum curve, no training required.

    python experiments/run_qneuro3_cycle_002.py ceiling      # instant, and the key result
    python experiments/run_qneuro3_cycle_002.py all

Expected: transfer FAILS, niche PASSES. Those are the results, not errors in reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from qneuro3.adaptive import MEASURED_CROSSOVER_BATCH, expected_max_halt, plan

ROOT = Path(__file__).resolve().parents[1]


def stage_ceiling() -> list[dict[str, Any]]:
    """Why the advantage has a ceiling. Pure arithmetic; no model, no training, no seeds."""

    pmf = 0.8 ** torch.arange(1, 33, dtype=torch.double)
    mean = float(
        (torch.arange(1, 33, dtype=torch.double) * (pmf / pmf.sum())).sum()
    )
    print("A batch cannot exit until its slowest member does.")
    print(f"P(halt = k) proportional to 0.8^k on 1..32   =>   E[halt] = {mean:.2f}\n")
    print(f"{'batch':>7} {'E[max halt]':>12} {'realisable saving':>18} {'mode chosen':>16}")
    rows = []
    for batch in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
        maximum = expected_max_halt(pmf, batch)
        chosen = plan(pmf, batch)
        rows.append(
            {
                "batch": batch,
                "expected_max_halt": maximum,
                "realisable_saving": 32 / maximum,
                "mode": chosen.mode,
                "early_exit": chosen.early_exit,
            }
        )
        print(f"{batch:7d} {maximum:12.2f} {32 / maximum:17.2f}x {chosen.mode:>16}")
    print(
        f"\nMeasured crossover on the M2 reference machine: batch {MEASURED_CROSSOVER_BATCH}. "
        "Above it, early exit is a measured 0.97-0.99x, i.e. a penalty."
    )
    return rows


def stage_reliability(seeds: int) -> list[dict[str, Any]]:
    from research.qneuro3.variants import VARIANTS, train_and_evaluate

    rows = []
    print(f"{'variant':>18} {'seed':>5} {'acc':>8} {'steps':>7}   accuracy by distance 1..8")
    for name in VARIANTS:
        for seed in range(seeds):
            record = train_and_evaluate(name, seed)
            rows.append(record)
            profile = " ".join(f"{v:.2f}" for v in record["accuracy_by_distance"])
            print(
                f"{name:>18} {seed:5d} {record['accuracy']:8.4f} "
                f"{record['mean_steps']:7.2f}   {profile}"
            )
    for name in VARIANTS:
        subset = [r for r in rows if r["variant"] == name]
        reliable = sum(r["accuracy"] >= 0.99 for r in subset)
        print(f"  {name:>18}: {reliable}/{len(subset)} seeds at accuracy >= 0.99")
    return rows


def stage_attribution(seeds: int) -> list[dict[str, Any]]:
    from research.qneuro3.decoupled import run

    rows = []
    print(f"{'model':>18} {'seed':>5} {'answer':>8} {'step-id':>8} {'steps':>7}")
    print("  (chance 0.042; guessing any node carrying the query label gives 0.291)")
    for kind in ("fixed", "fixed_supervised", "gated", "mean_pooled", "select", "arrival"):
        for seed in range(seeds):
            record = run(kind, seed, normalise=True)
            rows.append(record)
            step_id = record["halt_accuracy"]
            shown = f"{step_id:8.4f}" if step_id is not None else "     n/a"
            print(
                f"{kind:>18} {seed:5d} {record['answer_accuracy']:8.4f} {shown} "
                f"{record['mean_steps']:7.2f}"
            )
    return rows


def stage_transfer(seeds: int) -> list[dict[str, Any]]:
    from research.qneuro3.streaming import run

    print("Opening frozen QNEURO3-TRANSFER-P1. Expected verdict: FAIL (T1 and T2).")
    rows = []
    for kind in ("select", "arrival", "fixed", "fixed_supervised", "gated", "mean_pooled"):
        for seed in range(seeds):
            record = run(kind, seed)
            rows.append(record)
            print(
                f"{kind:>18} {seed:5d} answer {record['answer_accuracy']:.4f} "
                f"steps {record['mean_steps']:.2f}"
            )
    return rows


def stage_niche(seeds: int) -> list[dict[str, Any]]:
    from research.qneuro3.decoupled import run

    print("Opening frozen QNEURO3-NICHE-P1 (accuracy and allocation halves).")
    rows = []
    for kind in ("select", "arrival"):
        for seed in range(seeds):
            record = run(kind, seed, normalise=True, max_depth=24, n_nodes=32, tail=0.85)
            rows.append(record)
            print(
                f"{kind:>8} {seed:3d} answer {record['answer_accuracy']:.4f} "
                f"steps {record['mean_steps']:.2f} of 24  E[d] {record['expected_distance']:.2f}"
            )
    print("\nThe wall-clock halves (N3, N4) need genuine early exit; see")
    print("research/qneuro3/QNEURO3-NICHE-P1-RESULT.json for the measured crossover curve.")
    return rows


STAGES = {
    "ceiling": lambda seeds: stage_ceiling(),
    "reliability": stage_reliability,
    "attribution": stage_attribution,
    "transfer": stage_transfer,
    "niche": stage_niche,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    names = list(STAGES) if args.stage == "all" else [args.stage]
    results: dict[str, Any] = {}
    for name in names:
        print(f"\n=== {name} ===")
        results[name] = STAGES[name](args.seeds)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
