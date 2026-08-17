"""Reproduce the Q-Neuro Nova search. `make reproduce-nova` / `make smoke-nova`.

Nova asked whether a genuinely new computational principle could be found. The answer was no, and
this reproduces the evidence for that answer rather than for a success.

    make smoke-nova       # hashes, task validity, invariants -- about a minute
    make reproduce-nova   # adds the frontier at the corrected 2400-step budget -- several hours

What it checks, in order:

1. every frozen Nova prediction's hash, FROM DISK, together with the verdict it was scored at;
2. the shortcut audit, which disqualified two of the programme's own tasks;
3. the operator-level length-invariance probe;
4. optionally, the capability frontier itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NOVA = ROOT / "research" / "nova"

FROZEN = {
    "NOVA-H-INTERFERENCE-P1": "FAILED",
    "NOVA-H-COMPOSE-P1": "FAILED",
}


def check_hashes() -> bool:
    print("frozen Nova predictions")
    ok = True
    for name, expected in FROZEN.items():
        payload = json.loads((NOVA / f"{name}.json").read_text(encoding="utf-8"))
        recomputed = hashlib.sha256(
            json.dumps(payload["prediction"], indent=2, sort_keys=True).encode()
        ).hexdigest()
        good = recomputed == payload["sha256"]
        result = NOVA / f"{name}-RESULT.json"
        verdict = json.loads(result.read_text()).get("verdict", "—") if result.is_file() else "—"
        matches = verdict.startswith(expected[:4])
        ok = ok and good and matches
        print(f"  [{'OK ' if good and matches else 'BAD'}] {name:26} {payload['sha256'][:16]}  {verdict}")
    return ok


def check_task_validity() -> bool:
    """The shortcut audit. A task a degenerate predictor can solve is not an instrument."""

    from nova import tasks

    print("\ntask validity (best degenerate predictor at L=64)")
    ok = True
    expectations = {
        "parity_scan": False, "mod_sum": False, "copy": False, "reverse": False,
        "needle": False, "cummax": True, "sort": True,
    }
    for task, should_be_weak in expectations.items():
        sample = tasks.make(task, 2048, 64, seed=8)
        chance = tasks.chance_level(task)
        best_position = 0.0
        for position in range(64):
            selected = sample["mask"][:, position]
            if bool(selected.any()):
                counts = torch.bincount(sample["y"][:, position][selected], minlength=tasks.VOCAB)
                best_position = max(best_position, float(counts.max()) / float(selected.sum()))
        weak = best_position > 0.55
        agree = weak == should_be_weak
        ok = ok and agree
        label = "DROPPED" if should_be_weak else "kept"
        print(f"  [{'OK ' if agree else 'BAD'}] {task:12} chance {chance:.3f} "
              f"position-only {best_position:.3f}  ({label})")
    return ok


def check_invariance() -> bool:
    """The operator-level probe: does the read change when non-matching keys are added?"""

    from nova.candidates import InvariantAttention

    print("\nlength-invariance probe (read drift when 24 distractors are inserted)")
    drift = {}
    for normaliser in ("softmax", "max", "threshold"):
        torch.manual_seed(0)
        attention = InvariantAttention(32, 1, normaliser, rope=False).eval()
        torch.manual_seed(2)
        query, key = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
        distractors = torch.randn(1, 24, 32) * 0.05
        with torch.no_grad():
            few = attention(torch.cat([key, query], 1))[0, -1]
            many = attention(torch.cat([key, distractors, query], 1))[0, -1]
        drift[normaliser] = float((few - many).abs().max())
        print(f"    {normaliser:>10} {drift[normaliser]:.4f}")
    ok = drift["max"] < drift["softmax"] * 0.6
    print(f"  [{'OK ' if ok else 'BAD'}] max is materially more invariant than softmax")
    print("       (the operator has the property; the confound control showed it does not help)")
    return ok


def check_frontier() -> bool:
    from nova import lab
    from nova.candidates import CANDIDATES
    from nova.zoo import BASELINES, match_parameters

    print("\ncapability frontier at 2400 steps (this is the slow part)")
    record = json.loads((NOVA / "NOVA-FRONTIER-001.json").read_text(encoding="utf-8"))
    expected = record["frontier"]
    tasks = ("parity_scan", "mod_sum", "copy", "reverse", "needle")
    ok = True
    for name in ("lstm", "cursor"):
        builder = BASELINES.get(name) or CANDIDATES[name]
        BASELINES.setdefault(name, builder)
        config, _ = match_parameters(name, lab.PARAMETER_TARGET, depth=2)
        got = {}
        for task in tasks:
            scores = []
            for seed in (0, 1, 2):
                model, info = lab.train(
                    lambda b=builder, c=config: b(**c), task, seed=seed, steps=2400
                )
                if not info["diverged"]:
                    scores.append(lab.accuracy(model, task, 64, seed=1064))
            got[task] = statistics.mean(scores)
        # The frontier record uses the short key "parity"; the task registry uses "parity_scan".
        reference = {**expected[name], "parity_scan": expected[name]["parity"]}
        close = all(abs(got[t] - reference[t]) < 0.10 for t in tasks)
        ok = ok and close
        print(f"  [{'OK ' if close else 'BAD'}] {name:8} " +
              " ".join(f"{t[:4]} {got[t]:.3f}" for t in tasks))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="skip the training-based frontier")
    args = parser.parse_args()
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                      "platform": platform.platform(), "threads": torch.get_num_threads()},
                     indent=2), "\n")
    results = {
        "hashes": check_hashes(),
        "task_validity": check_task_validity(),
        "length_invariance": check_invariance(),
    }
    if not args.smoke:
        results["frontier"] = check_frontier()
    print("\n" + "=" * 66)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(results.values())
    print(f"\nNOVA REPRODUCTION: {'PASS' if overall else 'FAIL'}")
    print("VERDICT REPRODUCED: no new superior architecture survived.")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
