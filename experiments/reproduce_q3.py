"""Umbrella reproduction for every Q-Neuro 3.0 headline claim. `make reproduce-q3`.

Written to be run by someone with no knowledge of this repository. It verifies every frozen
prediction's hash **from disk**, re-derives the quantitative claims, and prints an explicit
pass/fail against criteria fixed in advance. Nothing here is tuned by what it finds.

    make reproduce-q3            # everything, ~90 min on the reference machine
    make reproduce-q3-quick      # hashes, environment and invariants only, ~1 min

WHAT IS CLAIMED, AND WHERE IT STOPS
-----------------------------------
Supervised predicate halting attains the optimal per-example compute allocation and gives a large
batch-1 latency saving at matched accuracy **on tasks that supply a genuine halt target**. On a real
dataset that does not supply one, it LOSES to confidence-based early exit and to ACT
(`QNEURO3-HAR-P1`, failed). The durable results are the execution-policy characterisation and the
measured boundaries, not the architecture. None of the mechanisms is novel; see
`docs/PRIOR_ART_RUNTIME.md` and `docs/GATE4_NOVELTY_AUDIT.md`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "research" / "qneuro3"

#: Every frozen prediction and the verdict it was scored at. A reproduction that silently changed a
#: verdict would be worse than one that failed, so both are checked.
FROZEN = {
    "QNEURO3-Q3-P1": "FAILED",
    "QNEURO3-Q4-P1": "FAILED",
    "QNEURO3-ATTRIB-P1": "FAILED",
    "QNEURO3-TRANSFER-P1": "FAILED",
    "QNEURO3-EXTRAP-P1": "FAILED",
    "QNEURO3-PARETO-P1": "FAILED",
    "QNEURO3-NICHE-P1": "PASSED",
    "QNEURO3-RUNTIME-P1": "FAILED",
    "QNEURO3-RUNTIME-P2": "FAILED",
    "QNEURO3-HAR-P1": "FAILED",
}


def check_hashes() -> bool:
    """A frozen prediction whose hash cannot be re-verified from disk is not frozen."""

    print("frozen prediction integrity")
    ok = True
    for name, expected_verdict in FROZEN.items():
        path = FROZEN_DIR / f"{name}.json"
        if not path.is_file():
            print(f"  [MISSING] {name}")
            ok = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = "sha256" if "sha256" in payload else "sha256_of_prediction"
        recomputed = hashlib.sha256(
            json.dumps(payload["prediction"], indent=2, sort_keys=True).encode()
        ).hexdigest()
        good = recomputed == payload[key]
        result = FROZEN_DIR / f"{name}-RESULT.json"
        verdict = "—"
        if result.is_file():
            verdict = json.loads(result.read_text(encoding="utf-8")).get("verdict", "—")
        matches = verdict.startswith(expected_verdict[:4])
        ok = ok and good and matches
        flag = "OK " if good and matches else "BAD"
        print(f"  [{flag}] {name:24} {payload[key][:16]}  verdict {verdict:8} (expected {expected_verdict})")
    return ok


def check_invariants() -> bool:
    """The properties that make every comparison in this programme meaningful."""

    from qneuro3.adaptive import expected_max_halt, first_arrival, plan
    from qneuro3.runtime import compacted, lockstep, verify_equivalence

    print("\ninvariants")
    ok = True

    class Scripted:
        def __init__(self, depths):
            self.depths, self.max_depth = torch.tensor(depths), max(depths)

        def init_state(self, batch):
            return {"depth": batch["depth"].clone(), "tag": batch["tag"].clone()}

        def step(self, state, position):
            halt = (position + 1 >= state["depth"]).float()
            logits = torch.stack([state["tag"].float(), -state["tag"].float()], dim=-1)
            return state, halt, logits

    depths = [3, 1, 4, 1, 5, 2, 6, 2]
    core = Scripted(depths)
    batch = {"depth": torch.tensor(depths), "tag": torch.arange(len(depths))}
    reference = lockstep(core, batch)
    try:
        verify_equivalence(reference, compacted(core, batch))
        print("  [OK ] execution policies produce identical answers")
    except ValueError as error:
        print(f"  [BAD] policies disagree: {error}")
        ok = False

    if compacted(core, batch).example_steps == sum(depths):
        print(f"  [OK ] compaction executes sum(d_i) = {sum(depths)} rows, not n*max = {8 * 6}")
    else:
        print("  [BAD] compaction does not reach the ideal row count")
        ok = False

    p = torch.rand(64, 8)
    total = first_arrival(p)[0].exp().sum(dim=1)
    if bool((total <= 1.0 + 1e-6).all()):
        print("  [OK ] first-arrival masses never exceed one")
    else:
        print("  [BAD] first-arrival distribution is malformed")
        ok = False

    pmf = 0.8 ** torch.arange(1, 33, dtype=torch.double)
    values = [expected_max_halt(pmf, n) for n in (1, 8, 64, 1024)]
    if values == sorted(values) and values[-1] > 29.0:
        print("  [OK ] E[max halt] rises 4.97 -> 29.42 with batch (the lockstep ceiling)")
    else:
        print("  [BAD] the batch-maximum ceiling does not reproduce")
        ok = False

    if plan(pmf, 256, step_cost_us=2.66).policy == "compacted" and (
        plan(pmf, 256, step_cost_us=0.33).policy != "compacted"
    ):
        print("  [OK ] planner selects compaction only when the step cost justifies it")
    else:
        print("  [BAD] planner does not encode the measured boundary")
        ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="hashes and invariants only")
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "threads": torch.get_num_threads(),
            },
            indent=2,
        ),
        "\n",
    )
    results = {"hashes": check_hashes(), "invariants": check_invariants()}

    if not args.quick:
        print("\nniche reproduction (QNEURO3-NICHE-P1, the one prediction that passed)")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "reproduce_q3_niche.py"),
             "--with-compaction"],
            cwd=ROOT, env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
            check=False,
        )
        results["niche"] = completed.returncode == 0

    print("\n" + "=" * 70)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(results.values())
    print(f"\nQ-NEURO 3.0 REPRODUCTION: {'PASS' if overall else 'FAIL'}")
    if args.quick:
        print("(--quick skipped the training-based reproduction)")
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
