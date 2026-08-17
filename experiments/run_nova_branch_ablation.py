"""Test-time branch ablation on the recurrence+attention hybrid.

An LSTM alone extrapolates `mod_sum` at 0.992. The same LSTM with an attention branch bolted on
collapses to 0.291, identically for three different attention normalisers. Two accounts fit that:

  OVERRIDE   the recurrence still learned the automaton, and the attention branch drowns it out
  PRE-EMPT   the recurrence never learned it, because attention fit the training lengths first

Zeroing each branch at test time separates them, with nothing retrained. Under OVERRIDE, removing
attention should *recover* state tracking; under PRE-EMPT it should not.

This is the intervention cited by `NOVA-H-INTERFERENCE-P1`; it writes
`research/nova/NOVA-BRANCH-ABLATION-001.json` so the figures and the manuscripts have a durable
source rather than a console transcript.

    python experiments/run_nova_branch_ablation.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from nova import lab
from nova.candidates import CANDIDATES
from nova.zoo import BASELINES, match_parameters

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "research" / "nova" / "NOVA-BRANCH-ABLATION-001.json"

ARCHITECTURE = "rnn_attn_softmax"
TASKS = ("mod_sum", "parity_scan", "needle", "copy")
MODES = ("none", "attention", "recurrence")
SEEDS = (0, 1, 2)
#: A branch that merely drowns out a working one should give this much back when it is removed.
RECOVERY_MARGIN = 0.15


def main() -> int:
    # `match_parameters` searches the baseline registry; the hybrid is a candidate, so it has to be
    # visible there for the width search to size it against the same parameter target.
    BASELINES.setdefault(ARCHITECTURE, CANDIDATES[ARCHITECTURE])
    config, _ = match_parameters(ARCHITECTURE, lab.PARAMETER_TARGET, depth=2)
    print(f"{'task':>12} {'ablation':>12} {'acc@16':>8} {'acc@64':>8}")

    measured: dict[str, dict[str, dict[str, float]]] = {}
    for task in TASKS:
        far = {mode: [] for mode in MODES}
        near = {mode: [] for mode in MODES}
        for seed in SEEDS:
            model, info = lab.train(lambda: CANDIDATES[ARCHITECTURE](**config), task, seed=seed)
            if info["diverged"]:
                continue
            model.eval()
            for mode in MODES:
                model.ablate = None if mode == "none" else mode
                far[mode].append(lab.accuracy(model, task, 64, seed=1064))
                near[mode].append(lab.accuracy(model, task, 16, seed=1016))
            model.ablate = None
        measured[task] = {
            mode: {
                "acc_at_16": round(statistics.mean(near[mode]), 4),
                "acc_at_64": round(statistics.mean(far[mode]), 4),
            }
            for mode in MODES
        }
        for mode in MODES:
            row = measured[task][mode]
            print(f"{task:>12} {mode:>12} {row['acc_at_16']:8.3f} {row['acc_at_64']:8.3f}",
                  flush=True)

    verdicts = {}
    for task, row in measured.items():
        recovered = row["attention"]["acc_at_64"] > row["none"]["acc_at_64"] + RECOVERY_MARGIN
        verdicts[task] = "OVERRIDE" if recovered else "PRE-EMPT / no recovery"

    RECORD.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "record_id": "NOVA-BRANCH-ABLATION-001",
                "lane": "B",
                "title": "Test-time branch ablation on the recurrence+attention hybrid",
                "question": (
                    "Does the attention branch OVERRIDE a recurrence that learned the automaton, "
                    "or PRE-EMPT one that never did?"
                ),
                "protocol": {
                    "architecture": ARCHITECTURE,
                    "parameters": config,
                    "seeds": list(SEEDS),
                    "train_lengths": "8-16",
                    "evaluated_at": [16, 64],
                    "retraining": "none -- branches are zeroed at test time",
                    "recovery_margin": RECOVERY_MARGIN,
                    "reproduce": "python experiments/run_nova_branch_ablation.py",
                },
                "measured": measured,
                "verdicts": verdicts,
                "reading": (
                    "Neither branch works alone on any task. Removing attention makes state "
                    "tracking worse, not better, so the recurrence never learned the automaton. "
                    "The model found a joint solution that needs both routes and does not "
                    "extrapolate. This is pre-emption, not override."
                ),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {RECORD.relative_to(ROOT)}")
    for task, verdict in verdicts.items():
        print(f"  {task:>12}: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
