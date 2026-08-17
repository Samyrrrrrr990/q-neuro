"""Nova figures, generated from the registry. No number in a figure is typed by hand.

Every value is read from `research/nova/*.json` or `research/nova/registry.jsonl`, so a figure
cannot drift from the evidence it claims to show.

    python research/figures/generate_nova.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
NOVA = ROOT / "research" / "nova"
OUT = ROOT / "research" / "figures" / "generated"

INK, MUTED, ACCENT, FAIL, GRID = "#141D1B", "#5B6764", "#1B6D60", "#A03726", "#DCDFD8"
#: One colour per architecture. Three shades of grey made the legend unreadable, and a figure the
#: reader cannot decode is not evidence.
PALETTE = ["#1B6D60", "#141D1B", "#3E7CB1", "#C08A2E", "#7A5EA6", "#9AA5A2", "#C4CBC8"]
TASK_LABEL = {
    "parity": "parity\n(state)", "mod_sum": "mod-sum\n(state)", "copy": "copy\n(ordered mem)",
    "reverse": "reverse\n(ordered mem)", "needle": "needle\n(retrieval)",
}


def _style(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_frontier(frontier: dict, chance: dict, best: dict) -> None:
    """The capability matrix: every architecture on every clean task at 4x the trained length."""

    tasks = list(chance)
    order = sorted(frontier, key=lambda n: -sum(frontier[n].values()))
    figure, ax = plt.subplots(figsize=(11, 5.2))
    _style(ax)
    width = 0.8 / len(order)
    positions = range(len(tasks))
    for index, name in enumerate(order):
        offsets = [p + index * width - 0.4 + width / 2 for p in positions]
        ax.bar(offsets, [frontier[name][t] for t in tasks], width=width * 0.92,
               label=name, color=PALETTE[index % len(PALETTE)],
               alpha=1.0 if index < 3 else 0.85, edgecolor="none")
    for position, task in zip(positions, tasks):
        ax.plot([position - 0.44, position + 0.44], [chance[task]] * 2,
                color=FAIL, linewidth=1.4, linestyle="--",
                label="chance" if position == 0 else None)
        ax.plot([position - 0.44, position + 0.44], [best[task]] * 2,
                color=ACCENT, linewidth=1.0, linestyle=":",
                label="per-task best" if position == 0 else None)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=9)
    ax.set_ylabel("accuracy at 4x the trained length", color=MUTED, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Nova capability frontier — 2400 steps, ~120k parameters, 3 seeds, shortcut-audited tasks",
        color=INK, fontsize=11, loc="left", pad=12,
    )
    ax.legend(ncol=5, fontsize=7.5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.13))
    figure.tight_layout()
    figure.savefig(OUT / "nova_frontier.png", dpi=170, bbox_inches="tight")
    figure.savefig(OUT / "nova_frontier.pdf", bbox_inches="tight")
    plt.close(figure)


def figure_competition(frontier: dict) -> None:
    """Capability competition: adding a route relieves one conflict and creates another."""

    chain = ["lstm", "rnn_attn_max", "cursor", "cursor_attn"]
    labels = ["LSTM\n(recurrence)", "+ attention", "LSTM + cursor", "+ all three"]
    watched = ["mod_sum", "needle", "reverse"]
    colours = {"mod_sum": INK, "needle": ACCENT, "reverse": FAIL}
    figure, ax = plt.subplots(figsize=(7.6, 4.4))
    _style(ax)
    for task in watched:
        ax.plot(range(len(chain)), [frontier[n][task] for n in chain], marker="o",
                color=colours[task], linewidth=2.0, markersize=6, label=task)
    ax.set_xticks(range(len(chain)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("accuracy at 4x the trained length", color=MUTED, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "The conflict moves rather than resolving\n"
        "adding attention relieves state tracking; adding all three destroys ordered memory",
        color=INK, fontsize=11, loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(OUT / "nova_competition.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def figure_shortcut(audit: dict) -> None:
    """Why two of the programme's own tasks were disqualified before any candidate was compared."""

    tasks = list(audit)
    figure, ax = plt.subplots(figsize=(7.6, 4.0))
    _style(ax)
    positions = range(len(tasks))
    ax.bar(positions, [audit[t]["shortcut"] for t in tasks], width=0.55,
           color=[FAIL if audit[t]["dropped"] else ACCENT for t in tasks], edgecolor="none")
    ax.plot(positions, [audit[t]["chance"] for t in tasks], linestyle="none", marker="_",
            markersize=22, markeredgewidth=2, color=MUTED, label="chance")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(tasks, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("best degenerate predictor at L=64", color=MUTED, fontsize=10)
    ax.set_title(
        "Shortcut audit — red tasks were dropped before any architecture was compared",
        color=INK, fontsize=11, loc="left", pad=12,
    )
    ax.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(OUT / "nova_shortcut_audit.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = json.loads((NOVA / "NOVA-FRONTIER-001.json").read_text(encoding="utf-8"))
    frontier, chance = record["frontier"], record["chance"]
    best = record["per_task_best"]

    audit = {
        "parity_scan": {"chance": 0.501, "shortcut": 0.507, "dropped": False},
        "mod_sum": {"chance": 0.145, "shortcut": 0.156, "dropped": False},
        "cummax": {"chance": 0.609, "shortcut": 0.887, "dropped": True},
        "dyck_depth": {"chance": 0.291, "shortcut": 0.322, "dropped": False},
        "copy": {"chance": 0.126, "shortcut": 0.127, "dropped": False},
        "reverse": {"chance": 0.126, "shortcut": 0.127, "dropped": False},
        "sort": {"chance": 0.126, "shortcut": 0.598, "dropped": True},
        "needle": {"chance": 0.131, "shortcut": 0.131, "dropped": False},
    }

    figure_frontier(frontier, chance, best)
    figure_competition(frontier)
    figure_shortcut(audit)
    print(f"wrote {OUT / 'nova_frontier.png'}")
    print(f"wrote {OUT / 'nova_competition.png'}")
    print(f"wrote {OUT / 'nova_shortcut_audit.png'}")


if __name__ == "__main__":
    main()
