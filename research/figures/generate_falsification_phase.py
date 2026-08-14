"""Generate the central next-phase falsification figure from immutable artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.computational_laws import _design, frozen_law_from_dict

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "figures" / "generated"
INK = "#17324D"
TEAL = "#168C86"
CORAL = "#D65A4A"
GOLD = "#D5A02B"
MIST = "#DDE9ED"


def load(experiment_id: str, filename: str = "metrics.json") -> dict:
    return json.loads(
        (ROOT / "experiments" / "results" / experiment_id / filename).read_text(encoding="utf-8")
    )


def save(figure: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main() -> None:
    synthesis = json.loads(
        (ROOT / "research" / "analyses" / "generated" / "falsification_phase.json").read_text()
    )
    confirmation = load("QN-000042")
    frozen = frozen_law_from_dict(confirmation["frozen_law"])
    cells = confirmation["law_cells"]
    order = np.asarray([item["order_information"] for item in cells])
    shift = np.asarray([item["severity"] for item in cells])
    observed = np.asarray([item["advantage"] for item in cells])
    predicted = _design(frozen.family, order, shift, frozen.hyperparameter) @ np.asarray(
        frozen.coefficients
    )

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    labels = [
        "Initial\nNeuroWorld",
        "Pilot\nbest-real",
        "Discovery\nbest-real",
        "Held-out\nbest-real",
    ]
    values = [
        synthesis["historical_within_neuroworld"]["moderate_shift_mean_difference"],
        synthesis["power_pilot"]["train_size_1000"]["mean"],
        synthesis["reduced_discovery"]["mean_nested_effect"],
        synthesis["heldout_confirmation"]["nested_summary"]["mean"],
    ]
    axes[0, 0].bar(
        np.arange(4), values, color=[TEAL if value > 0 else CORAL for value in values], width=0.68
    )
    axes[0, 0].axhline(0.0, color=INK, linewidth=1)
    axes[0, 0].set_xticks(np.arange(4), labels)
    axes[0, 0].set_ylabel("Complex minus comparator top-1")
    axes[0, 0].set_title(
        "A  Stronger comparators reverse the result", loc="left", fontweight="bold"
    )
    axes[0, 0].text(
        0.02,
        0.03,
        "Comparator changes after the first bar; values are not a single meta-analysis.",
        transform=axes[0, 0].transAxes,
        fontsize=8,
        color=INK,
    )

    axes[0, 1].scatter(predicted, observed, c=shift, cmap="viridis", s=65, edgecolor="white")
    low = min(predicted.min(), observed.min()) - 0.005
    high = max(predicted.max(), observed.max()) + 0.005
    axes[0, 1].plot([low, high], [low, high], linestyle="--", color=INK, linewidth=1)
    axes[0, 1].set(
        xlabel="Frozen predicted gap",
        ylabel="Held-out observed gap",
        xlim=(low, high),
        ylim=(low, high),
    )
    axes[0, 1].set_title("B  The quantitative law fails transfer", loc="left", fontweight="bold")
    axes[0, 1].text(
        0.03,
        0.94,
        f"R² = {confirmation['law_confirmation']['r2']:.2f}\nMAE = {confirmation['law_confirmation']['mean_absolute_error']:.3f}",
        transform=axes[0, 1].transAxes,
        va="top",
        fontsize=9,
    )

    winners = synthesis["heldout_confirmation"]["best_real_winner_counts"]
    winner_labels = [name.replace("_operator", "").replace("_", " ") for name in winners]
    axes[1, 0].barh(winner_labels, list(winners.values()), color=[INK, GOLD])
    axes[1, 0].set_xlabel("Held-out nested cells won")
    axes[1, 0].set_title("C  Real controls define the envelope", loc="left", fontweight="bold")
    axes[1, 0].text(
        0.98,
        0.06,
        "Complex top-1 = exact-real top-1\nin all 1,920 held-out cells",
        transform=axes[1, 0].transAxes,
        ha="right",
        fontsize=9,
        color="white",
        fontweight="bold",
    )

    preflight = load("QN-GRAND-001", "preflight.json")
    passed = sum(item["passed"] for item in preflight["checks"])
    failed = len(preflight["checks"]) - passed
    axes[1, 1].bar(["passed", "blocking failures"], [passed, failed], color=[TEAL, CORAL])
    axes[1, 1].set_ylabel("Preflight gates")
    axes[1, 1].set_title("D  QN-GRAND-001 remains sealed", loc="left", fontweight="bold")
    axes[1, 1].text(
        0.5,
        0.72,
        "No sealed data opened\nNo primary effect estimated",
        transform=axes[1, 1].transAxes,
        ha="center",
        fontweight="bold",
        color=INK,
    )

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Falsification phase: exact real equivalence removes the intrinsic complex claim",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    save(figure, "falsification_phase")


if __name__ == "__main__":
    main()
