"""Generate critical-ablation plots from the latest registered artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[2]
COMPLEX_ABLATIONS = {
    "complex_accumulator": ("Commutative\naccumulator", "#9B8ABD"),
    "two_channel_operator": ("Two-channel\nreal", "#C28B16"),
    "complex_magnitude_readout": ("No readout\ninterference", "#8B6BAE"),
    "complex_no_negative": ("No negative\nevidence", "#7952A0"),
    "complex_operator": ("Full complex\noperator", "#673C9E"),
}
DENSITY_RANKS = ("density_rank1", "density_rank2", "density_rank4")


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "critical_ablation_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed critical ablation suite found")
    return max(candidates)[1]


def metric(result: dict[str, Any], model: str, section: str, name: str) -> dict[str, float]:
    if section == "in_domain":
        return result["summary"][model][section][name]
    return result["summary"][model][section]["across_worlds"][name]


def errors(values: list[dict[str, float]]) -> np.ndarray:
    means = np.asarray([value["mean"] for value in values])
    return np.asarray(
        [
            [mean - value["ci95_low"] for mean, value in zip(means, values, strict=True)],
            [value["ci95_high"] - mean for mean, value in zip(means, values, strict=True)],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "research" / "figures" / "generated",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    result_directory = latest_result()
    result = json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))
    analysis = json.loads(
        (
            ROOT / "research" / "analyses" / "generated" / "critical_ablation_paired_effects.json"
        ).read_text(encoding="utf-8")
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.3), constrained_layout=True)
    models = list(COMPLEX_ABLATIONS)
    x = np.arange(len(models))
    shift_values = [metric(result, model, "moderate", "top1") for model in models]
    colors = [COMPLEX_ABLATIONS[model][1] for model in models]
    axes[0, 0].bar(
        x,
        [value["mean"] for value in shift_values],
        yerr=errors(shift_values),
        color=colors,
        capsize=3,
    )
    axes[0, 0].set_xticks(
        x,
        [COMPLEX_ABLATIONS[model][0] for model in models],
        rotation=12,
        ha="right",
    )
    axes[0, 0].set_ylim(0.35, 0.70)
    axes[0, 0].set_ylabel("Moderate-shift top-1")
    axes[0, 0].set_title("A  Complex mechanism ablations", loc="left", fontweight="bold")

    pair_values = [
        metric(result, model, "in_domain", "counterfactual_pair_accuracy") for model in models
    ]
    axes[0, 1].bar(
        x,
        [value["mean"] for value in pair_values],
        yerr=errors(pair_values),
        color=colors,
        capsize=3,
    )
    axes[0, 1].set_xticks(
        x,
        [COMPLEX_ABLATIONS[model][0] for model in models],
        rotation=12,
        ha="right",
    )
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_ylabel("Counterfactual pair accuracy")
    axes[0, 1].set_title(
        "B  Order resolution requires operator composition", loc="left", fontweight="bold"
    )

    rank_axis = np.asarray([1, 2, 4])
    for name, color, marker in (
        ("top1", "#2474B5", "o"),
        ("counterfactual_pair_accuracy", "#16847A", "s"),
    ):
        values = [metric(result, model, "in_domain", name) for model in DENSITY_RANKS]
        axes[1, 0].errorbar(
            rank_axis,
            [value["mean"] for value in values],
            yerr=errors(values),
            color=color,
            marker=marker,
            capsize=3,
            label="In-domain top-1" if name == "top1" else "Counterfactual pairs",
        )
    shifted = [metric(result, model, "moderate", "top1") for model in DENSITY_RANKS]
    axes[1, 0].errorbar(
        rank_axis,
        [value["mean"] for value in shifted],
        yerr=errors(shifted),
        color="#673C9E",
        marker="D",
        capsize=3,
        label="Moderate shift",
    )
    axes[1, 0].set_xticks(rank_axis)
    axes[1, 0].set_xlabel("Density factor rank K")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("C  More density rank does not help", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    contrast_names = (
        "complex_operator_minus_complex_accumulator",
        "complex_operator_minus_two_channel",
        "complex_operator_minus_magnitude_readout",
        "complex_operator_minus_no_negative_evidence",
    )
    labels = ("vs commutative", "vs two-channel", "vs no interference", "vs no negative")
    values = [analysis["contrasts"][name]["moderate_shift"]["top1"] for name in contrast_names]
    means = np.asarray([value["mean"] for value in values])
    axes[1, 1].barh(
        np.arange(len(values)),
        means,
        xerr=errors(values),
        color="#673C9E",
        capsize=3,
    )
    axes[1, 1].axvline(0.0, color="#555555", linewidth=0.8)
    axes[1, 1].set_yticks(np.arange(len(values)), labels)
    axes[1, 1].set_xlabel("Full complex minus ablation top-1")
    axes[1, 1].set_title("D  Paired effects across unseen worlds", loc="left", fontweight="bold")

    for axis in axes.flat:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle("Critical Q-Neuro ablations", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        (
            "Bars show means and Student-t 95% intervals. Shift effects use three unseen world means "
            f"after seed averaging. Source: {result_directory.name}; synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"critical_ablation_suite.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
