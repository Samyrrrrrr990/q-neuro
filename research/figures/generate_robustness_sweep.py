"""Generate the multi-world robustness confirmation figure from QN artifacts."""

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
MODELS = {
    "mlp": ("MLP", "#777777", "o"),
    "transformer": ("Transformer", "#2474B5", "o"),
    "gru": ("Tuned GRU", "#16847A", "s"),
    "real_operator": ("Real operator", "#D05A2D", "o"),
    "two_channel_operator": ("Two-channel real", "#C28B16", "D"),
    "complex_operator": ("Complex operator", "#673C9E", "o"),
}
SEVERITIES = ("nuisance", "mild", "moderate", "severe")


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "robustness_world_sweep":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed robustness sweep found")
    return max(candidates)[1]


def aggregate(result: dict[str, Any], model: str, stage: str, metric: str) -> dict[str, Any]:
    if stage == "in_domain":
        return result["summary"][model][stage][metric]
    return result["summary"][model][stage]["across_worlds"][metric]


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

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.1), constrained_layout=True)

    stages = ("in_domain", *SEVERITIES)
    x = np.arange(len(stages))
    for model, (label, color, marker) in MODELS.items():
        values = [aggregate(result, model, stage, "top1") for stage in stages]
        means = np.asarray([value["mean"] for value in values])
        errors = np.asarray(
            [
                [mean - value["ci95_low"] for mean, value in zip(means, values, strict=True)],
                [value["ci95_high"] - mean for mean, value in zip(means, values, strict=True)],
            ]
        )
        axes[0, 0].errorbar(
            x,
            means,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            capsize=2,
            label=label,
        )
    axes[0, 0].set_title("A  Accuracy across shift severity", loc="left", fontweight="bold")
    axes[0, 0].set_xticks(x, ["ID", "Nuisance", "Mild", "Moderate", "Severe"])
    axes[0, 0].set_ylim(0.1, 1.03)
    axes[0, 0].set_ylabel("Top-1 accuracy")
    axes[0, 0].legend(frameon=False, ncols=2, loc="lower left")

    severity_x = np.arange(len(SEVERITIES))
    for model, (label, color, marker) in MODELS.items():
        values = [
            aggregate(result, model, severity, "counterfactual_pair_accuracy")
            for severity in SEVERITIES
        ]
        means = np.asarray([value["mean"] for value in values])
        axes[0, 1].plot(
            severity_x,
            means,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            label=label,
        )
    axes[0, 1].set_title("B  Counterfactual order robustness", loc="left", fontweight="bold")
    axes[0, 1].set_xticks(severity_x, [value.title() for value in SEVERITIES])
    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 1].set_ylabel("Pair accuracy")

    comparisons = {
        "real_operator": ("Complex − real", "#D05A2D", "o"),
        "two_channel_operator": ("Complex − two-channel", "#C28B16", "D"),
        "transformer": ("Complex − Transformer", "#2474B5", "o"),
    }
    for baseline, (label, color, marker) in comparisons.items():
        values = [
            result["paired_world_effects"][severity][baseline]["top1"] for severity in SEVERITIES
        ]
        means = np.asarray([value["mean"] for value in values])
        errors = np.asarray(
            [
                [mean - value["ci95_low"] for mean, value in zip(means, values, strict=True)],
                [value["ci95_high"] - mean for mean, value in zip(means, values, strict=True)],
            ]
        )
        axes[1, 0].errorbar(
            severity_x,
            means,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.8,
            capsize=3,
            label=label,
        )
    axes[1, 0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 0].set_title(
        "C  Complex top-1 paired effect across worlds", loc="left", fontweight="bold"
    )
    axes[1, 0].set_xticks(severity_x, [value.title() for value in SEVERITIES])
    axes[1, 0].set_ylabel("Top-1 difference")
    axes[1, 0].legend(frameon=False, loc="upper right")

    bar_x = np.arange(len(MODELS))
    raw_ece = [aggregate(result, model, "moderate", "ece")["mean"] for model in MODELS]
    calibrated_ece = [
        aggregate(result, model, "moderate", "calibrated_ece")["mean"] for model in MODELS
    ]
    width = 0.36
    axes[1, 1].bar(bar_x - width / 2, raw_ece, width, label="Raw", color="#5B8DB8")
    axes[1, 1].bar(
        bar_x + width / 2,
        calibrated_ece,
        width,
        label="ID temperature-scaled",
        color="#B86B5B",
    )
    axes[1, 1].set_title(
        "D  In-domain calibration does not transfer", loc="left", fontweight="bold"
    )
    axes[1, 1].set_xticks(bar_x, ["MLP", "Tr.", "GRU", "Real", "2-ch.", "Complex"], rotation=20)
    axes[1, 1].set_ylabel("Moderate-shift ECE")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle(
        "Q-Neuro robustness confirmation across five unseen worlds",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Shift summaries and paired effects use unseen world seed as the statistical unit "
            f"(n=5). Source: {result_directory.name}. Synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"robustness_world_sweep.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
