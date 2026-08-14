"""Generate task-suite figures directly from the latest complete QN artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "mlp": ("MLP", "#777777"),
    "transformer": ("Transformer", "#2474B5"),
    "gru": ("GRU", "#16847A"),
    "real_operator": ("Real operator", "#D05A2D"),
    "two_channel_operator": ("Two-channel", "#C28B16"),
    "complex_operator": ("Complex", "#673C9E"),
}


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "neuro_task_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no complete NeuroWorld task suite found")
    return max(candidates)[1]


def metric(result: dict[str, Any], model: str, context: str, name: str) -> dict[str, float]:
    return result["summary"][model][context][name]


def errorbars(values: list[dict[str, float]]) -> np.ndarray:
    means = np.asarray([value["mean"] for value in values])
    return np.asarray(
        [
            [mean - value["ci95_low"] for mean, value in zip(means, values, strict=True)],
            [value["ci95_high"] - mean for mean, value in zip(means, values, strict=True)],
        ]
    )


def grouped_bars(
    axis: plt.Axes,
    result: dict[str, Any],
    context: str,
    metric_names: tuple[str, ...],
    labels: tuple[str, ...],
    ylabel: str,
) -> None:
    x = np.arange(len(MODELS))
    width = 0.8 / len(metric_names)
    offsets = (np.arange(len(metric_names)) - (len(metric_names) - 1) / 2) * width
    shades = (0.72, 1.0)
    for index, (metric_name, label) in enumerate(zip(metric_names, labels, strict=True)):
        values = [metric(result, model, context, metric_name) for model in MODELS]
        means = np.asarray([value["mean"] for value in values])
        colors = [MODELS[model][1] for model in MODELS]
        axis.bar(
            x + offsets[index],
            means,
            width,
            yerr=errorbars(values),
            capsize=2,
            color=colors,
            alpha=shades[index],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
    axis.set_xticks(x, [value[0] for value in MODELS.values()], rotation=22, ha="right")
    axis.set_ylabel(ylabel)
    axis.legend(frameon=False)


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
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)

    grouped_bars(
        axes[0, 0],
        result,
        "composition",
        ("reference_top1", "composition_top1"),
        ("Reference combinations", "Held-out combinations"),
        "Top-1 accuracy",
    )
    axes[0, 0].set_title("A  Composition test reaches a ceiling", loc="left", fontweight="bold")
    axes[0, 0].set_ylim(0.68, 1.02)

    x = np.arange(len(MODELS))
    ambiguity_values = [metric(result, model, "base", "ambiguity_pair_nll") for model in MODELS]
    axes[0, 1].bar(
        x,
        [value["mean"] for value in ambiguity_values],
        yerr=errorbars(ambiguity_values),
        capsize=3,
        color=[value[1] for value in MODELS.values()],
    )
    axes[0, 1].axhline(math.log(2.0), color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 1].text(5.48, math.log(2.0) + 0.12, "ideal twin-only NLL", ha="right", fontsize=7.5)
    axes[0, 1].set_xticks(x, [value[0] for value in MODELS.values()], rotation=22, ha="right")
    axes[0, 1].set_ylabel("Ambiguous-pair NLL (lower is better)")
    axes[0, 1].set_title(
        "B  Complex dynamics are poorly calibrated for ambiguity", loc="left", fontweight="bold"
    )

    grouped_bars(
        axes[1, 0],
        result,
        "unknown_disease",
        ("ood_auroc_msp", "representation_ood_auroc"),
        ("Maximum-softmax uncertainty", "Centroid distance"),
        "Unknown-disease AUROC",
    )
    axes[1, 0].set_title("C  Unknown disease is detectable", loc="left", fontweight="bold")
    axes[1, 0].set_ylim(0.45, 1.03)

    grouped_bars(
        axes[1, 1],
        result,
        "base",
        ("hidden_ood_auroc_msp", "hidden_representation_ood_auroc"),
        ("Maximum-softmax uncertainty", "Centroid distance"),
        "Hidden-syndrome AUROC",
    )
    axes[1, 1].set_title(
        "D  Hidden-syndrome geometry is model-dependent", loc="left", fontweight="bold"
    )
    axes[1, 1].set_ylim(0.40, 1.03)

    for axis in axes.flat:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle(
        "Orthogonal NeuroWorld task suite",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Bars show means and Student-t 95% intervals across three training seeds. "
            f"Source: {result_directory.name}. Synthetic data only; exploratory inference."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"neuro_task_suite.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
