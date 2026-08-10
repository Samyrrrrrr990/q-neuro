"""Plot learning-law accuracy, compute, calibration, and gradient diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    "adamw": "AdamW",
    "sgd": "SGD",
    "gradient_accumulation": "Gradient accumulation",
    "multiobjective_adamw": "Multi-objective AdamW",
    "pcgrad": "PCGrad",
    "phase_gradient": "PGO",
    "local_plasticity": "Local plasticity",
    "hybrid_local_global": "Local→global hybrid",
    "zerobackprop": "ZeroBackprop",
}
COLORS = {
    "adamw": "#315A7D",
    "sgd": "#8A8A8A",
    "gradient_accumulation": "#6B91B2",
    "multiobjective_adamw": "#16847A",
    "pcgrad": "#D98B2B",
    "phase_gradient": "#673C9E",
    "local_plasticity": "#D05A2D",
    "hybrid_local_global": "#E07A5F",
    "zerobackprop": "#444444",
}


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") == "training_law_suite" and "SMOKE PROFILE" not in config.get(
            "description", ""
        ):
            candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed full training-law suite found")
    return max(candidates)[1]


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
    methods = list(METHODS)
    sizes = sorted(int(value) for value in result["summary"]["adamw"])
    largest = str(max(sizes))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    y = np.arange(len(methods))
    width = 0.34
    in_domain = [
        result["summary"][method][largest]["in_domain"]["top1"]["mean"]
        for method in methods
    ]
    shifted = [
        result["summary"][method][largest]["shifted"]["across_worlds"]["top1"]["mean"]
        for method in methods
    ]
    axes[0, 0].barh(y + width / 2, in_domain, width, color="#9CB9CF", label="In-domain")
    axes[0, 0].barh(y - width / 2, shifted, width, color="#315A7D", label="Moderate shift")
    axes[0, 0].set_yticks(y, [METHODS[method] for method in methods])
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(0, 1.03)
    axes[0, 0].set_xlabel("Top-1 accuracy")
    axes[0, 0].set_title("A  Backprop-free laws do learn—but do not transfer", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, loc="lower right")

    compute_offsets = {
        "adamw": (4, -12),
        "gradient_accumulation": (4, -12),
        "multiobjective_adamw": (4, 5),
        "pcgrad": (4, 6),
        "phase_gradient": (-16, -12),
    }
    for method in methods:
        seconds = result["summary"][method][largest]["in_domain"]["training_seconds"]["mean"]
        top1 = result["summary"][method][largest]["shifted"]["across_worlds"]["top1"]["mean"]
        axes[0, 1].scatter(seconds, top1, color=COLORS[method], s=42)
        offset = compute_offsets.get(method, (4, 2))
        axes[0, 1].annotate(
            METHODS[method],
            (seconds, top1),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.8,
        )
    axes[0, 1].set_xlabel("Mean CPU training time (s)")
    axes[0, 1].set_ylabel("Moderate-shift top-1")
    axes[0, 1].set_title("B  PGO adds compute without a frontier gain", loc="left", fontweight="bold")

    ambiguity = [
        result["summary"][method][largest]["in_domain"]["ambiguity_pair_nll"]["mean"]
        for method in methods
    ]
    shift_nll = [
        result["summary"][method][largest]["shifted"]["across_worlds"]["nll"]["mean"]
        for method in methods
    ]
    axes[1, 0].scatter(ambiguity, shift_nll, c=[COLORS[method] for method in methods], s=42)
    calibration_labels = {
        "adamw": (8, 8),
        "multiobjective_adamw": (-22, -18),
        "phase_gradient": (-52, -5),
        "hybrid_local_global": (4, 2),
        "local_plasticity": (4, 2),
        "sgd": (4, 2),
        "zerobackprop": (4, 2),
    }
    for method, first, second in zip(methods, ambiguity, shift_nll, strict=True):
        if method in calibration_labels:
            axes[1, 0].annotate(
                METHODS[method],
                (first, second),
                xytext=calibration_labels[method],
                textcoords="offset points",
                fontsize=6.8,
            )
    axes[1, 0].set_xlabel("Ambiguous-pair NLL (lower is better)")
    axes[1, 0].set_ylabel("Shifted NLL (lower is better)")
    axes[1, 0].set_title("C  Local pretraining changes the error tradeoff", loc="left", fontweight="bold")

    selected = ("adamw", "multiobjective_adamw", "pcgrad", "phase_gradient")
    size_colors = ("#B9C9D6", "#315A7D")
    for size_index, size in enumerate(sizes):
        values = [
            result["summary"][method][str(size)]["shifted"]["across_worlds"]["top1"]["mean"]
            for method in selected
        ]
        axes[1, 1].bar(
            np.arange(len(selected)) + (size_index - 0.5) * 0.34,
            values,
            0.34,
            color=size_colors[size_index],
            label=f"n={size}",
        )
    axes[1, 1].set_xticks(np.arange(len(selected)), [METHODS[value] for value in selected], rotation=12)
    axes[1, 1].set_ylim(0.25, 0.68)
    axes[1, 1].set_ylabel("Moderate-shift top-1")
    axes[1, 1].set_title("D  Auxiliary supervision helps; gradient law does not", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(True, color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle("Experiment Six: unconventional training laws", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        (
            f"One fixed complex operator architecture; means over three seeds/worlds. Source: {result_directory.name}. "
            "Times are descriptive CPU measurements; synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"training_law_suite.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
