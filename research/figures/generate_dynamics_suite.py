"""Plot predictive, ambiguity, and efficiency tradeoffs across computational laws."""

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
    "logistic": ("Logistic", "LR", "#777777", "o"),
    "mlp": ("MLP", "MLP", "#999999", "o"),
    "complex_mlp": ("Complex MLP", "C-MLP", "#8E6AAE", "o"),
    "transformer": ("Transformer", "Tr.", "#2474B5", "o"),
    "gru": ("GRU", "GRU", "#16847A", "s"),
    "state_space": ("State space", "SSM", "#3E9A8D", "s"),
    "hopfield": ("Hopfield", "Hop.", "#536D88", "D"),
    "graph_network": ("Graph", "GNN", "#6A8098", "D"),
    "coupled_tensor": ("Coupled tensor", "Tensor", "#B07552", "D"),
    "real_operator": ("Real operator", "Real", "#D05A2D", "o"),
    "two_channel_operator": ("Two-channel", "2-ch.", "#C28B16", "D"),
    "complex_operator": ("Complex operator", "Complex", "#673C9E", "o"),
    "energy_attractor": ("Energy attractor", "Energy", "#C44E52", "^"),
    "adaptive_attractor": ("Adaptive attractor", "Adaptive", "#E07A5F", "^"),
    "hamiltonian": ("Hamiltonian", "Ham.", "#7A5195", "P"),
    "dissipative": ("Dissipative", "Diss.", "#EF5675", "P"),
    "hybrid_dynamics": ("Hybrid", "Hybrid", "#BC5090", "P"),
    "density_dynamics": ("Density D3", "D3", "#003F5C", "X"),
}
DYNAMICS = (
    "real_operator",
    "complex_operator",
    "energy_attractor",
    "adaptive_attractor",
    "hamiltonian",
    "dissipative",
    "hybrid_dynamics",
    "density_dynamics",
)


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "dynamics_mechanism_suite":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed dynamics suite found")
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

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.7), constrained_layout=True)

    for model, (_, short, color, marker) in MODELS.items():
        in_domain = metric(result, model, "in_domain", "top1")["mean"]
        shifted = metric(result, model, "moderate", "top1")["mean"]
        axes[0, 0].scatter(in_domain, shifted, color=color, marker=marker, s=36)
        axes[0, 0].annotate(
            short, (in_domain, shifted), xytext=(3, 2), textcoords="offset points", fontsize=6.7
        )
    axes[0, 0].plot([0.25, 1.0], [0.25, 1.0], color="#BBBBBB", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title(
        "A  In-domain accuracy does not predict robustness", loc="left", fontweight="bold"
    )
    axes[0, 0].set_xlabel("In-domain top-1")
    axes[0, 0].set_ylabel("Moderate-shift top-1")
    axes[0, 0].set_xlim(0.25, 1.01)
    axes[0, 0].set_ylim(0.15, 0.70)

    for model, (_, short, color, marker) in MODELS.items():
        ambiguity = metric(result, model, "in_domain", "ambiguity_pair_nll")["mean"]
        shifted = metric(result, model, "moderate", "top1")["mean"]
        axes[0, 1].scatter(ambiguity, shifted, color=color, marker=marker, s=36)
        axes[0, 1].annotate(
            short, (ambiguity, shifted), xytext=(3, 2), textcoords="offset points", fontsize=6.7
        )
    axes[0, 1].axvline(np.log(2.0), color="#999999", linestyle="--", linewidth=0.8)
    axes[0, 1].set_title("B  Robustness–ambiguity tradeoff", loc="left", fontweight="bold")
    axes[0, 1].set_xlabel("Ambiguous-pair NLL (lower is better)")
    axes[0, 1].set_ylabel("Moderate-shift top-1")

    x = np.arange(len(DYNAMICS))
    shift_values = [metric(result, model, "moderate", "top1") for model in DYNAMICS]
    axes[1, 0].bar(
        x,
        [value["mean"] for value in shift_values],
        yerr=errors(shift_values),
        capsize=3,
        color=[MODELS[model][2] for model in DYNAMICS],
    )
    axes[1, 0].set_xticks(x, [MODELS[model][1] for model in DYNAMICS], rotation=25, ha="right")
    axes[1, 0].set_ylabel("Moderate-shift top-1")
    axes[1, 0].set_ylim(0.35, 0.70)
    axes[1, 0].set_title("C  Computational-law ablation", loc="left", fontweight="bold")

    for model, (_, short, color, marker) in MODELS.items():
        runtime = metric(result, model, "in_domain", "training_seconds")["mean"]
        shifted = metric(result, model, "moderate", "top1")["mean"]
        axes[1, 1].scatter(runtime, shifted, color=color, marker=marker, s=36)
        axes[1, 1].annotate(
            short, (runtime, shifted), xytext=(3, 2), textcoords="offset points", fontsize=6.7
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].minorticks_off()
    axes[1, 1].set_title("D  Training-time robustness frontier", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Training time per seed (s, log scale)")
    axes[1, 1].set_ylabel("Moderate-shift top-1")

    for axis in axes.flat:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle("Computational-law mechanism suite", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        (
            "In-domain summaries use three training seeds; shifted summaries use three unseen world "
            f"means after seed averaging. Source: {result_directory.name}; synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"dynamics_suite.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
