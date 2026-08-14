"""Plot frozen hierarchical-probe accuracy and Hermitian observable comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[2]
PROPERTIES = ("mechanism", "localization", "temporality", "context")
MODELS = {
    "logistic": "Logistic",
    "mlp": "MLP",
    "complex_mlp": "Complex MLP",
    "transformer": "Transformer",
    "gru": "GRU",
    "state_space": "State space",
    "hopfield": "Hopfield",
    "graph_network": "Graph",
    "coupled_tensor": "Coupled tensor",
    "real_operator": "Real operator",
    "two_channel_operator": "Two-channel",
    "complex_operator": "Complex operator",
    "energy_attractor": "Energy attractor",
    "adaptive_attractor": "Adaptive attractor",
    "hamiltonian": "Hamiltonian",
    "dissipative": "Dissipative",
    "hybrid_dynamics": "Hybrid",
    "density_dynamics": "Density D3",
}


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "hierarchical_observable_probe":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed observable probe experiment found")
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
    source = json.loads(
        (ROOT / "experiments" / "results" / result["source_experiment"] / "metrics.json").read_text(
            encoding="utf-8"
        )
    )

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
    figure, axes = plt.subplots(2, 2, figsize=(10.7, 8.0), constrained_layout=True)
    matrix = np.asarray(
        [
            [
                result["summary"][model][property_name]["linear"]["accuracy"]["mean"]
                for property_name in PROPERTIES
            ]
            for model in MODELS
        ]
    )
    image = axes[0, 0].imshow(matrix, vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
    axes[0, 0].set_xticks(np.arange(4), [value.title() for value in PROPERTIES])
    axes[0, 0].set_yticks(np.arange(len(MODELS)), list(MODELS.values()))
    axes[0, 0].set_title(
        "A  Linear extractability from frozen states", loc="left", fontweight="bold"
    )
    figure.colorbar(image, ax=axes[0, 0], label="Probe accuracy", fraction=0.046)

    selected = ("gru", "state_space", "real_operator", "complex_operator", "adaptive_attractor")
    x = np.arange(len(PROPERTIES))
    width = 0.16
    colors = ("#16847A", "#3E9A8D", "#D05A2D", "#673C9E", "#E07A5F")
    for index, (model, color) in enumerate(zip(selected, colors, strict=True)):
        values = [
            result["summary"][model][property_name]["linear"]["accuracy"]["mean"]
            for property_name in PROPERTIES
        ]
        axes[0, 1].bar(
            x + (index - 2) * width,
            values,
            width,
            color=color,
            label=MODELS[model],
        )
    axes[0, 1].set_xticks(x, [value.title() for value in PROPERTIES])
    axes[0, 1].set_ylim(0.75, 1.01)
    axes[0, 1].set_ylabel("Linear probe accuracy")
    axes[0, 1].set_title(
        "B  Strong hierarchy is not unique to complex state", loc="left", fontweight="bold"
    )
    axes[0, 1].legend(frameon=False, ncols=2)

    hermitian_models = ("complex_operator", "hamiltonian", "hybrid_dynamics")
    markers = ("o", "P", "X")
    for model, marker in zip(hermitian_models, markers, strict=True):
        linear = [
            result["summary"][model][property_name]["linear"]["accuracy"]["mean"]
            for property_name in PROPERTIES
        ]
        hermitian = [
            result["summary"][model][property_name]["hermitian"]["accuracy"]["mean"]
            for property_name in PROPERTIES
        ]
        axes[1, 0].scatter(linear, hermitian, marker=marker, s=42, label=MODELS[model])
        for property_name, first, second in zip(PROPERTIES, linear, hermitian, strict=True):
            axes[1, 0].annotate(
                property_name[0].upper(),
                (first, second),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=7,
            )
    axes[1, 0].plot([0.78, 0.97], [0.78, 0.97], color="#777777", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xlabel("Linear probe accuracy")
    axes[1, 0].set_ylabel("Hermitian observable accuracy")
    axes[1, 0].set_title(
        "C  Quadratic observables can expose structure", loc="left", fontweight="bold"
    )
    axes[1, 0].legend(frameon=False)

    labeled_models = {
        "complex_operator",
        "density_dynamics",
        "graph_network",
        "gru",
        "hamiltonian",
        "hybrid_dynamics",
        "real_operator",
        "state_space",
        "two_channel_operator",
    }
    for model, label in MODELS.items():
        probe_mean = float(matrix[list(MODELS).index(model)].mean())
        shift = source["summary"][model]["moderate"]["across_worlds"]["top1"]["mean"]
        axes[1, 1].scatter(
            probe_mean, shift, color="#673C9E" if model == "complex_operator" else "#777777", s=32
        )
        if model in labeled_models:
            axes[1, 1].annotate(
                label,
                (probe_mean, shift),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=6.5,
            )
    correlation = np.corrcoef(
        matrix.mean(axis=1),
        [source["summary"][model]["moderate"]["across_worlds"]["top1"]["mean"] for model in MODELS],
    )[0, 1]
    axes[1, 1].set_xlabel("Mean hierarchical linear-probe accuracy")
    axes[1, 1].set_ylabel("Moderate-shift top-1")
    axes[1, 1].set_title(
        f"D  Hierarchy and robustness are distinct (r={correlation:+.2f})",
        loc="left",
        fontweight="bold",
    )

    for axis in axes.flat[1:]:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle("Emergent hierarchical observables", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        (
            "Probes use frozen QN-000014 representations and training-only validation selection. "
            f"Means span three model seeds. Source: {result_directory.name}; synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"observable_probe.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
