"""Generate the Q-Neuro generator-shift replication figure from registered artifacts."""

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


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "experiment_zero_generator_shift":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        run_id = int(config_path.parent.name.split("-")[1])
        candidates.append((run_id, config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed generator-shift result found")
    return max(candidates)[1]


def curve(
    result: dict[str, Any], model: str, environment: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size_map = result["summary"][model]
    sizes = sorted(int(size) for size in size_map)
    aggregates = [size_map[str(size)][environment][metric] for size in sizes]
    return (
        np.asarray(sizes),
        np.asarray([value["mean"] for value in aggregates]),
        np.asarray([value["ci95_low"] for value in aggregates]),
        np.asarray([value["ci95_high"] for value in aggregates]),
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
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.0), constrained_layout=True)
    panels = (
        ("in_domain", "top1", "A  In-domain top-1 accuracy"),
        ("nuisance_seed_shift", "top1", "B  Nuisance-seed shift top-1"),
        ("noisy_sparse_shift", "top1", "C  Noisy/sparse shift top-1"),
        (
            "noisy_sparse_shift",
            "counterfactual_pair_accuracy",
            "D  Shifted counterfactual pair accuracy",
        ),
    )
    for axis, (environment, metric, title) in zip(axes.flat, panels, strict=True):
        for model, (label, color, marker) in MODELS.items():
            sizes, means, lows, highs = curve(result, model, environment, metric)
            axis.plot(
                sizes,
                means,
                label=label,
                color=color,
                marker=marker,
                markersize=4.5,
                linewidth=1.8,
            )
            axis.fill_between(
                sizes,
                np.clip(lows, 0.0, 1.0),
                np.clip(highs, 0.0, 1.0),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        axis.set_xscale("log")
        axis.set_xlim(220, 1150)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks([250, 500, 1000], labels=["250", "500", "1,000"])
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Training cases")
        axis.grid(True, which="major", color="#DADADA", linewidth=0.6, alpha=0.8)
    axes[0, 0].legend(frameon=False, ncols=2, loc="lower right")
    figure.suptitle(
        "Generator shift separates in-domain fit from robustness",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        (
            f"Mean over 3 seeds; shading is 95% Student-t CI. Source: "
            f"{result_directory.name}. Synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"generator_shift_replication.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
