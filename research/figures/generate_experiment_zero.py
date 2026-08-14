"""Generate publication-quality Experiment Zero learning-curve figures from run artifacts."""

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
COLORS = {
    "mlp": "#767676",
    "transformer": "#2474B5",
    "real_operator": "#D05A2D",
    "complex_operator": "#673C9E",
}
LABELS = {
    "mlp": "MLP (unordered)",
    "transformer": "Tiny Transformer",
    "real_operator": "Real operator",
    "complex_operator": "Complex operator",
}


def latest_result(experiment_name: str) -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != experiment_name:
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        run_id = int(config_path.parent.name.split("-")[1])
        if (config_path.parent / "VALIDITY.md").exists():
            continue
        candidates.append((run_id, config_path.parent))
    if not candidates:
        raise FileNotFoundError(f"no valid result found for {experiment_name}")
    return max(candidates)[1]


def load_metrics(result_directory: Path) -> dict[str, Any]:
    return json.loads((result_directory / "metrics.json").read_text(encoding="utf-8"))


def metric_curve(
    sample_result: dict[str, Any],
    full_result: dict[str, Any],
    model: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sizes = sorted(int(size) for size in sample_result["summary"][model])
    means = [sample_result["summary"][model][str(size)][metric]["mean"] for size in sizes]
    lows = [sample_result["summary"][model][str(size)][metric]["ci95_low"] for size in sizes]
    highs = [sample_result["summary"][model][str(size)][metric]["ci95_high"] for size in sizes]
    full_size = 14_000
    full_metric = full_result["summary"][model][metric]
    sizes.append(full_size)
    means.append(full_metric["mean"])
    lows.append(full_metric["ci95_low"])
    highs.append(full_metric["ci95_high"])
    return tuple(np.asarray(values, dtype=float) for values in (sizes, means, lows, highs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "research" / "figures" / "generated",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)

    sample_dir = latest_result("experiment_zero_sample_efficiency")
    full_dir = latest_result("experiment_zero")
    sample_result = load_metrics(sample_dir)
    full_result = load_metrics(full_dir)

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
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    panels = (
        ("top1", "Top-1 accuracy", (0.40, 1.02), False),
        ("counterfactual_pair_accuracy", "Counterfactual pair accuracy", (-0.02, 1.02), False),
        ("nll", "Negative log-likelihood", (0.001, 2.0), True),
        ("ece", "Expected calibration error", (0.001, 0.8), True),
    )
    for panel_index, (axis, panel) in enumerate(zip(axes.flat, panels, strict=True)):
        metric, title, limits, log_y = panel
        for model, label in LABELS.items():
            sizes, means, lows, highs = metric_curve(sample_result, full_result, model, metric)
            color = COLORS[model]
            axis.plot(
                sizes,
                means,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=label,
            )
            axis.fill_between(
                sizes,
                np.clip(lows, 0.0, None),
                np.clip(highs, 0.0, None),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.set_xscale("log")
        if log_y:
            axis.set_yscale("log")
        axis.set_ylim(*limits)
        axis.set_title(f"{chr(65 + panel_index)}  {title}", loc="left", fontweight="bold")
        axis.set_xlabel("Training cases (log scale)")
        axis.grid(True, which="major", color="#DADADA", linewidth=0.6, alpha=0.8)
    axes[0, 0].legend(frameon=False, loc="lower right")
    figure.suptitle(
        "Experiment Zero: ordered models learn the synthetic chronology task efficiently",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        (
            f"Mean over 3 seeds; shading is 95% Student-t CI. Sources: "
            f"{sample_dir.name} and {full_dir.name}. Synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"experiment_zero_learning_curves.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
