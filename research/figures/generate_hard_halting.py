"""Plot validation calibration, accuracy, latency, and calibration for hard halting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[2]


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") == "hard_velocity_halting" and "SMOKE PROFILE" not in config.get(
            "description", ""
        ):
            candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed full hard-halting experiment found")
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
    figure, axes = plt.subplots(2, 2, figsize=(10.7, 7.6), constrained_layout=True)
    for run in result["runs"]:
        curve = sorted(run["calibration_curve"], key=lambda value: value["mean_executed_steps"])
        steps = [value["mean_executed_steps"] for value in curve]
        top1 = [value["top1"] for value in curve]
        axes[0, 0].plot(steps, top1, marker="o", markersize=3, alpha=0.75, label=f"seed {run['seed']}")
        selected = min(
            curve,
            key=lambda value: abs(value["threshold"] - run["selected_velocity_threshold"]),
        )
        axes[0, 0].scatter(
            selected["mean_executed_steps"],
            selected["top1"],
            marker="*",
            s=90,
            color="#D05A2D",
            zorder=4,
        )
    axes[0, 0].set_xlabel("Mean validation states executed")
    axes[0, 0].set_ylabel("Validation top-1")
    axes[0, 0].set_title("A  Validation selects the two-state boundary", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    modes = ("soft", "fixed_final", "hard")
    labels = ("Soft ACT mix", "Fixed final (8)", "Hard halt (2)")
    x = np.arange(len(modes))
    width = 0.34
    in_domain = [result["summary"][mode]["in_domain"]["top1"]["mean"] for mode in modes]
    shifted = [
        result["summary"][mode]["shifted"]["across_worlds"]["top1"]["mean"] for mode in modes
    ]
    axes[0, 1].bar(x - width / 2, in_domain, width, color="#9CB9CF", label="In-domain")
    axes[0, 1].bar(x + width / 2, shifted, width, color="#315A7D", label="Moderate shift")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0.35, 0.77)
    axes[0, 1].set_ylabel("Top-1 accuracy")
    axes[0, 1].set_title("B  Early truncation preserves accuracy", loc="left", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    latency = [
        result["summary"][mode]["in_domain"]["latency_ms_per_case"]["mean"] for mode in modes
    ]
    states = [8.0, 8.0, result["summary"]["hard"]["in_domain"]["mean_executed_steps"]["mean"]]
    axes[1, 0].bar(x - width / 2, latency, width, color="#673C9E", label="Latency (ms/case)")
    second_axis = axes[1, 0].twinx()
    second_axis.bar(x + width / 2, states, width, color="#D98B2B", label="States executed")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylabel("CPU latency (ms/case)", color="#673C9E")
    second_axis.set_ylabel("Diagnostic states", color="#D98B2B")
    second_axis.set_ylim(0, 9)
    axes[1, 0].set_title("C  Fewer updates become lower wall time", loc="left", fontweight="bold")
    handles_a, labels_a = axes[1, 0].get_legend_handles_labels()
    handles_b, labels_b = second_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    nll = [result["summary"][mode]["shifted"]["across_worlds"]["nll"]["mean"] for mode in modes]
    ece = [result["summary"][mode]["shifted"]["across_worlds"]["ece"]["mean"] for mode in modes]
    axes[1, 1].bar(x - width / 2, nll, width, color="#3E9A8D", label="NLL")
    axes[1, 1].bar(x + width / 2, ece, width, color="#E07A5F", label="ECE")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("Shifted calibration error")
    axes[1, 1].set_title("D  Later attraction amplifies overconfidence", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle("Realized hard halting of attractor inference", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        (
            f"Thresholds selected on source validation only; three checkpoint seeds. Source: {result_directory.name}. "
            "All cases halt at step 2, so the result is truncation—not case adaptivity."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"hard_halting.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
