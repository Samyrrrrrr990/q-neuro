"""Generate active-evidence curves and compute-efficiency plots from experiment artifacts."""

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
    "gru": ("GRU", "#16847A", "s"),
    "real_operator": ("Real operator", "#D05A2D", "o"),
    "two_channel_operator": ("Two-channel", "#C28B16", "D"),
    "complex_operator": ("Complex", "#673C9E", "o"),
}
STRATEGIES = {
    "random": ("Random", 0.55),
    "fixed_information": ("Fixed global information", 0.78),
    "expected_information_gain": ("Expected information gain", 1.0),
}


def latest_result() -> Path:
    candidates: list[tuple[int, Path]] = []
    for config_path in (ROOT / "experiments" / "results").glob("QN-*/config.yaml"):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config.get("experiment") != "active_evidence_acquisition":
            continue
        if "SMOKE PROFILE" in config.get("description", ""):
            continue
        candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed active-evidence result found")
    return max(candidates)[1]


def summary_metric(
    result: dict[str, Any], model: str, strategy: str, metric: str
) -> dict[str, float]:
    return result["summary"][model]["policies"][strategy][metric]


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
    max_queries = len(result["runs"][0]["policies"]["random"]["curve"])
    query_axis = np.arange(1, max_queries + 1)

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
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)

    for model, (label, color, marker) in MODELS.items():
        values = [
            summary_metric(
                result, model, "expected_information_gain", f"accuracy_at_{query}_queries"
            )
            for query in query_axis
        ]
        means = np.asarray([value["mean"] for value in values])
        low = np.asarray([value["ci95_low"] for value in values])
        high = np.asarray([value["ci95_high"] for value in values])
        axes[0, 0].plot(
            query_axis,
            means,
            color=color,
            marker=marker,
            markevery=2,
            markersize=4,
            linewidth=1.8,
            label=label,
        )
        axes[0, 0].fill_between(query_axis, low, high, color=color, alpha=0.09)
    axes[0, 0].set_title(
        "A  Accuracy under model-conditioned querying", loc="left", fontweight="bold"
    )
    axes[0, 0].set_xlabel("Findings revealed (of 40)")
    axes[0, 0].set_ylabel("Top-1 accuracy")
    axes[0, 0].set_ylim(0.0, 0.95)
    axes[0, 0].legend(frameon=False, ncols=2, loc="upper left")

    x = np.arange(len(MODELS))
    width = 0.24
    offsets = (-width, 0.0, width)
    for offset, (strategy, (label, alpha)) in zip(offsets, STRATEGIES.items(), strict=True):
        values = [summary_metric(result, model, strategy, "accuracy_auc") for model in MODELS]
        axes[0, 1].bar(
            x + offset,
            [value["mean"] for value in values],
            width,
            yerr=errors(values),
            capsize=2,
            color=[MODELS[model][1] for model in MODELS],
            alpha=alpha,
            label=label,
        )
    axes[0, 1].set_title("B  Evidence-efficiency depends on policy", loc="left", fontweight="bold")
    axes[0, 1].set_xticks(x, [value[0] for value in MODELS.values()], rotation=22, ha="right")
    axes[0, 1].set_ylabel("Mean accuracy over queries 1–12")
    axes[0, 1].set_ylim(0.1, 0.68)
    axes[0, 1].legend(frameon=False, fontsize=7.3)

    for model, (label, color, marker) in MODELS.items():
        fixed_values = [
            summary_metric(result, model, strategy, "final_accuracy")["mean"]
            for strategy in STRATEGIES
        ]
        axes[1, 0].plot(
            np.arange(len(STRATEGIES)),
            fixed_values,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.6,
            label=label,
        )
    axes[1, 0].set_title("C  Accuracy after 12 revealed findings", loc="left", fontweight="bold")
    axes[1, 0].set_xticks(
        np.arange(len(STRATEGIES)),
        ["Random", "Fixed info", "Expected info"],
    )
    axes[1, 0].set_ylabel("Top-1 accuracy")
    axes[1, 0].set_ylim(0.25, 0.90)

    for model, (label, color, marker) in MODELS.items():
        for strategy, (_, alpha) in STRATEGIES.items():
            runtime = summary_metric(result, model, strategy, "policy_seconds")["mean"]
            auc = summary_metric(result, model, strategy, "accuracy_auc")["mean"]
            axes[1, 1].scatter(
                runtime,
                auc,
                color=color,
                marker=marker,
                alpha=alpha,
                s=38,
                edgecolor="white",
                linewidth=0.4,
            )
        eig_runtime = summary_metric(result, model, "expected_information_gain", "policy_seconds")[
            "mean"
        ]
        eig_auc = summary_metric(result, model, "expected_information_gain", "accuracy_auc")["mean"]
        axes[1, 1].annotate(
            label, (eig_runtime, eig_auc), xytext=(4, 2), textcoords="offset points", fontsize=7
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].minorticks_off()
    axes[1, 1].set_title("D  Query-policy compute frontier", loc="left", fontweight="bold")
    axes[1, 1].set_xlabel("Policy evaluation time for 200 cases (s, log scale)")
    axes[1, 1].set_ylabel("Mean accuracy over queries 1–12")

    for axis in axes.flat:
        axis.grid(True, axis="y", color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle(
        "Active evidence acquisition in factorial NeuroWorld", fontsize=13, fontweight="bold"
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Means and Student-t 95% intervals use three training seeds. Policies reveal binary "
            f"finding outcomes only. Source: {result_directory.name}; synthetic data only."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"active_evidence.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
