"""Render the signature Q-Neuro evidence-state and counterfactual trajectory figure."""

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
        if (
            config.get("experiment") == "complex_state_trajectory_study"
            and (config_path.parent / "metrics.json").exists()
        ):
            candidates.append((int(config_path.parent.name.split("-")[1]), config_path.parent))
    if not candidates:
        raise FileNotFoundError("no completed trajectory study found")
    return max(candidates)[1]


def short_token(label: str) -> str:
    sign = label[0]
    stem = label[1:]
    prefixes = {
        "order_marker_": "O",
        "mechanism_signal_": "M",
        "localization_signal_": "L",
        "temporal_signal_": "T",
        "context_signal_": "C",
    }
    for prefix, abbreviation in prefixes.items():
        if stem.startswith(prefix):
            return f"{sign}{abbreviation}{stem.removeprefix(prefix)}"
    return label


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
    artifact = json.loads(
        (result_directory / "selected_trajectories.json").read_text(encoding="utf-8")
    )
    case = artifact["case"]
    pair = artifact["counterfactual_pair"]
    probabilities = np.asarray(case["probabilities"])
    amplitude = np.asarray(case["amplitude_real"]) + 1j * np.asarray(case["amplitude_imag"])
    token_labels = ["start", *[short_token(value) for value in case["token_labels"]]]
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
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 8.1), constrained_layout=True)
    image = axes[0, 0].imshow(probabilities.T, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    axes[0, 0].set_yticks(np.arange(20), [f"D{value:02d}" for value in range(20)])
    tick_positions = np.arange(0, len(token_labels), 2)
    axes[0, 0].set_xticks(
        tick_positions, [token_labels[value] for value in tick_positions], rotation=70
    )
    axes[0, 0].set_xlabel("Observed evidence step (+ present, − observed absent)")
    axes[0, 0].set_ylabel("Diagnostic hypothesis")
    axes[0, 0].set_title(
        "A  Hypothesis intensity evolves with evidence", loc="left", fontweight="bold"
    )
    figure.colorbar(image, ax=axes[0, 0], label="Measured probability", fraction=0.046)

    top_diagnoses = np.argsort(probabilities[-1])[-5:][::-1]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(top_diagnoses)))
    for diagnosis, color in zip(top_diagnoses, colors, strict=True):
        path = amplitude[:, diagnosis]
        axes[0, 1].plot(path.real, path.imag, color=color, alpha=0.85, linewidth=1.2)
        axes[0, 1].scatter(path.real, path.imag, color=color, s=8, alpha=0.65)
        axes[0, 1].scatter(path.real[-1], path.imag[-1], color=color, s=45, marker="*")
        axes[0, 1].annotate(
            f"D{diagnosis:02d}",
            (path.real[-1], path.imag[-1]),
            xytext=(3, 2),
            textcoords="offset points",
        )
    axes[0, 1].axhline(0.0, color="#BBBBBB", linewidth=0.6)
    axes[0, 1].axvline(0.0, color="#BBBBBB", linewidth=0.6)
    axes[0, 1].set_xlabel("Real hypothesis amplitude")
    axes[0, 1].set_ylabel("Imaginary hypothesis amplitude")
    axes[0, 1].set_title(
        "B  Top hypotheses trace complex-plane paths", loc="left", fontweight="bold"
    )

    steps = np.arange(probabilities.shape[0])
    entropy = np.asarray(case["entropy"]) / np.log(20.0)
    velocity = np.asarray(case["velocity"])
    true_probability = probabilities[:, int(case["label"])]
    axes[1, 0].plot(steps, true_probability, color="#315A7D", label="True-hypothesis probability")
    axes[1, 0].plot(steps, probabilities.max(axis=-1), color="#16847A", label="Top probability")
    axes[1, 0].plot(steps, entropy, color="#D98B2B", label="Normalized entropy")
    axes[1, 0].plot(steps, velocity, color="#673C9E", label="State velocity")
    for index, token in enumerate(case["token_ids"], start=1):
        if int(token) >= 40:
            axes[1, 0].axvspan(index - 0.45, index + 0.45, color="#D05A2D", alpha=0.035)
    axes[1, 0].set_xlim(0, steps[-1])
    axes[1, 0].set_ylim(0, 1.03)
    axes[1, 0].set_xlabel("Evidence step")
    axes[1, 0].set_ylabel("Probability / normalized diagnostic")
    axes[1, 0].set_title(
        "C  Confidence, entropy, and motion are visible", loc="left", fontweight="bold"
    )
    axes[1, 0].legend(frameon=False, fontsize=7)

    first = np.asarray(pair["first_probabilities"])
    second = np.asarray(pair["second_probabilities"])
    label_a, label_b = [int(value) for value in pair["labels"]]
    pair_steps = np.arange(first.shape[0])
    axes[1, 1].plot(
        pair_steps, first[:, label_a], color="#315A7D", label=f"AB case → D{label_a:02d}"
    )
    axes[1, 1].plot(
        pair_steps, first[:, label_b], color="#D05A2D", label=f"AB case → D{label_b:02d}"
    )
    axes[1, 1].plot(
        pair_steps,
        second[:, label_a],
        color="#315A7D",
        linestyle="--",
        label=f"BA case → D{label_a:02d}",
    )
    axes[1, 1].plot(
        pair_steps,
        second[:, label_b],
        color="#D05A2D",
        linestyle="--",
        label=f"BA case → D{label_b:02d}",
    )
    axes[1, 1].set_xlim(0, pair_steps[-1])
    axes[1, 1].set_ylim(0, 1.03)
    axes[1, 1].set_xlabel("Evidence step")
    axes[1, 1].set_ylabel("Twin-hypothesis probability")
    axes[1, 1].set_title(
        "D  Reversing marker order bifurcates diagnosis", loc="left", fontweight="bold"
    )
    axes[1, 1].legend(frameon=False, fontsize=7, ncols=2)
    for axis in axes.flat[1:]:
        axis.grid(True, color="#DADADA", linewidth=0.6, alpha=0.8)
    figure.suptitle("Q-Neuro's actual internal computation", fontsize=13, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        (
            f"Seed {artifact['seed']}; deterministic first factorial case (true D{case['label']:02d}) and first chronology pair. "
            f"Source: {result_directory.name}; no generated reasoning trace."
        ),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    for extension in ("png", "pdf"):
        output_path = args.output_directory / f"trajectory_signature.{extension}"
        metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
        figure.savefig(output_path, dpi=220, bbox_inches="tight", metadata=metadata)
        print(output_path)
    plt.close(figure)


if __name__ == "__main__":
    main()
