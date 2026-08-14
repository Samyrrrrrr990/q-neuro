"""Generate extended-data figures from versioned result and research artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "figures" / "generated"
INK = "#17324D"
TEAL = "#168C86"
CORAL = "#D65A4A"
GOLD = "#D5A02B"
MIST = "#DDE9ED"
MODELS = {
    "mlp": "MLP",
    "transformer": "Transformer",
    "gru": "GRU",
    "real_operator": "Real operator",
    "two_channel_operator": "Two-channel",
    "complex_operator": "Complex operator",
}


def load(experiment_id: str) -> dict:
    return json.loads(
        (ROOT / "experiments" / "results" / experiment_id / "metrics.json").read_text(
            encoding="utf-8"
        )
    )


def metric(container: dict, name: str) -> float:
    return float(container[name]["mean"])


def save(figure: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(
        OUT / f"{name}.pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def architecture_overview() -> None:
    figure, axis = plt.subplots(figsize=(12, 4.8))
    axis.axis("off")
    boxes = [
        (0.03, 0.55, 0.15, 0.25, "Evidence\n(token, sign, time)", MIST),
        (0.25, 0.55, 0.17, 0.25, "Low-rank operator\nO(eₜ, Ψₜ)", "#CDEBE7"),
        (0.49, 0.55, 0.18, 0.25, "Complex hypothesis\nstate  Ψₜ ∈ ℂᴴ", "#B8DDD9"),
        (0.74, 0.55, 0.15, 0.25, "Measurement\n|WΨ|² / Σ|WΨ|²", "#F4DEDA"),
        (0.74, 0.10, 0.15, 0.22, "Differential\n+ uncertainty", "#F8ECE9"),
        (0.49, 0.10, 0.18, 0.22, "Actual trajectory\nprobability · phase · velocity", "#E8F1F4"),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(
            plt.Rectangle((x, y), width, height, transform=axis.transAxes, fc=color, ec=INK, lw=1.1)
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    for start, end in [
        ((0.18, 0.675), (0.25, 0.675)),
        ((0.42, 0.675), (0.49, 0.675)),
        ((0.67, 0.675), (0.74, 0.675)),
        ((0.815, 0.55), (0.815, 0.32)),
        ((0.58, 0.55), (0.58, 0.32)),
    ]:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.8},
        )
    axis.annotate(
        "",
        xy=(0.49, 0.61),
        xytext=(0.42, 0.58),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": CORAL, "connectionstyle": "arc3,rad=.45"},
    )
    axis.text(
        0.455,
        0.49,
        "ordered recurrence",
        transform=axis.transAxes,
        ha="center",
        color=CORAL,
        fontsize=9,
    )
    axis.text(
        0.03,
        0.94,
        "A  Q-Neuro is an evidence-driven state machine, not a physical quantum claim",
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    save(figure, "architecture_overview")


def experiment_map() -> None:
    ids = [3, 4, 6, 8, 10, 12, 14, 16, 19, 21, 23, 25, 26]
    labels = [
        "order",
        "samples",
        "strong controls",
        "world shift",
        "task suite",
        "active",
        "18 laws",
        "ablations",
        "probes",
        "training",
        "hard exit",
        "trajectories",
        "discovery",
    ]
    figure, axis = plt.subplots(figsize=(12, 3.4))
    axis.plot(ids, np.zeros(len(ids)), color=INK, linewidth=1.4)
    colors = [
        TEAL if value in {3, 8, 16, 25} else CORAL if value in {10, 21, 23} else GOLD
        for value in ids
    ]
    axis.scatter(
        ids, np.zeros(len(ids)), s=100, color=colors, edgecolor="white", linewidth=1.5, zorder=3
    )
    for index, (value, label) in enumerate(zip(ids, labels, strict=True)):
        offset = 0.16 if index % 2 == 0 else -0.16
        axis.text(value, offset, f"QN-{value:06d}\n{label}", ha="center", va="center", fontsize=8.5)
    axis.set_ylim(-0.31, 0.31)
    axis.set_xlim(2, 27)
    axis.axis("off")
    axis.set_title(
        "Registered evidence sequence: controls precede expansion",
        loc="left",
        fontweight="bold",
        color=INK,
    )
    save(figure, "experiment_evidence_map")


def pareto_field() -> None:
    records = json.loads(
        (ROOT / "research/discovery/generated/candidate_registry.json").read_text()
    )
    records = [item for item in records if item["context"] == "architecture"]
    figure, axis = plt.subplots(figsize=(7.4, 5.6))
    for item in records:
        size = 28 + 28 * np.sqrt(
            item["training_seconds"] / max(v["training_seconds"] for v in records)
        )
        axis.scatter(
            item["shifted_top1"],
            item["in_domain_top1"],
            s=size,
            color=TEAL if item["pareto"] else MIST,
            edgecolor=INK if item["pareto"] else "none",
            alpha=0.9,
        )
    for name in [
        "complex_operator",
        "gru",
        "real_operator",
        "two_channel_operator",
        "adaptive_attractor",
    ]:
        item = next(value for value in records if value["candidate_id"] == name)
        axis.annotate(
            name.replace("_", " "),
            (item["shifted_top1"], item["in_domain_top1"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xlabel="Moderate unseen-world top-1",
        ylabel="Source-world top-1",
        xlim=(0.1, 0.7),
        ylim=(0.15, 1.02),
    )
    axis.grid(alpha=0.2)
    axis.set_title(
        "Architecture accuracy tradeoff; Pareto status uses six objectives",
        loc="left",
        fontweight="bold",
    )
    save(figure, "architecture_pareto_field")


def calibration_transport() -> None:
    result = load("QN-000008")["summary"]
    models = ["real_operator", "two_channel_operator", "complex_operator"]
    raw = [metric(result[name]["moderate"]["across_worlds"], "ece") for name in models]
    calibrated = [
        metric(result[name]["moderate"]["across_worlds"], "calibrated_ece") for name in models
    ]
    x = np.arange(len(models))
    width = 0.34
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.bar(x - width / 2, raw, width, color=TEAL, label="raw")
    axis.bar(x + width / 2, calibrated, width, color=CORAL, label="source-fitted temperature")
    axis.set_xticks(x, [MODELS[name] for name in models])
    axis.set_ylabel("Moderate-shift ECE")
    axis.set_title(
        "Source calibration does not transport to shifted worlds", loc="left", fontweight="bold"
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save(figure, "calibration_transport")


def ambiguity_mass() -> None:
    result = load("QN-000010")["summary"]
    models = list(MODELS)
    mass = [metric(result[name]["base"], "ambiguity_twin_mass") for name in models]
    nll = [metric(result[name]["base"], "ambiguity_pair_nll") for name in models]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    order = np.argsort(mass)
    axes[0].barh(
        np.arange(len(models)),
        np.array(mass)[order],
        color=[CORAL if models[i] == "complex_operator" else TEAL for i in order],
    )
    axes[0].set_yticks(np.arange(len(models)), [MODELS[models[i]] for i in order])
    axes[0].axvline(1.0, color=INK, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Probability mass on two valid labels")
    axes[0].set_title("A  Valid differential mass", loc="left", fontweight="bold")
    order_nll = np.argsort(nll)[::-1]
    axes[1].barh(
        np.arange(len(models)),
        np.array(nll)[order_nll],
        color=[CORAL if models[i] == "complex_operator" else TEAL for i in order_nll],
    )
    axes[1].set_yticks(np.arange(len(models)), [MODELS[models[i]] for i in order_nll])
    axes[1].axvline(np.log(2), color=INK, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Ambiguous-pair NLL (lower is better)")
    axes[1].set_title("B  Irreducible ambiguity", loc="left", fontweight="bold")
    figure.tight_layout()
    save(figure, "ambiguity_differential")


def ood_scores() -> None:
    result = load("QN-000010")["summary"]
    models = list(MODELS)
    unknown = [metric(result[name]["unknown_disease"], "ood_auroc_msp") for name in models]
    hidden = [metric(result[name]["base"], "hidden_representation_ood_auroc") for name in models]
    x = np.arange(len(models))
    width = 0.34
    figure, axis = plt.subplots(figsize=(9, 4.7))
    axis.bar(x - width / 2, unknown, width, color=INK, label="omitted disease · MSP")
    axis.bar(x + width / 2, hidden, width, color=TEAL, label="hidden syndrome · representation")
    axis.set_xticks(x, [MODELS[name] for name in models], rotation=18)
    axis.set_ylim(0.45, 1.03)
    axis.set_ylabel("AUROC")
    axis.set_title(
        "OOD separability is strong but not uniquely complex", loc="left", fontweight="bold"
    )
    axis.legend(frameon=False, ncols=2)
    axis.grid(axis="y", alpha=0.2)
    save(figure, "ood_separability")


def sample_compute_frontier() -> None:
    result = load("QN-000004")["summary"]
    figure, axis = plt.subplots(figsize=(7.4, 5.1))
    palette = {
        "mlp": "#888888",
        "transformer": GOLD,
        "real_operator": INK,
        "complex_operator": TEAL,
    }
    label_layout = {
        "mlp": ((4, 3), "left"),
        "transformer": ((-4, -18), "right"),
        "real_operator": ((-4, 17), "right"),
        "complex_operator": ((-4, 4), "right"),
    }
    for model, values in result.items():
        for size, container in values.items():
            axis.scatter(
                metric(container, "training_seconds"),
                metric(container, "top1"),
                s=28 + int(size) / 160,
                color=palette[model],
                alpha=0.85,
            )
        largest = values[max(values, key=int)]
        offset, alignment = label_layout[model]
        axis.annotate(
            MODELS.get(model, model),
            (metric(largest, "training_seconds"), metric(largest, "top1")),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            ha=alignment,
        )
    axis.set_xscale("log")
    axis.set_xlabel("CPU training time (s, log scale)")
    axis.set_ylabel("Source top-1")
    axis.set_ylim(0.43, 1.055)
    axis.set_title("Sample-efficiency gains trade against CPU time", loc="left", fontweight="bold")
    axis.grid(alpha=0.2)
    save(figure, "sample_compute_frontier")


def claim_status() -> None:
    statuses = []
    for line in (ROOT / "docs/CLAIMS.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("|---") and "| Claim |" not in line:
            cells = [value.strip() for value in line.strip("|").split("|")]
            if len(cells) == 5:
                status = cells[-1].lower()
                if (
                    "replicated" in status
                    or "supported" in status
                    and "not supported" not in status
                ):
                    statuses.append("supported/replicated")
                elif (
                    "preliminary" in status or "unresolved" in status or "not established" in status
                ):
                    statuses.append("preliminary/unresolved")
                else:
                    statuses.append("refuted/unsupported")
    labels, counts = np.unique(statuses, return_counts=True)
    palette = {
        "supported/replicated": TEAL,
        "preliminary/unresolved": GOLD,
        "refuted/unsupported": CORAL,
    }
    figure, axis = plt.subplots(figsize=(7, 4.3))
    axis.barh(labels, counts, color=[palette[value] for value in labels])
    for index, value in enumerate(counts):
        axis.text(value + 0.3, index, str(value), va="center", fontsize=10)
    axis.set_xlabel("Claims in ledger")
    axis.set_title(
        "The claim ledger contains more boundaries than headlines", loc="left", fontweight="bold"
    )
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "claim_status_audit")


def surprise_taxonomy() -> None:
    surprises = json.loads((ROOT / "research/discovery/generated/surprises.json").read_text())
    types = [value["type"].replace("_", " ") for value in surprises]
    labels, counts = np.unique(types, return_counts=True)
    order = np.argsort(counts)
    figure, axis = plt.subplots(figsize=(7.6, 4.7))
    axis.barh(
        np.arange(len(labels)),
        counts[order],
        color=[TEAL if "ambiguity" in labels[i] else CORAL for i in order],
    )
    axis.set_yticks(np.arange(len(labels)), labels[order])
    axis.set_xlabel("Flagged candidates")
    axis.set_title(
        "Predeclared surprise rules expose recurring metric tensions", loc="left", fontweight="bold"
    )
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "surprise_taxonomy")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "text.color": INK,
        }
    )
    architecture_overview()
    experiment_map()
    pareto_field()
    calibration_transport()
    ambiguity_mass()
    ood_scores()
    sample_compute_frontier()
    claim_status()
    surprise_taxonomy()
    print("Generated 9 extended-data figure pairs")


if __name__ == "__main__":
    main()
