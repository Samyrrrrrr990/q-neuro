"""Generate manuscript tables directly from registered experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT = ROOT / "paper" / "tables"

MODEL_NAMES = {
    "logistic": "Logistic regression",
    "mlp": "MLP",
    "complex_mlp": "Complex MLP",
    "transformer": "Tiny Transformer",
    "gru": "Tuned GRU",
    "state_space": "Diagonal state-space",
    "real_operator": "Real operator",
    "two_channel_operator": "Two-channel real",
    "complex_operator": "Complex operator",
    "hamiltonian": "Hamiltonian-style",
    "dissipative": "Dissipative",
    "hybrid_dynamics": "Hamiltonian-dissipative",
    "density_dynamics": "Density dynamics (rank 2)",
    "complex_accumulator": "Commutative accumulator",
    "complex_magnitude_readout": "Magnitude-only readout",
    "complex_no_negative": "No observed-negative channel",
    "density_rank1": "Density dynamics (rank 1)",
    "density_rank2": "Density dynamics (rank 2)",
    "density_rank4": "Density dynamics (rank 4)",
}

TRAINING_NAMES = {
    "adamw": "AdamW",
    "sgd": "SGD",
    "gradient_accumulation": "Gradient accumulation",
    "multiobjective_adamw": "Multi-objective AdamW",
    "pcgrad": "PCGrad",
    "phase_gradient": "Phase Gradient Optimization",
    "local_plasticity": "Transition-local plasticity",
    "hybrid_local_global": "Local + global hybrid",
    "zerobackprop": "ZeroBackprop",
}


@dataclass
class Table:
    identifier: str
    caption: str
    columns: list[str]
    rows: list[list[str]]
    note: str


def load(experiment_id: str) -> dict[str, Any]:
    path = RESULTS / experiment_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def at(container: dict[str, Any], *path: str) -> Any:
    value: Any = container
    for part in path:
        value = value[part]
    return value


def mean(container: dict[str, Any], *path: str) -> float:
    value = at(container, *path)
    return float(value["mean"] if isinstance(value, dict) and "mean" in value else value)


def fmt(value: float, digits: int = 3) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def escape_tex(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "±": r"$\pm$",
        "→": r"$\rightarrow$",
        "↓": r"$\downarrow$",
        "×": r"$\times$",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def architecture_table() -> Table:
    summary = load("QN-000014")["summary"]
    models = [
        "logistic",
        "mlp",
        "transformer",
        "gru",
        "state_space",
        "real_operator",
        "two_channel_operator",
        "complex_operator",
        "hamiltonian",
        "dissipative",
        "hybrid_dynamics",
        "density_dynamics",
    ]
    rows = []
    for model in models:
        values = summary[model]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(mean(values, "in_domain", "parameter_count"), 0),
                fmt(mean(values, "in_domain", "top1")),
                fmt(mean(values, "moderate", "across_worlds", "top1")),
                fmt(mean(values, "in_domain", "ambiguity_pair_nll")),
                fmt(mean(values, "in_domain", "counterfactual_pair_accuracy")),
            ]
        )
    return Table(
        "architecture_results",
        "Matched architecture comparison at 1,000 training cases.",
        ["Model", "Parameters", "ID top-1", "Shift top-1", "Ambiguity NLL ↓", "Order pairs"],
        rows,
        "Means over three training seeds; shift values average three unseen moderate worlds.",
    )


def world_table() -> Table:
    summary = load("QN-000008")["summary"]
    models = [
        "mlp",
        "transformer",
        "gru",
        "real_operator",
        "two_channel_operator",
        "complex_operator",
    ]
    rows = []
    for model in models:
        values = summary[model]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(mean(values, "in_domain", "top1")),
                fmt(mean(values, "nuisance", "across_worlds", "top1")),
                fmt(mean(values, "mild", "across_worlds", "top1")),
                fmt(mean(values, "moderate", "across_worlds", "top1")),
                fmt(mean(values, "severe", "across_worlds", "top1")),
            ]
        )
    return Table(
        "world_robustness",
        "Top-1 accuracy across the preregistered unseen-world severity sweep.",
        ["Model", "In-domain", "Nuisance", "Mild", "Moderate", "Severe"],
        rows,
        "World intervals treat the unseen simulator seed as the replication unit after averaging training seeds.",
    )


def sample_table() -> Table:
    summary = load("QN-000004")["summary"]
    models = ["mlp", "transformer", "real_operator", "complex_operator"]
    rows = []
    for size in ["250", "500", "1000", "2000", "5000"]:
        rows.append(
            [f"{int(size):,}"] + [fmt(mean(summary[model][size], "top1")) for model in models]
        )
    return Table(
        "sample_efficiency",
        "Source-world top-1 accuracy under nested training-set sizes.",
        ["Cases", "MLP", "Transformer", "Real operator", "Complex operator"],
        rows,
        "The same validation/test generator and seed set is reused across sizes.",
    )


def ablation_table() -> Table:
    summary = load("QN-000016")["summary"]
    models = [
        "complex_accumulator",
        "two_channel_operator",
        "complex_magnitude_readout",
        "complex_no_negative",
        "complex_operator",
        "density_rank1",
        "density_rank2",
        "density_rank4",
    ]
    rows = []
    for model in models:
        values = summary[model]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(mean(values, "in_domain", "top1")),
                fmt(mean(values, "moderate", "across_worlds", "top1")),
                fmt(mean(values, "in_domain", "ambiguity_pair_nll")),
                fmt(mean(values, "in_domain", "counterfactual_pair_accuracy")),
            ]
        )
    return Table(
        "critical_ablations",
        "Critical ablations of order, phase-sensitive measurement, negative evidence, and density rank.",
        ["Law", "ID top-1", "Shift top-1", "Ambiguity NLL ↓", "Order pairs"],
        rows,
        "Each row is retrained from scratch with the same data split and three seeds.",
    )


def training_table() -> Table:
    summary = load("QN-000021")["summary"]
    methods = [
        "adamw",
        "sgd",
        "gradient_accumulation",
        "multiobjective_adamw",
        "pcgrad",
        "phase_gradient",
        "local_plasticity",
        "hybrid_local_global",
        "zerobackprop",
    ]
    rows = []
    for method in methods:
        values = summary[method]["1000"]
        reverse_mode_calls = mean(values, "in_domain", "backward_passes") + mean(
            values, "in_domain", "autograd_gradient_calls"
        )
        rows.append(
            [
                TRAINING_NAMES[method],
                fmt(mean(values, "in_domain", "top1")),
                fmt(mean(values, "shifted", "across_worlds", "top1")),
                fmt(mean(values, "in_domain", "ambiguity_pair_nll")),
                fmt(mean(values, "in_domain", "training_seconds"), 2),
                fmt(reverse_mode_calls, 0),
            ]
        )
    return Table(
        "training_laws",
        "Training-law comparison for the complex operator at 1,000 cases.",
        [
            "Training law",
            "ID top-1",
            "Shift top-1",
            "Ambiguity NLL ↓",
            "Train s",
            "Reverse-mode calls",
        ],
        rows,
        "Shift top-1 is averaged across the declared unseen worlds; timing is CPU wall time.",
    )


def halting_table() -> Table:
    summary = load("QN-000023")["summary"]
    names = {"soft": "Soft ACT", "fixed_final": "Fixed eight states", "hard": "Hard velocity exit"}
    rows = []
    for method in ["soft", "fixed_final", "hard"]:
        values = summary[method]
        executed_steps = (
            8.0 if method == "soft" else mean(values, "in_domain", "mean_executed_steps")
        )
        rows.append(
            [
                names[method],
                fmt(mean(values, "in_domain", "top1")),
                fmt(mean(values, "shifted", "across_worlds", "top1")),
                fmt(mean(values, "shifted", "across_worlds", "nll")),
                fmt(mean(values, "shifted", "across_worlds", "ece")),
                fmt(executed_steps, 2),
                fmt(mean(values, "in_domain", "latency_ms_per_case"), 4),
            ]
        )
    return Table(
        "hard_halting",
        "Hard-halting audit of executed states and measured inference latency.",
        ["Inference law", "ID top-1", "Shift top-1", "Shift NLL", "Shift ECE", "States", "ms/case"],
        rows,
        "The hard exit removes halted examples from subsequent computation; it selected two states for every case.",
    )


def uncertainty_table() -> Table:
    summary = load("QN-000010")["summary"]
    models = [
        "mlp",
        "transformer",
        "gru",
        "real_operator",
        "two_channel_operator",
        "complex_operator",
    ]
    rows = []
    for model in models:
        values = summary[model]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(mean(values, "base", "ambiguity_pair_nll")),
                fmt(mean(values, "base", "ambiguity_twin_mass")),
                fmt(mean(values, "unknown_disease", "ood_auroc_msp")),
                fmt(mean(values, "base", "hidden_representation_ood_auroc")),
            ]
        )
    return Table(
        "uncertainty_ood",
        "Irreducible ambiguity and out-of-distribution separation.",
        [
            "Model",
            "Ambiguity NLL ↓",
            "Valid-twin mass",
            "Omitted-disease AUROC",
            "Hidden-syndrome AUROC",
        ],
        rows,
        "OOD AUROC measures separation on synthetic held-out constructs, not clinical disease discovery.",
    )


def trajectory_table() -> Table:
    summary = load("QN-000025")["summary"]
    labels = [
        ("Final top-1", "final_top1"),
        ("Entropy change", "mean_entropy_change"),
        ("Normalized path length", "mean_normalized_path_length"),
        ("Final state velocity", "mean_final_state_velocity"),
        ("Positive-token Δ true probability", "mean_positive_token_delta_true_probability"),
        ("Observed-negative Δ true probability", "mean_negative_token_delta_true_probability"),
        ("Chronology-twin final distance", "mean_counterfactual_final_state_distance"),
        ("Contradiction drops >0.05", "negative_contradiction_drop_case_fraction"),
        ("Recovery after a qualifying drop", "revival_given_negative_drop_fraction"),
    ]
    rows = []
    for label, key in labels:
        metric = summary[key]
        rows.append(
            [
                label,
                fmt(float(metric["mean"])),
                f"[{fmt(float(metric['ci95_low']))}, {fmt(float(metric['ci95_high']))}]",
            ]
        )
    return Table(
        "trajectory_metrics",
        "Evidence-level trajectory diagnostics for the trained complex operator.",
        ["Quantity", "Mean", "95% Student-t interval"],
        rows,
        "Trajectories are actual recurrent states; they are diagnostics, not textual explanations.",
    )


def full_data_table() -> Table:
    summary = load("QN-000003")["summary"]
    models = ["mlp", "transformer", "real_operator", "complex_operator"]
    rows = []
    for model in models:
        values = summary[model]
        rows.append(
            [
                MODEL_NAMES[model],
                fmt(mean(values, "parameter_count"), 0),
                fmt(mean(values, "top1")),
                fmt(mean(values, "counterfactual_pair_accuracy")),
                fmt(mean(values, "nll")),
                fmt(mean(values, "ece")),
                fmt(mean(values, "training_seconds"), 2),
            ]
        )
    return Table(
        "full_data",
        "Corrected full-data Experiment Zero comparison.",
        ["Model", "Parameters", "Top-1", "Order pairs", "NLL", "ECE", "Train s"],
        rows,
        "Four matched models, 14,000 training cases, 3,000 validation cases, 3,000 test cases, and three seeds.",
    )


def write_table(table: Table) -> None:
    (OUT / f"{table.identifier}.json").write_text(
        json.dumps(asdict(table), indent=2) + "\n", encoding="utf-8"
    )
    columns = "l" + "r" * (len(table.columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{escape_tex(table.caption)}}}",
        rf"\label{{tab:{table.identifier}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(escape_tex(value) for value in table.columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(escape_tex(value) for value in row) + r" \\" for row in table.rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\begin{{minipage}}{{0.96\linewidth}}\footnotesize {escape_tex(table.note)}\end{{minipage}}",
            r"\end{table}",
            "",
        ]
    )
    (OUT / f"{table.identifier}.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = [
        full_data_table(),
        sample_table(),
        world_table(),
        architecture_table(),
        uncertainty_table(),
        ablation_table(),
        training_table(),
        halting_table(),
        trajectory_table(),
    ]
    for table in tables:
        write_table(table)
    print(f"Generated {len(tables)} synchronized manuscript tables in {OUT}")


if __name__ == "__main__":
    main()
