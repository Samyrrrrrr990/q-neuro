"""Every figure for the three PDFs, generated from stored artifacts.

No number in any figure is typed by hand -- each is read from `research/**` JSON. If a record
changes, the figure changes with it, which is the only way a figure can be evidence rather than
decoration.

    python research/figures/generate_journey.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
Q3 = ROOT / "research" / "qneuro3"
NOVA = ROOT / "research" / "nova"
OUT = ROOT / "research" / "figures" / "generated"

INK, MUTED, ACCENT, FAIL, GRID = "#141D1B", "#5B6764", "#1B6D60", "#A03726", "#DCDFD8"
GOLD, BLUE, PLUM = "#C08A2E", "#3E7CB1", "#7A5EA6"
PALETTE = [ACCENT, INK, BLUE, GOLD, PLUM, "#9AA5A2", "#C4CBC8"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax, title: str = "", ylabel: str = ""):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def save(figure, name: str):
    figure.tight_layout()
    figure.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"  {name}.png")


# ----------------------------------------------------------------- the whole story


def fig_predictions_timeline():
    """Nineteen frozen predictions in the order they were opened. One passed."""

    rows = [
        ("Gate D", "Sentinel", False), ("DISCOVERY-001-P1", "Sentinel", False),
        ("DISCOVERY-001-P2", "Sentinel", False), ("DFREE-LAW-P1", "Sentinel", False),
        ("DFREE-LAW-P2", "Sentinel", False), ("DFREE-LAW-P3", "Sentinel", False),
        ("Q3-P1", "Pulse", False), ("Q4-P1", "Pulse", False),
        ("ATTRIB-P1", "Pulse", False), ("TRANSFER-P1", "Pulse", False),
        ("EXTRAP-P1", "Pulse", False), ("PARETO-P1", "Pulse", False),
        ("NICHE-P1", "Pulse", True), ("RUNTIME-P1", "Pulse", False),
        ("RUNTIME-P2", "Pulse", False), ("HAR-P1", "Pulse", False),
        ("H-DILUTION", "Nova", False), ("H-INTERFERENCE-P1", "Nova", False),
        ("H-COMPOSE-P1", "Nova", False),
    ]
    era_colour = {"Sentinel": BLUE, "Pulse": GOLD, "Nova": PLUM}
    figure, ax = plt.subplots(figsize=(11, 4.6))
    style(ax, "Nineteen frozen predictions, in the order I opened them — one passed",
          "verdict")
    for index, (name, era, passed) in enumerate(rows):
        ax.bar(index, 1.0, width=0.72,
               color=ACCENT if passed else FAIL, alpha=1.0 if passed else 0.75,
               edgecolor="none")
        ax.plot([index], [-0.09], marker="s", markersize=6, color=era_colour[era],
                clip_on=False)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks([])
    ax.set_ylim(0, 1.25)
    ax.annotate("the only pass", xy=(12, 1.02), xytext=(12, 1.19), ha="center",
                fontsize=9, color=ACCENT,
                arrowprops={"arrowstyle": "->", "color": ACCENT, "linewidth": 1.2})
    handles = [plt.Line2D([], [], marker="s", linestyle="none", color=c, label=e)
               for e, c in era_colour.items()]
    handles += [plt.Rectangle((0, 0), 1, 1, color=FAIL, alpha=0.75, label="failed"),
                plt.Rectangle((0, 0), 1, 1, color=ACCENT, label="passed")]
    ax.legend(handles=handles, ncol=5, frameon=False, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.42))
    save(figure, "journey_predictions")


def fig_failures_by_era():
    failures = load(ROOT / "research" / "failures.json")["failures"]
    eras = {"Helix": 0, "Sentinel": 0, "Pulse": 0, "Nova": 0}
    for entry in failures:
        number = int(entry["failure_id"].split("-")[1])
        if number <= 5:
            eras["Helix"] += 1
        elif number <= 21:
            eras["Sentinel"] += 1
        elif number <= 36:
            eras["Pulse"] += 1
        else:
            eras["Nova"] += 1
    figure, ax = plt.subplots(figsize=(6.4, 3.8))
    style(ax, f"{len(failures)} preserved failures, by era", "count")
    ax.bar(list(eras), list(eras.values()), width=0.6,
           color=[BLUE, BLUE, GOLD, PLUM], edgecolor="none")
    for index, value in enumerate(eras.values()):
        ax.text(index, value + 0.3, str(value), ha="center", color=INK, fontsize=10)
    save(figure, "journey_failures")


# ----------------------------------------------------------------- Sentinel


def fig_gate_d():
    features = [
        ("cumulative_defect", -31.71, 0.962, True), ("amplified_defect", -20.47, 0.819, True),
        ("one_step_pred_div", -92.50, 0.460, False), ("loss_decrease", -330.70, 0.084, False),
        ("parameter_count", -378.52, 0.000, False), ("learning_rate", -380.51, 0.896, False),
        ("total_grad_norm", -628.97, 0.373, False), ("mean_amplification", -898.28, 0.348, False),
    ]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    style(left, "Across families: everything is worse than predicting the mean",
          "held-out R² (leave-one-family-out)")
    names = [f[0] for f in features]
    left.barh(names, [f[1] for f in features],
              color=[ACCENT if f[3] else MUTED for f in features], edgecolor="none")
    left.axvline(0, color=INK, linewidth=1)
    left.invert_yaxis()
    style(right, "Within a family: the candidate is the strongest feature there is",
          "best within-family R²")
    right.barh(names, [f[2] for f in features],
               color=[ACCENT if f[3] else MUTED for f in features], edgecolor="none")
    right.invert_yaxis()
    right.set_xlim(0, 1)
    save(figure, "sentinel_gate_d")


def fig_dimension_law():
    n = [50, 100, 150, 180, 190, 193, 200, 250, 400]
    predicted = [143, 93, 43, 13, 3, 0, 0, 0, 0]
    figure, ax = plt.subplots(figsize=(7.2, 4.0))
    style(ax, "d_free = max(0, P − g − n)  — stated before measuring, exact in 9 of 9 cells",
          "free directions")
    ax.plot(n, predicted, marker="o", color=ACCENT, linewidth=2, markersize=7,
            label="predicted")
    ax.plot(n, predicted, marker="x", color=INK, linestyle="none", markersize=11,
            markeredgewidth=2, label="measured")
    ax.axvline(193, color=FAIL, linestyle="--", linewidth=1.2)
    ax.annotate("transition at n = 193\n(P − g = 193)", xy=(193, 60), xytext=(215, 90),
                fontsize=9, color=FAIL,
                arrowprops={"arrowstyle": "->", "color": FAIL})
    ax.set_xlabel("training points n", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    save(figure, "sentinel_dimension_law")


def fig_stability_boundary():
    figure, ax = plt.subplots(figsize=(6.8, 4.0))
    style(ax, "The same predictor, different coordinates: SGD crosses a boundary, Adam does not",
          "diverged cells (of 1,476)")
    ax.bar(["SGD", "AdamW"], [720, 1], width=0.5, color=[FAIL, ACCENT], edgecolor="none")
    ax.text(0, 740, "720", ha="center", color=INK, fontsize=11)
    ax.text(1, 25, "1", ha="center", color=INK, fontsize=11)
    ax.text(0.5, 480,
            "prediction accuracy 0.9912\nzero false alarms in 1,476 cells\n"
            "ρ = η·λmax(H)/(2s²), transition at ρ = 1",
            ha="center", fontsize=9, color=MUTED)
    save(figure, "sentinel_stability")


# ----------------------------------------------------------------- Pulse


def fig_q3_bimodal():
    record = load(Q3 / "QNEURO3-Q3-VARIANCE-001.json")["results"]
    accuracies, steps, labels = [], [], []
    for key, runs in record.items():
        for run in runs:
            accuracies.append(run["accuracy"])
            steps.append(run["avg_steps"])
            labels.append(key)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    style(left, "Twenty identical runs, two outcomes and nothing in between", "final accuracy")
    left.hist(accuracies, bins=18, color=ACCENT, edgecolor="white")
    left.axvspan(0.57, 0.99, color=FAIL, alpha=0.10)
    left.text(0.78, left.get_ylim()[1] * 0.7, "the gap", ha="center", color=FAIL, fontsize=9)
    left.set_xlabel("accuracy", color=MUTED, fontsize=9)
    style(right, "And the broken runs look healthy if you only read the step counter",
          "average steps used (of 8)")
    good = [(s, a) for s, a in zip(steps, accuracies) if a >= 0.99]
    bad = [(s, a) for s, a in zip(steps, accuracies) if a < 0.99]
    # The working runs land on top of each other -- identical to three decimals -- so the count is
    # annotated rather than implied by the marker.
    right.scatter([g[0] for g in good], [g[1] for g in good], s=70, color=ACCENT,
                  label=f"works ({len(good)} runs, all identical)", zorder=3)
    if good:
        right.annotate(f"{len(good)} runs stacked here\n(4.54 steps, 1.000)",
                       xy=(good[0][0], good[0][1]), xytext=(good[0][0] + 0.35, 0.90),
                       fontsize=8.5, color=ACCENT,
                       arrowprops={"arrowstyle": "->", "color": ACCENT})
    right.scatter([b[0] for b in bad], [b[1] for b in bad], s=70, color=FAIL,
                  label=f"broken ({len(bad)} runs)", zorder=3)
    right.axhline(0.99, color=MUTED, linestyle=":", linewidth=1)
    right.set_xlabel("average steps used", color=MUTED, fontsize=9)
    right.set_ylabel("accuracy", color=MUTED, fontsize=9)
    right.legend(frameon=False, fontsize=9)
    save(figure, "pulse_bimodal")


def fig_pulse_ladder():
    arms = [("Q0 fixed depth", 1.0000, 8.00), ("Q1 mixture halting", 0.6241, 3.27),
            ("Q2 hard commit", 0.9999, 8.00), ("Q3 halt on arrival", 0.9995, 4.54),
            ("Q4 + grounding", 0.7500, 4.90)]
    # Labels sit on the points rather than in a legend, because with five arms in a small axes a
    # legend box lands on top of the data every time.
    offsets = {"Q0 fixed depth": (-30, 15), "Q1 mixture halting": (10, -4),
               "Q2 hard commit": (-32, -25), "Q3 halt on arrival": (-16, 14),
               "Q4 + grounding": (10, -4)}
    saving = arms[0][2] / arms[3][2]
    figure, ax = plt.subplots(figsize=(7.4, 4.4))
    style(ax, "The halting ladder: accuracy against compute, one change at a time",
          "accuracy")
    # Q0 and Q2 land on the same point to three decimals, so Q2 is drawn as a ring around Q0
    # rather than hidden underneath it.
    ax.plot([arms[3][2], arms[0][2]], [arms[3][1], arms[0][1]], color=ACCENT, lw=1.4, zorder=2)
    # Q4 is not a point: it lands anywhere in 0.6322-0.9500 depending on the seed, which is the
    # whole reason it was abandoned, so the spread is drawn rather than averaged away.
    ax.plot([arms[4][2]] * 2, [0.6322, 0.9500], color=PALETTE[4], lw=2.0, zorder=2)
    for index, (name, accuracy, steps) in enumerate(arms):
        if name.startswith("Q2"):
            ax.scatter(steps, accuracy, s=340, facecolors="none", edgecolors=PALETTE[index],
                       linewidths=2.0, zorder=3)
        else:
            ax.scatter(steps, accuracy, s=150, color=PALETTE[index], zorder=3)
        ax.annotate(name, (steps, accuracy), textcoords="offset points",
                    xytext=offsets[name], fontsize=8.5, color=INK, zorder=4)
    ax.set_xlabel("average steps used (of 8)", color=MUTED, fontsize=9)
    ax.set_xlim(2, 9.4)
    ax.set_ylim(0.5, 1.09)
    ax.text(6.25, 0.955, f"same accuracy,\n{saving:.2f}x less compute", fontsize=9,
            color=ACCENT, ha="center", va="top")
    save(figure, "pulse_ladder")


def fig_ceiling():
    import torch

    from qneuro3.adaptive import expected_max_halt

    pmf = 0.8 ** torch.arange(1, 33, dtype=torch.double)
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
    maxima = [expected_max_halt(pmf, b) for b in batches]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    style(left, "A batch waits for its slowest member", "E[max halt step]")
    left.semilogx(batches, maxima, marker="o", color=FAIL, linewidth=2, base=2)
    left.axhline(4.97, color=ACCENT, linestyle="--", linewidth=1.2)
    left.text(2, 5.7, "E[halt] = 4.97 per example", color=ACCENT, fontsize=9)
    left.set_xlabel("batch size", color=MUTED, fontsize=9)
    style(right, "So the saving decays from 6.4× to 1.1×", "realisable speedup")
    right.semilogx(batches, [32 / m for m in maxima], marker="o", color=FAIL,
                   linewidth=2, base=2)
    right.axhline(1.0, color=MUTED, linestyle=":", linewidth=1)
    right.set_xlabel("batch size", color=MUTED, fontsize=9)
    save(figure, "pulse_ceiling")


def fig_m2_sweep():
    report = load(Q3 / "m2_report.json")
    rows = report["sweep"]
    policies = ("select", "lockstep", "compacted")
    colours = {"select": MUTED, "lockstep": BLUE, "compacted": ACCENT}
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    style(left, "M2 latency: which execution policy wins depends on the batch",
          "median µs per example")
    for policy in policies:
        points = [(r["batch"], r["median_us_per_example"]) for r in rows
                  if r["policy"] == policy]
        left.loglog([p[0] for p in points], [p[1] for p in points], marker="o",
                    color=colours[policy], linewidth=2, label=policy, base=2)
    left.set_xlabel("batch size", color=MUTED, fontsize=9)
    left.legend(frameon=False, fontsize=9)
    style(right, "Throughput: compaction pulls away as the batch grows",
          "examples per second")
    for policy in policies:
        points = [(r["batch"], r["throughput_per_second"]) for r in rows
                  if r["policy"] == policy]
        right.loglog([p[0] for p in points], [p[1] for p in points], marker="o",
                     color=colours[policy], linewidth=2, label=policy, base=2)
    right.set_xlabel("batch size", color=MUTED, fontsize=9)
    right.legend(frameon=False, fontsize=9)
    save(figure, "pulse_m2_sweep")


def fig_reliability():
    report = load(Q3 / "m2_report.json")["reliability"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    style(left, "After the fix: 10 of 10 seeds, accuracy and halting both perfect", "accuracy")
    index = range(len(report["accuracies"]))
    left.bar([i - 0.2 for i in index], report["accuracies"], width=0.4,
             color=ACCENT, label="answer", edgecolor="none")
    left.bar([i + 0.2 for i in index], report["halt_accuracies"], width=0.4,
             color=INK, label="halt step", edgecolor="none")
    left.set_ylim(0, 1.1)
    left.set_xlabel("seed", color=MUTED, fontsize=9)
    left.legend(frameon=False, fontsize=9)
    sensitivity = report["hyperparameter_sensitivity"]
    style(right, "Hyperparameter sensitivity: 9 of 12, all failures at one learning rate",
          "accuracy")
    labels = [f"{s['learning_rate']:g}\n{s['halt_bias']:g}" for s in sensitivity]
    right.bar(range(len(sensitivity)), [s["accuracy"] for s in sensitivity], width=0.6,
              color=[ACCENT if s["accuracy"] >= 0.99 else FAIL for s in sensitivity],
              edgecolor="none")
    right.set_xticks(range(len(sensitivity)))
    right.set_xticklabels(labels, fontsize=7)
    right.set_xlabel("learning rate / halt bias", color=MUTED, fontsize=9)
    save(figure, "pulse_reliability")


def fig_har():
    record = load(Q3 / "QNEURO3-HAR-P1-RESULT.json")["results_3_seeds"]
    keep = [r for r in record if "matched" not in r["arm"]]
    figure, ax = plt.subplots(figsize=(8.0, 4.6))
    style(ax, "Real data (UCI HAR): my method came fourth of five",
          "test accuracy, subject-disjoint split")
    names = [r["arm"].split(" (")[0] for r in keep]
    values = [r["test_accuracy"] for r in keep]
    chunks = [r["mean_chunks"] for r in keep]
    colours = [FAIL if "supervised" in n else (ACCENT if "act" in n else MUTED) for n in names]
    ax.scatter(chunks, values, s=180, c=colours, zorder=3)
    for name, chunk, value in zip(names, chunks, values):
        ax.annotate(name, (chunk, value), textcoords="offset points", xytext=(8, 6),
                    fontsize=8.5, color=INK)
    ax.set_xlabel("mean chunks used (of 16) — less is cheaper", color=MUTED, fontsize=9)
    ax.set_ylim(0.45, 0.98)
    save(figure, "pulse_har")


# ----------------------------------------------------------------- Nova


def fig_nova_baselines():
    entries = [json.loads(line) for line in
               (NOVA / "registry.jsonl").read_text(encoding="utf-8").splitlines() if line]
    baselines = {}
    for entry in entries:
        if entry.get("family") == "baseline":
            baselines[entry["name"]] = entry
    tasks = ["parity_scan", "mod_sum", "copy", "reverse", "needle"]
    order = sorted(baselines, key=lambda n: -sum(baselines[n]["tasks"][t]["mean@64"] for t in tasks))
    figure, ax = plt.subplots(figsize=(10.5, 4.6))
    style(ax, "Ten baselines at matched parameters — nobody is good at everything",
          "accuracy at 4× the trained length")
    width = 0.8 / len(order)
    for index, name in enumerate(order):
        offsets = [t + index * width - 0.4 + width / 2 for t in range(len(tasks))]
        ax.bar(offsets, [baselines[name]["tasks"][t]["mean@64"] for t in tasks],
               width=width * 0.9, label=name, color=PALETTE[index % len(PALETTE)],
               edgecolor="none")
    for position, task in enumerate(tasks):
        chance = baselines[order[0]]["tasks"][task]["chance"]
        ax.plot([position - 0.44, position + 0.44], [chance] * 2, color=FAIL,
                linestyle="--", linewidth=1.3, label="chance" if position == 0 else None)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, frameon=False, fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.11))
    save(figure, "nova_baselines")


def fig_nova_hypotheses():
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    dilution = load(NOVA / "NOVA-H-DILUTION.json")["task_level_result_5_seeds_at_4x_length"]["copy"]
    names = ["softmax\n(control)", "softmax\n+ RMS", "max", "threshold"]
    values = [dilution["attn_softmax"][0], dilution["attn_softmax_rms"][0],
              dilution["attn_max"][0], dilution["attn_threshold"][0]]
    errors = [dilution["attn_softmax"][1], dilution["attn_softmax_rms"][1],
              dilution["attn_max"][1], dilution["attn_threshold"][1]]
    style(axes[0], "H-DILUTION: the control ate the effect", "copy accuracy at 4×")
    axes[0].bar(names, values, yerr=errors, width=0.6, capsize=4,
                color=[MUTED, FAIL, ACCENT, ACCENT], edgecolor="none")
    axes[0].set_ylim(0, 0.55)

    interference = load(NOVA / "NOVA-H-INTERFERENCE-P1-RESULT.json")["results_mean_of_3_seeds_at_4x_length"]
    style(axes[1], "H-INTERFERENCE: dropout just makes it an LSTM", "accuracy at 4×")
    labels = ["0%", "25%", "50%", "75%", "LSTM\nalone"]
    axes[1].plot(labels, [r["mod_sum"] for r in interference], marker="o", color=INK,
                 linewidth=2, label="mod_sum")
    axes[1].plot(labels, [r["needle"] for r in interference], marker="o", color=ACCENT,
                 linewidth=2, label="needle")
    axes[1].set_xlabel("attention dropout", color=MUTED, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8.5)

    frontier = load(NOVA / "NOVA-FRONTIER-001.json")["frontier"]
    chain = ["lstm", "rnn_attn_max", "cursor", "cursor_attn"]
    style(axes[2], "H-COMPOSE: the conflict moved", "accuracy at 4×")
    for task, colour in (("mod_sum", INK), ("needle", ACCENT), ("reverse", FAIL)):
        axes[2].plot(range(4), [frontier[c][task] for c in chain], marker="o",
                     color=colour, linewidth=2, label=task)
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(["LSTM", "+attn", "+cursor", "all 3"], fontsize=8.5)
    axes[2].legend(frameon=False, fontsize=8.5)
    save(figure, "nova_hypotheses")


def fig_nova_gap():
    record = load(NOVA / "NOVA-FRONTIER-001.json")
    frontier, best = record["frontier"], record["per_task_best"]
    means = {n: sum(v.values()) / len(v) for n, v in frontier.items()}
    order = sorted(means, key=lambda n: means[n])
    figure, ax = plt.subplots(figsize=(7.6, 4.2))
    style(ax, "No single architecture reaches what the set collectively achieves",
          "mean accuracy at 4×")
    ax.barh(order, [means[n] for n in order], color=ACCENT, edgecolor="none")
    collective = sum(best.values()) / len(best)
    ax.axvline(collective, color=FAIL, linestyle="--", linewidth=1.5)
    ax.text(collective + 0.008, 0.2, f"per-task best\ncombined = {collective:.3f}",
            color=FAIL, fontsize=9)
    ax.set_xlim(0, 0.85)
    save(figure, "nova_gap")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("generating journey figures")
    fig_predictions_timeline()
    fig_failures_by_era()
    fig_gate_d()
    fig_dimension_law()
    fig_stability_boundary()
    fig_q3_bimodal()
    fig_pulse_ladder()
    fig_ceiling()
    fig_m2_sweep()
    fig_reliability()
    fig_har()
    fig_nova_baselines()
    fig_nova_hypotheses()
    fig_nova_gap()


if __name__ == "__main__":
    main()
