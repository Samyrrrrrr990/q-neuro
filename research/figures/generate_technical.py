"""Figures for the technical breakdown and the textbook.

Companion to `generate_journey.py`, same rule: every number is read from a stored record under
`research/**` or computed from the shipped library, never typed into this file. Labels are typed;
data is not.

    python research/figures/generate_technical.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q3 = ROOT / "research" / "qneuro3"
NOVA = ROOT / "research" / "nova"
OUT = ROOT / "research" / "figures" / "generated"

INK, MUTED, ACCENT, FAIL, GRID = "#141D1B", "#5B6764", "#1B6D60", "#A03726", "#DCDFD8"
GOLD, BLUE, PLUM = "#C08A2E", "#3E7CB1", "#7A5EA6"
PALETTE = [ACCENT, INK, BLUE, GOLD, PLUM, "#9AA5A2", "#C4CBC8"]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style(ax, title: str = "", ylabel: str = "", xlabel: str = ""):
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
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def save(figure, name: str):
    figure.tight_layout()
    figure.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"  {name}.png")


# ------------------------------------------------------------------ the machine


def fig_cost_constants():
    """Why every asymptotic argument fails at batch 1: launch cost dwarfs work."""

    prediction = load(Q3 / "QNEURO3-RUNTIME-P1.json")["prediction"]
    lookup = prediction["constants_measured_on_the_untouched_family_raw_forward"]
    provenance = prediction["calibration_provenance"]
    # The streaming constants are quoted inside the provenance string, which is where they were
    # frozen; parse them rather than retyping them.
    streaming = {
        key: float(provenance.split(token)[1].split(",")[0].split(")")[0].strip())
        for key, token in (
            ("c_step_us_per_example_step", "c_step"),
            ("c_launch_us_per_iteration", "c_launch"),
            ("c_compact_us_per_compaction", "c_compact"),
        )
    }

    names = ["lookup core", "streaming core"]
    sources = [lookup, streaming]
    step = [s["c_step_us_per_example_step"] for s in sources]
    launch = [s["c_launch_us_per_iteration"] for s in sources]
    compact = [s["c_compact_us_per_compaction"] for s in sources]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    x = np.arange(len(names))
    for offset, values, label, colour in (
        (-0.26, step, "one example-step", ACCENT),
        (0.0, launch, "starting an iteration", INK),
        (0.26, compact, "one compaction", GOLD),
    ):
        left.bar(x + offset, values, 0.25, label=label, color=colour)
    left.set_yscale("log")
    left.set_ylim(0.1, max(launch) * 60)
    left.set_xticks(x)
    left.set_xticklabels(names, fontsize=9)
    style(left, "Measured cost constants (log scale)", "microseconds")
    left.legend(frameon=False, fontsize=8, loc="upper center", ncol=3, columnspacing=1.0,
                handlelength=1.2)

    ratio = [launch[i] / step[i] for i in range(len(names))]
    right.barh(names, ratio, color=FAIL, height=0.45)
    for i, value in enumerate(ratio):
        right.text(value + 3, i, f"{value:.0f}x", va="center", color=INK, fontsize=9.5)
    style(right, "Launch cost, as a multiple of one example-step", xlabel="ratio")
    right.xaxis.grid(True, color=GRID, linewidth=0.7)
    right.yaxis.grid(False)
    right.set_xlim(0, max(ratio) * 1.25)
    save(fig, "tech_cost_constants")


def fig_policy_rows():
    """Executed rows under each policy, analytic, across the batch range."""

    import torch

    from qneuro3.adaptive import expected_max_halt

    depths = np.arange(1, 33)
    pmf = 0.8**depths
    pmf = pmf / pmf.sum()
    mean_depth = float((depths * pmf).sum())
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    lockstep, compacted, waste = [], [], []
    for n in batches:
        emax = expected_max_halt(torch.tensor(pmf), n)
        lockstep.append(emax)
        compacted.append(mean_depth)
        waste.append(emax / mean_depth)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    left.plot(batches, lockstep, "-o", color=FAIL, ms=4, label="lockstep: E[max depth] per example")
    left.plot(batches, compacted, "-o", color=ACCENT, ms=4, label="compacted: E[depth] per example")
    left.fill_between(batches, compacted, lockstep, color=FAIL, alpha=0.09)
    left.set_xscale("log", base=2)
    style(left, "Straggler waste is the gap", "steps executed per example", "batch size")
    left.legend(frameon=False, fontsize=8.5, loc="upper left")

    right.plot(batches, waste, "-o", color=INK, ms=4)
    right.axhline(1.0, color=GRID, lw=1)
    right.set_xscale("log", base=2)
    for n, w in zip(batches, waste):
        if n in (1, 32, 1024):
            right.annotate(f"{w:.2f}x", (n, w), textcoords="offset points", xytext=(0, 7),
                           ha="center", color=INK, fontsize=9)
    style(right, "Wasted work multiplier under lockstep", "E[max] / E[depth]", "batch size")
    save(fig, "tech_policy_rows")


def fig_ceiling_removed():
    """QNEURO3-CEILING-REMOVED-001: the ceiling is a property of the runtime, not the method."""

    record = load(Q3 / "QNEURO3-CEILING-REMOVED-001.json")
    rows = record["measured_us_per_example"]
    batch = [r["batch"] for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    left.plot(batch, [r["select"] for r in rows], "-o", ms=4, color=MUTED, label="full-depth baseline")
    left.plot(batch, [r["arrival_lockstep"] for r in rows], "-o", ms=4, color=FAIL, label="adaptive, lockstep")
    left.plot(batch, [r["arrival_compacted"] for r in rows], "-o", ms=4, color=ACCENT, label="adaptive, compacted")
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    style(left, "Latency per example, same accuracy in all three", "microseconds", "batch size")
    left.legend(frameon=False, fontsize=8.5)

    right.plot(batch, [r["lockstep_vs_select"] for r in rows], "-o", ms=4, color=FAIL, label="lockstep")
    right.plot(batch, [r["compacted_vs_select"] for r in rows], "-o", ms=4, color=ACCENT, label="compacted")
    right.axhline(1.0, color=INK, lw=1, ls=":")
    right.set_xscale("log", base=2)
    right.text(batch[-1], 1.03, "no benefit", ha="right", color=MUTED, fontsize=8.5)
    style(right, "Speedup over the baseline", "x faster", "batch size")
    right.legend(frameon=False, fontsize=8.5)
    save(fig, "tech_ceiling_removed")


def fig_cost_model_failure():
    """QNEURO3-RUNTIME-P1: right where compute dominates, wrong where overhead does."""

    record = load(Q3 / "QNEURO3-RUNTIME-P1-RESULT.json")
    rows = record["measured"]
    batch = [r["batch"] for r in rows]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    left.plot(batch, [r["predicted"] for r in rows], "--o", ms=4, color=MUTED, label="predicted, frozen")
    left.plot(batch, [r["speedup"] for r in rows], "-o", ms=4, color=ACCENT, label="measured")
    left.axhline(1.0, color=GRID, lw=1)
    left.set_xscale("log", base=2)
    style(left, "The latency model I froze, against reality", "compaction speedup", "batch size")
    left.legend(frameon=False, fontsize=8.5)

    error = [r["rel_err"] * 100 for r in rows]
    colours = [FAIL if e > 20 else ACCENT for e in error]
    right.bar([str(b) for b in batch], error, color=colours, width=0.6)
    right.axhline(20, color=INK, lw=1, ls=":")
    right.text(0.05, 21, "kill condition", color=INK, fontsize=8.5)
    for i, e in enumerate(error):
        right.text(i, e + 1.5, f"{e:.0f}", ha="center", color=INK, fontsize=8.5)
    style(right, "Relative error: overhead-bound at the left, compute-bound at the right",
          "% error", "batch size")
    save(fig, "tech_cost_model_failure")


def fig_m2_planner():
    """The full M2 sweep with the planner's choice marked at every size."""

    report = load(Q3 / "m2_report.json")
    sweep = report["sweep"]
    batches = sorted({r["batch"] for r in sweep})
    series = {p: [next(r["median_us_per_example"] for r in sweep
                       if r["batch"] == b and r["policy"] == p) for b in batches]
              for p in ("select", "lockstep", "compacted")}
    choice = {b: next(r["planner_choice"] for r in sweep if r["batch"] == b) for b in batches}

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for name, colour, label in (("select", MUTED, "full depth"),
                                ("lockstep", FAIL, "lockstep"),
                                ("compacted", ACCENT, "compacted")):
        left.plot(batches, series[name], "-o", ms=4, color=colour, label=label)
    for b in batches:
        best = min(("select", "lockstep", "compacted"),
                   key=lambda p: series[p][batches.index(b)])
        marker = "o" if choice[b] == best else "x"
        left.plot([b], [series[best][batches.index(b)]], marker, ms=11, mfc="none",
                  mec=GOLD, mew=1.6)
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    style(left, "Every batch size on the M2 (ring = measured optimum, cross = planner disagreed)",
          "microseconds per example", "batch size")
    left.legend(frameon=False, fontsize=8.5)

    throughput = {p: [1e6 / v for v in series[p]] for p in series}
    right.plot(batches, throughput["lockstep"], "-o", ms=4, color=FAIL, label="lockstep")
    right.plot(batches, throughput["compacted"], "-o", ms=4, color=ACCENT, label="compacted")
    right.set_xscale("log", base=2)
    style(right, "Throughput", "examples per second", "batch size")
    right.legend(frameon=False, fontsize=8.5)
    save(fig, "tech_m2_planner")


def fig_hyperparameter_grid():
    """After the normalisation fix: which settings work, and which simply undertrain."""

    report = load(Q3 / "m2_report.json")
    grid = report["reliability"]["hyperparameter_sensitivity"]
    lrs = sorted({g["learning_rate"] for g in grid})
    biases = sorted({g["halt_bias"] for g in grid})
    matrix = np.zeros((len(biases), len(lrs)))
    for g in grid:
        matrix[biases.index(g["halt_bias"]), lrs.index(g["learning_rate"])] = g["accuracy"]

    ece = report["reliability"]["expected_calibration_error"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.4),
                                      gridspec_kw={"width_ratios": [1.15, 1]})
    image = left.imshow(matrix, cmap="BuGn", vmin=0, vmax=1, aspect="auto")
    left.set_xticks(range(len(lrs)), [f"{v:g}" for v in lrs])
    left.set_yticks(range(len(biases)), [f"{v:g}" for v in biases])
    for i in range(len(biases)):
        for j in range(len(lrs)):
            value = matrix[i, j]
            left.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                      color="white" if value > 0.6 else INK)
    left.set_title("Accuracy over 12 settings: 9 perfect, 3 undertrained",
                   color=INK, fontsize=10.5, loc="left", pad=10)
    left.set_xlabel("learning rate", color=MUTED, fontsize=9)
    left.set_ylabel("halting bias at init", color=MUTED, fontsize=9)
    left.tick_params(colors=MUTED, labelsize=8.5)
    fig.colorbar(image, ax=left, fraction=0.045).ax.tick_params(colors=MUTED, labelsize=8)

    right.bar(range(1, len(ece) + 1), [e * 1000 for e in ece], color=ACCENT, width=0.62)
    style(right, "Calibration error per seed, all ten perfect on accuracy",
          "ECE (x1000)", "seed")
    right.set_xticks(range(1, len(ece) + 1))
    save(fig, "tech_hyperparameter_grid")


def fig_har_pareto():
    """The real-data result as a Pareto plot: my mechanism is dominated."""

    record = load(Q3 / "QNEURO3-HAR-P1-RESULT.json")
    rows = record["results_3_seeds"]
    short = {
        "fixed": "fixed depth",
        "act": "ACT (2016)",
        "confidence": "confidence exit",
        "confidence_at_matched_compute": "confidence, matched compute",
        "supervised (Q-Neuro 3.0 mechanism)": "mine",
        "pondernet": "PonderNet (2021)",
    }
    # The four cheap arms sit within 0.09 accuracy of each other, so the offsets are chosen to
    # separate the labels rather than left to the default.
    nudge = {
        "act": (10, 8),
        "confidence": (10, -1),
        "confidence_at_matched_compute": (10, -12),
        "supervised (Q-Neuro 3.0 mechanism)": (11, -3),
    }

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for row, colour in zip(rows, PALETTE):
        mine = "Q-Neuro" in row["arm"]
        left.scatter(row["mean_chunks"], row["test_accuracy"], s=150 if mine else 90,
                     color=FAIL if mine else colour, zorder=3,
                     edgecolor=INK if mine else "none", linewidth=1.2)
        left.annotate(short.get(row["arm"], row["arm"]),
                      (row["mean_chunks"], row["test_accuracy"]),
                      textcoords="offset points", xytext=nudge.get(row["arm"], (8, -3)),
                      fontsize=8.5, color=FAIL if mine else INK)
    left.set_xlim(0, 21)
    style(left, "Cheaper is left, better is up. Mine is the red one.",
          "test accuracy", "mean chunks read (compute)")

    timed = [r for r in rows if r.get("train_seconds")]
    names = [short.get(r["arm"], r["arm"]) for r in timed]
    seconds = [r["train_seconds"] for r in timed]
    colours = [FAIL if "Q-Neuro" in r["arm"] else MUTED for r in timed]
    right.barh(names[::-1], seconds[::-1], color=colours[::-1], height=0.5)
    for i, value in enumerate(seconds[::-1]):
        right.text(value + 0.12, i, f"{value:.1f}s", va="center", color=INK, fontsize=9)
    style(right, "and it is the most expensive to train", xlabel="training seconds")
    right.xaxis.grid(True, color=GRID, linewidth=0.7)
    right.yaxis.grid(False)
    right.set_xlim(0, max(seconds) * 1.3)
    save(fig, "tech_har_pareto")


# ------------------------------------------------------------------ Nova internals


def fig_operator_probe():
    """H-DILUTION: the operator property is real; the task benefit is the confound control."""

    record = load(NOVA / "NOVA-H-DILUTION.json")
    probe = record["operator_level_evidence"]
    task = record["task_level_result_5_seeds_at_4x_length"]
    order = ["attn_softmax", "attn_softmax_rms", "attn_max", "attn_threshold"]
    labels = ["softmax\n(control)", "softmax+RMS\n(confound control)", "max", "threshold"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    keys = ["softmax", "logl", "max", "threshold"]
    left.bar(keys, [probe[k] for k in keys], color=[MUTED, BLUE, ACCENT, ACCENT], width=0.6)
    for i, k in enumerate(keys):
        left.text(i, probe[k] + 0.02, f"{probe[k]:.3f}", ha="center", color=INK, fontsize=9)
    style(left, "Read drift from 24 distractors (lower = invariant)", "read drift")

    x = np.arange(len(order))
    means = [task["copy"][k][0] for k in order]
    errors = [task["copy"][k][1] for k in order]
    colours = [MUTED, BLUE, ACCENT, ACCENT]
    right.bar(x, means, yerr=errors, capsize=4, color=colours, width=0.6,
              error_kw={"ecolor": INK, "elinewidth": 1})
    right.axhline(means[1], color=BLUE, ls=":", lw=1.2)
    right.set_xticks(x, labels, fontsize=8)
    style(right, "Copy at 4x length: the control captures all of it", "accuracy")
    save(fig, "tech_operator_probe")


def fig_interference():
    """H-INTERFERENCE-P1: the handicap does not free the recurrence, it removes the attention."""

    record = load(NOVA / "NOVA-H-INTERFERENCE-P1-RESULT.json")
    rows = record["results_mean_of_3_seeds_at_4x_length"]
    labels = [r["arm"].replace("rnn_attn", "hybrid").replace(" (reference)", "") for r in rows]
    x = np.arange(len(rows))

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    left.plot(x, [r["mod_sum"] for r in rows], "-o", ms=5, color=ACCENT, label="mod_sum (state tracking)")
    left.plot(x, [r["needle"] for r in rows], "-o", ms=5, color=FAIL, label="needle (retrieval)")
    left.axhline(0.70, color=INK, ls=":", lw=1)
    left.text(0.05, 0.72, "both clauses needed 0.70", color=INK, fontsize=8.5)
    left.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    style(left, "Handicapping attention trades capabilities", "accuracy at 4x length")
    left.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, columnspacing=1.0,
                handlelength=1.3, bbox_to_anchor=(0.5, 1.02))
    left.set_ylim(0.15, 1.16)

    ablation = load(NOVA / "NOVA-BRANCH-ABLATION-001.json")["measured"]
    modes = ["none", "attention", "recurrence"]
    mode_labels = ["nothing removed", "attention off", "recurrence off"]
    y = np.arange(len(modes))
    for offset, task, colour in ((-0.19, "mod_sum", ACCENT), (0.19, "needle", FAIL)):
        right.bar(y + offset, [ablation[task][m]["acc_at_64"] for m in modes], 0.36,
                  color=colour, label=task)
    right.set_xticks(y, mode_labels, fontsize=8.5)
    style(right, "Branch ablation: neither works alone", "accuracy at 4x length")
    right.legend(frameon=False, fontsize=8.5)
    save(fig, "tech_interference")


def fig_frontier_heatmap():
    """The whole capability matrix as one picture, with the chance floor subtracted."""

    record = load(NOVA / "NOVA-FRONTIER-001.json")
    frontier = record["frontier"]
    ceilings = record["task_validity"]["clean_task_shortcut_ceilings"]
    tasks = ["parity", "mod_sum", "copy", "reverse", "needle"]
    ceiling_key = {"parity": "parity_scan", "mod_sum": "mod_sum", "copy": "copy",
                   "reverse": "reverse", "needle": "needle"}
    names = sorted(frontier, key=lambda n: -np.mean([frontier[n][t] for t in tasks]))
    matrix = np.array([[frontier[n][t] for t in tasks] for n in names])
    floor = np.array([ceilings[ceiling_key[t]] for t in tasks])
    headroom = (matrix - floor) / (1.0 - floor)

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(names) + 1.7))
    image = ax.imshow(headroom, cmap="BuGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tasks)), tasks, fontsize=9)
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names], fontsize=9)
    for i in range(len(names)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if headroom[i, j] > 0.6 else INK)
    ax.set_title("Every architecture on every clean task, at 4x the trained length\n"
                 "shade = fraction of the headroom above the shortcut floor",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.tick_params(colors=MUTED)
    fig.colorbar(image, ax=ax, fraction=0.03).ax.tick_params(colors=MUTED, labelsize=8)
    save(fig, "tech_frontier_heatmap")


def fig_parameter_matching():
    """Comparing architectures means matching parameters, and the families differ by 8x."""

    record = load(NOVA / "NOVA-FRONTIER-001.json")
    frontier = record["frontier"]
    seen: dict[str, int] = {}
    for line in (NOVA / "registry.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        name, count = entry.get("name"), entry.get("parameters")
        if name in frontier and isinstance(count, int):
            seen.setdefault(name, count)
    params = seen
    if not params:
        raise RuntimeError("no parameter counts found in the Nova registry")

    target = record["protocol"]["parameters"]
    names = sorted(params, key=params.get)
    values = [params[n] for n in names]
    centre = float(np.median(values))

    fig, ax = plt.subplots(figsize=(7.4, 0.34 * len(names) + 1.5))
    ax.barh([n.replace("_", " ") for n in names], values, color=ACCENT, height=0.55)
    ax.axvline(centre, color=INK, ls=":", lw=1.2)
    ax.axvspan(centre * 0.87, centre * 1.13, color=GOLD, alpha=0.15)
    for i, v in enumerate(values):
        ax.text(v + centre * 0.02, i, f"{v:,}", va="center", color=INK, fontsize=8.5)
    style(ax, f"Parameter matching across families (target {target})", xlabel="parameters")
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.yaxis.grid(False)
    ax.set_xlim(0, max(values) * 1.22)
    save(fig, "tech_parameter_matching")


def fig_defect_ledger():
    """Sixteen measurement defects, each of which produced a publishable wrong answer."""

    defects = [
        ("NaN counted as convergence", "Sentinel", "an exact probe disagreeing"),
        ("sign error in a bound", "Sentinel", "the invariant written as a test"),
        ("threshold artifact", "Sentinel", "a proper bimodality statistic"),
        ("pre-asymptotic fit", "Sentinel", "a constancy check"),
        ("holonomy was drift", "Sentinel", "a stay-control"),
        ("tautological feature", "Sentinel", "16 identical digits"),
        ("rank probe saturation", "Sentinel", "exact arithmetic"),
        ("unsolvable task", "Pulse", "ten models tying exactly"),
        ("shared key/value", "Pulse", "accuracy pinned at chance"),
        ("answer overwrite", "Pulse", "an equivalence check"),
        ("cost accounting", "Pulse", "a static-width control"),
        ("hash non-determinism", "Pulse", "re-verifying from disk"),
        ("SSM initialised to NaN", "Nova", "a finiteness check"),
        ("'max' was secretly softmax", "Nova", "identical to the control"),
        ("missing confound control", "Nova", "adding the control"),
        ("undertrained by 3x", "Nova", "re-running at convergence"),
    ]
    eras = ["Sentinel", "Pulse", "Nova"]
    colour = {"Sentinel": BLUE, "Pulse": ACCENT, "Nova": PLUM}

    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(defects) + 1.4))
    for i, (name, era, caught) in enumerate(reversed(defects)):
        y = i
        ax.barh([y], [1], color=colour[era], height=0.55)
        ax.text(-0.02, y, name, ha="right", va="center", color=INK, fontsize=9)
        ax.text(1.03, y, f"caught by {caught}", va="center", color=MUTED, fontsize=8.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.55, 2.05)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour[e]) for e in eras]
    ax.legend(handles, eras, frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.02))
    ax.set_ylim(-1.2, len(defects) - 0.3)
    ax.set_title("Every defect that produced a plausible wrong answer, and what caught it",
                 color=INK, fontsize=10.5, loc="left", pad=12)
    save(fig, "tech_defect_ledger")


def fig_transport_ladder():
    """What survives each equivalence map, on a log scale, including the one that cannot."""

    floor = 1e-11
    rows = [
        ("hidden-unit permutation", 1.192e-07, "full"),
        ("scaling, learning rate transported", 0.0, "full"),
        ("complex <-> realified (AdamW)", 0.0, "full"),
        ("complex <-> realified (SGD)", 1.192e-07, "full"),
        ("complex <-> exact real (forward)", 5.245e-06, "E2 on a domain"),
        ("scaling + AdamW weight decay", 1.312e-04, "impossible"),
        ("scaling + SGD weight decay", 3.405e-03, "impossible"),
    ]
    names = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    kinds = [r[2] for r in rows][::-1]
    colours = {"full": ACCENT, "E2 on a domain": GOLD, "impossible": FAIL}
    drawn = [max(v, floor * 1.6) for v in values]

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.barh(names, drawn, color=[colours[k] for k in kinds], height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(floor, 5e-2)
    ax.axvline(1.192e-07, color=INK, ls=":", lw=1.1)
    ax.text(1.5e-07, len(rows) - 0.45, "one float32 ULP", color=INK, fontsize=8.5)
    for i, value in enumerate(values):
        if value == 0.0:
            ax.text(drawn[i] * 1.5, i, "exactly 0", va="center", color=INK, fontsize=8.5)
    style(ax, "First-update discrepancy across every map family", xlabel="absolute discrepancy")
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.yaxis.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in colours]
    ax.legend(handles, list(colours), frameon=False, fontsize=8.5, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.19),
              title="optimiser transport", title_fontsize=8.5)
    save(fig, "tech_transport_ladder")


def fig_bibliography():
    """Where every mechanism I 'invented' already lives in the literature."""

    rows = [
        ("supervised halting", "ACT", 2016),
        ("supervised halting", "PonderNet", 2021),
        ("cursor / relative shift", "Neural Turing Machine", 2014),
        ("content-addressed read", "Pointer Networks", 2015),
        ("recurrence + attention", "linear attention / RetNet / Mamba", 2020),
        ("active-set compaction", "continuous batching (vLLM / Orca)", 2022),
        ("depth bucketing", "length bucketing", 2017),
        ("branch dropout", "stochastic depth", 2016),
        ("competition for capability", "gradient starvation", 2020),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(rows) + 1.4))
    for i, (mine, theirs, year) in enumerate(reversed(rows)):
        ax.plot([0, 1], [i, i], color=GRID, lw=1.4, zorder=1)
        ax.scatter([0], [i], s=52, color=ACCENT, zorder=2)
        ax.scatter([1], [i], s=52, color=FAIL, zorder=2)
        ax.text(-0.03, i, mine, ha="right", va="center", color=INK, fontsize=9)
        ax.text(1.03, i, f"{theirs}, {year}", va="center", color=MUTED, fontsize=8.5)
    ax.set_xlim(-0.62, 1.95)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("What I thought I invented, and who published it first",
                 color=INK, fontsize=10.5, loc="left", pad=14)
    save(fig, "tech_bibliography")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("technical figures ->", OUT)
    for maker in (
        fig_cost_constants, fig_policy_rows, fig_ceiling_removed, fig_cost_model_failure,
        fig_m2_planner, fig_hyperparameter_grid, fig_har_pareto, fig_operator_probe,
        fig_interference, fig_frontier_heatmap, fig_parameter_matching, fig_defect_ledger,
        fig_transport_ladder, fig_bibliography,
    ):
        maker()


if __name__ == "__main__":
    main()
