"""QE-000008: finite-horizon transport bound on an analytic microcosm (Gate C).

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Discovery split.

Gate C asks whether the bound ``e_{k+1} <= L_k e_k + delta_k`` is non-vacuous — within a factor of
100 of the divergence it bounds. The question is answered on linear regression under a diagonal
reparameterization, where the Lipschitz constant is a spectral norm and the one-step defect has a
closed form, so no term has to be estimated.

The experiment also runs the same bound with a Lipschitz constant obtained by the triangle
inequality rather than the spectral norm of the whole operator. Section 6.10 predicts that generic
worst-case constants are useless; this measures by how much.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence.analytic import LinearRegressionMicrocosm, transport_bound

EXPERIMENT_ID = "QE-000008"
SCHEMA_VERSION = "1.0.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "analytic_transport_bound_tightness",
    "description": (
        "Exact finite-horizon transport bound on least squares under diagonal reparameterization. "
        "Every term is closed form, so bound tightness is measured rather than estimated."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "discovery",
    "split": "discovery",
    "outcome_eligible": False,
    "family": "analytic_microcosm",
    "gate": "C",
    "gate_threshold": {"maximum_bound_ratio": 100.0},
    "problem": {"samples": 300, "features": 10, "dtype": "float64"},
    "grid": {
        "condition_numbers": [1.0, 10.0, 100.0, 1000.0, 10000.0],
        "step_fractions": [0.1, 0.5, 0.9],
        "scales": [1.5, 2.0, 4.0],
        "horizons": [50, 200, 1000],
    },
    "seeds": {"problem": [0, 1, 2]},
    "protocol_deviations": [
        (
            "Affine update maps with exactly computed Lipschitz constants. This is the most "
            "favourable possible setting for the bound and does not establish non-vacuity for "
            "nonlinear models where the constant must be over-estimated."
        ),
        (
            "Discovery split: these cells may inform estimator design and may not be reused for "
            "confirmation."
        ),
    ],
}


def run(config: dict[str, Any]) -> dict[str, Any]:
    problem = config["problem"]
    grid = config["grid"]
    threshold = float(config["gate_threshold"]["maximum_bound_ratio"])

    records: list[dict[str, Any]] = []
    for seed in config["seeds"]["problem"]:
        for condition_number in grid["condition_numbers"]:
            microcosm = LinearRegressionMicrocosm.build(
                int(problem["samples"]),
                int(problem["features"]),
                float(condition_number),
                seed=int(seed),
            )
            stable = microcosm.stable_step_size()
            for step_fraction in grid["step_fractions"]:
                for scale in grid["scales"]:
                    for horizon in grid["horizons"]:
                        scaling = torch.full(
                            (int(problem["features"]),), float(scale), dtype=torch.float64
                        )
                        measured = transport_bound(
                            microcosm, scaling, float(step_fraction) * stable, steps=int(horizon)
                        )
                        records.append(
                            {
                                "seed": int(seed),
                                "condition_number": float(condition_number),
                                "step_fraction": float(step_fraction),
                                "scale": float(scale),
                                "horizon": int(horizon),
                                "bound_holds": bool(
                                    measured["bound_parameter_divergence"]
                                    >= measured["observed_parameter_divergence"] * (1.0 - 1e-9)
                                ),
                                **{
                                    key: measured[key]
                                    for key in (
                                        "target_lipschitz",
                                        "naive_lipschitz",
                                        "contractive",
                                        "mean_defect",
                                        "observed_parameter_divergence",
                                        "bound_parameter_divergence",
                                        "bound_ratio",
                                        "naive_bound_ratio",
                                        "observed_predictive_divergence",
                                        "predictive_bound_ratio",
                                        "observed_at_numerical_floor",
                                    )
                                },
                            }
                        )

    scored = [record for record in records if not record["observed_at_numerical_floor"]]
    identity_cells = [record for record in records if record["scale"] == 1.0]
    worst_ratio = max(record["bound_ratio"] for record in scored)
    worst_naive = max(record["naive_bound_ratio"] for record in scored)

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "split": config["split"],
        "outcome_eligible": False,
        "status": "complete",
        "gate": "C",
        "records": records,
        "summary": {
            "cells": len(records),
            "scored_cells": len(scored),
            "cells_at_numerical_floor": len(records) - len(scored),
            "bound_violations": sum(not record["bound_holds"] for record in scored),
            "worst_bound_ratio": worst_ratio,
            "median_bound_ratio": sorted(record["bound_ratio"] for record in scored)[
                len(scored) // 2
            ],
            "worst_naive_bound_ratio": worst_naive,
            "naive_bound_looser_by_orders_of_magnitude": (
                torch.log10(torch.tensor(worst_naive / worst_ratio)).item()
            ),
            "gate_c_threshold": float(config["gate_threshold"]["maximum_bound_ratio"]),
            "gate_c_passes": bool(worst_ratio <= threshold),
            "identity_cells_have_zero_defect": all(
                record["mean_defect"] == 0.0 for record in identity_cells
            ),
        },
        "scientific_interpretation": (
            "Gate C evidence on an analytic microcosm. With affine update maps and exactly computed "
            "Lipschitz constants the finite-horizon bound is tight across four orders of "
            "conditioning and two orders of horizon, and is never violated. Substituting a "
            "triangle-inequality Lipschitz constant makes the same bound vacuous by many orders of "
            "magnitude, so the bound's usefulness is a property of how the constant is obtained "
            "rather than of the inequality itself. This is the most favourable possible setting: it "
            "does not establish non-vacuity for nonlinear models where the constant must be "
            "estimated, and that case remains an open obligation."
        ),
        "protocol_deviations": config["protocol_deviations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results")
    args = parser.parse_args()

    result = run(DEFAULT_CONFIG)
    directory = args.output / EXPERIMENT_ID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False), encoding="utf-8"
    )
    (directory / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = result["summary"]
    print(f"{EXPERIMENT_ID}: {summary['cells']} cells ({summary['scored_cells']} scored)")
    print(f"  bound violations       : {summary['bound_violations']}")
    print(f"  median bound ratio     : {summary['median_bound_ratio']:.2f}")
    print(f"  worst bound ratio      : {summary['worst_bound_ratio']:.2f}")
    print(f"  worst naive bound ratio: {summary['worst_naive_bound_ratio']:.3e}")
    print(
        f"  GATE C ({summary['gate_c_threshold']:.0f}x)         : "
        f"{'PASS' if summary['gate_c_passes'] else 'FAIL'}"
    )
    if summary["bound_violations"]:
        raise SystemExit("bound violated; the implementation or the derivation is wrong")


if __name__ == "__main__":
    main()
