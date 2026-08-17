"""Run DISCOVERY-001 and write its Lane B record.

Lane B. Nothing here is a claim. See `docs/LANE_POLICY.md` for the promotion path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from experiments.run_experiment_zero import ROOT, environment_metadata
from research.discovery_lab.equivalence_phase import (
    DISCOVERY_ID,
    PREDICTED_CRITICAL_RHO,
    score,
    sweep,
)

SCHEMA_VERSION = "1.0.0"


def build_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = score(records)
    sgd = [record for record in records if record["optimizer"] == "sgd"]
    false_alarms = [
        record for record in sgd if record["rho"] <= PREDICTED_CRITICAL_RHO and record["diverged"]
    ]
    misses = [
        record
        for record in sgd
        if record["rho"] > PREDICTED_CRITICAL_RHO and not record["diverged"]
    ]
    # A miss exactly at the critical point is not a failed prediction: at rho = 1 the spectral
    # radius is exactly 1, the trajectory is marginally stable, and neither verdict is defined.
    off_boundary_misses = [
        record for record in misses if not math.isclose(record["rho"], 1.0, rel_tol=1e-9)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "discovery_id": DISCOVERY_ID,
        "lane": "B",
        "title": "A sharp, analytically predicted stability boundary in equivalence breaking",
        "status": "replicated; mechanism identified; not yet promoted",
        "observation": (
            "Two models that represent exactly the same predictor at initialization, trained on "
            "identical data with identical optimizer and hyperparameters, land on opposite sides of "
            "a stability boundary purely because of the coordinates they are written in. Across a "
            "1.4 percent change in the scale parameter the paired divergence moves by roughly "
            "fourteen orders of magnitude while the source remains stable throughout."
        ),
        "effect": {
            "control_parameter": "rho = eta * lambda_max(H) / (2 * s^2)",
            "predicted_transition": PREDICTED_CRITICAL_RHO,
            "sgd_prediction_accuracy": summary["sgd"]["prediction_accuracy"],
            "sgd_false_alarms": len(false_alarms),
            "sgd_misses_off_the_critical_point": len(off_boundary_misses),
            "sgd_misses_at_the_critical_point": len(misses) - len(off_boundary_misses),
            "adamw_diverged_cells": summary["adamw"]["diverged_cells"],
            "adamw_cells": summary["adamw"]["cells"],
            "equivalence_broken_cells_sgd": summary["sgd"]["equivalence_broken_cells"],
        },
        "scope": (
            "Least squares under uniform homogeneous scaling, float64, closed-form gradients, "
            "condition numbers 1 to 1000, three seeds, step sizes 0.25 to 0.75 of the stability "
            "threshold, 41 log-spaced scales per configuration. Not tested: nonlinear models, "
            "stochastic minibatches, non-uniform scaling, other optimizers, other dtypes."
        ),
        "competing_explanations": [
            (
                "Grid artifact: refuted, the grid is log-spaced symmetrically about the prediction "
                "and the prediction was derived before the sweep."
            ),
            (
                "Slow convergence mistaken for instability: this was a real defect in the first two "
                "measurement criteria and is why the criterion is now growth of the target's own "
                "parameter norm rather than any paired magnitude."
            ),
            (
                "Float overflow mistaken for convergence: also real, and fixed. A norm can overflow "
                "before any entry does, giving inf/inf = nan, and `nan > threshold` is False, so "
                "runaway runs were silently scored convergent."
            ),
        ],
        "simplest_boring_explanation": (
            "This is textbook gradient-descent stability. Reparameterizing changes the effective "
            "Hessian, hence the effective step size, hence stability. The mechanism is not new and "
            "this record does not claim it is. What the boundary contributes is that equivalence "
            "breaking has an exactly predictable location with a dimensionless control parameter, "
            "and that it supplies a mechanistic account of the Gate D failure: the discovery "
            "families straddle a phase boundary, so they are not one population and no single "
            "calibration can span them."
        ),
        "prior_art": (
            "Gradient-descent stability at eta * lambda_max < 2 is standard. Coordinate dependence "
            "of sharpness and of the effective Hessian is established (Dinh et al. 2017; Kristiadi, "
            "Dangel and Hennig 2023). Novelty is claimed for neither. The integration into an "
            "equivalence-breaking phase boundary is what is under test."
        ),
        "proposed_mechanism": (
            "Under uniform scale s with an untransported learning rate the target's update operator "
            "is I - (eta / s^2) H, so its effective step is eta / s^2 and it is stable exactly when "
            "rho < 1. The source is stable when rho * s^2 < 1, so for s < 1 there is an open window "
            "in which the source converges and its exact equivalent does not."
        ),
        "differential_prediction": (
            "Adam's update is scale free in the gradient, so its effective step does not acquire "
            "the 1/s^2 factor and the boundary should be absent at the same rho. Measured: "
            f"{summary['adamw']['diverged_cells']} of {summary['adamw']['cells']} Adam cells "
            f"diverged, against {summary['sgd']['diverged_cells']} of {summary['sgd']['cells']} "
            "for SGD."
        ),
        "falsifier": (
            "Any cell with rho strictly below 1 whose target diverges while the source is stable, "
            "or a systematic failure of the boundary to move as rho predicts when eta, lambda_max, "
            "or s are varied independently."
        ),
        "replication_status": (
            "Replicated across 3 seeds, 4 condition numbers, 3 step fractions, 2 optimizers, "
            f"{len(records)} cells total. Single implementation; no independent reimplementation."
        ),
        "promotion_stage": "3 of 5 (exploration, internal replication, mechanistic explanation)",
        "preregistration_eligible": True,
        "promotion_blockers": [
            "Step 4 requires a frozen numeric prediction serialized before new evidence exists.",
            (
                "Step 5 requires an untouched confirmatory test in Lane A, on a nonlinear system, "
                "since the present evidence is entirely analytic and linear."
            ),
        ],
        "summary_by_optimizer": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research" / "discovery_lab" / "generated"
    )
    args = parser.parse_args()

    records = sweep()
    record = build_record(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{DISCOVERY_ID}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / f"{DISCOVERY_ID}_environment.json").write_text(
        json.dumps(environment_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    effect = record["effect"]
    print(f"{DISCOVERY_ID}: {len(records)} cells")
    print(f"  control parameter        : {effect['control_parameter']}")
    print(f"  SGD prediction accuracy  : {effect['sgd_prediction_accuracy']:.4f}")
    print(f"  SGD false alarms         : {effect['sgd_false_alarms']}")
    print(f"  SGD misses off-boundary  : {effect['sgd_misses_off_the_critical_point']}")
    print(f"  SGD misses at rho = 1    : {effect['sgd_misses_at_the_critical_point']}")
    print(f"  AdamW diverged           : {effect['adamw_diverged_cells']}/{effect['adamw_cells']}")
    print(f"  promotion stage          : {record['promotion_stage']}")


if __name__ == "__main__":
    main()
