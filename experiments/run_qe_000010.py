"""QE-000010: estimator freeze record — gate-enforced.

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5.

This experiment exists to freeze one primary estimator before any sealed confirmation family is
opened. It **refuses to freeze** unless Gate D passed, and it writes the refusal as a first-class
artifact rather than exiting quietly.

That refusal is the point. A freeze record that can be produced regardless of the evidence is not a
freeze record, and the whole preregistration depends on this step being unable to rubber-stamp a
candidate that did not earn it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata

EXPERIMENT_ID = "QE-000010"
SCHEMA_VERSION = "1.0.0"
SOURCE_EXPERIMENT = "QE-000009"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "primary_estimator_freeze_record",
    "description": (
        "Freezes one primary transport-defect estimator, but only if Gate D passed. Otherwise "
        "records the refusal and leaves every sealed confirmation family closed."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "freeze",
    "outcome_eligible": False,
    "gate": "D",
    "source_experiment": SOURCE_EXPERIMENT,
}


def run(config: dict[str, Any]) -> dict[str, Any]:
    source_path = ROOT / "experiments" / "results" / SOURCE_EXPERIMENT / "metrics.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"{SOURCE_EXPERIMENT} has not been run; nothing to freeze from")
    source = json.loads(source_path.read_text(encoding="utf-8"))

    gate_d = source["gate_d"]
    passes = bool(gate_d["any_candidate_passes"])

    evaluation = source["evaluation"]
    diagnosis = {
        "target_spread_log10": evaluation.get("target_spread_log10", {}),
        "families_overlap_in_scale": False,
        "within_family_wins": gate_d.get("within_family_per_candidate", {}),
        "out_of_family_wins": {
            name: entry["families_beating_all_baselines"]
            for name, entry in gate_d["per_candidate"].items()
        },
    }
    # The families do not form one population on the target scale. Rather than a single overlap
    # boolean, report the clustering: sort by range and find the widest gap between consecutive
    # families. That gap is what a global log-log calibration would have to span.
    spread = diagnosis["target_spread_log10"]
    if spread:
        ordered = sorted(spread.items(), key=lambda item: item[1]["min"])
        gaps = [
            {
                "below": ordered[index][0],
                "above": ordered[index + 1][0],
                "gap_orders_of_magnitude": ordered[index + 1][1]["min"] - ordered[index][1]["max"],
            }
            for index in range(len(ordered) - 1)
        ]
        widest = max(gaps, key=lambda entry: entry["gap_orders_of_magnitude"])
        medians = [entry["median"] for entry in spread.values()]
        diagnosis["family_order_by_scale"] = [name for name, _ in ordered]
        diagnosis["consecutive_gaps"] = gaps
        diagnosis["widest_gap"] = widest
        diagnosis["family_median_spread_orders_of_magnitude"] = max(medians) - min(medians)
        diagnosis["families_overlap_in_scale"] = all(
            entry["gap_orders_of_magnitude"] <= 0.0 for entry in gaps
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "outcome_eligible": False,
        "status": "refused" if not passes else "frozen",
        "gate": "D",
        "gate_d_passed": passes,
        "frozen_estimator": None,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_sha256_note": "see release/manifest.json once QE artifacts are added to the manifest",
        "diagnosis": diagnosis,
        "decision_rule": (
            "No estimator may be frozen, and no sealed confirmation family may be opened, unless a "
            "candidate beat every section 6.4 baseline on at least two discovery families "
            "out-of-family."
        ),
        "decision": (
            "REFUSED. Gate D did not pass. No primary estimator is frozen. Rungs 5 to 8 remain "
            "sealed and QE-000012 may not run."
            if not passes
            else "Frozen."
        ),
        "scientific_interpretation": (
            "The candidate accumulated-defect estimators carry real within-family signal — "
            "cumulative_defect reaches within-family R-squared 0.962 on factorization and 0.812 on "
            "the scaling orbit, beating every baseline on both — but they do not transfer across "
            "families. Family medians on the target scale differ by about 6.5 orders of magnitude, "
            "from roughly 1e-7 for permutation, where the map is conjugate and there is almost "
            "nothing left to predict, to roughly 1e-0.6 for the scaling orbit. The ranges chain "
            "rather than separating cleanly, so the problem is not a single gap: it is that one "
            "global slope and intercept are forced onto families whose own intercepts differ by "
            "orders of magnitude, which makes every out-of-family fit worse than predicting the "
            "mean. This is a calibration failure, not an absence of signal, and it is recorded as "
            "a negative result under section 8 rather than retried with different features."
        ),
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
    (directory / "decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"{EXPERIMENT_ID}: status={result['status']}")
    print(f"  Gate D passed        : {result['gate_d_passed']}")
    print(f"  frozen estimator     : {result['frozen_estimator']}")
    diagnosis = result["diagnosis"]
    print(
        f"  family median spread : "
        f"{diagnosis['family_median_spread_orders_of_magnitude']:.2f} orders of magnitude"
    )
    print(f"  family order by scale: {diagnosis['family_order_by_scale']}")
    print(f"  {result['decision']}")


if __name__ == "__main__":
    main()
