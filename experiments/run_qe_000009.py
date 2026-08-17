"""QE-000009: candidate defect estimators versus the section 6.4 baselines (Gate D).

Registered by `docs/ML2_PREREGISTRATION_001.md` section 5. Discovery split.

Generates transport traces across the discovery families, then scores every candidate estimator and
every baseline by leave-one-family-out held-out R-squared in log space. Gate D requires a candidate
to beat *every* baseline on at least two families.

If no candidate clears it, that is a result, not a failure to be retried with different features.
Section 8 kill condition 1 and 2 apply, and the outcome is recorded as-is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.equivalence import (
    FactorizedToDenseMap,
    HiddenUnitPermutationMap,
    HomogeneousScalingMap,
)
from qneuro.equivalence.estimators import (
    BASELINES,
    CANDIDATES,
    evaluate_leave_one_family_out,
    gate_d_verdict,
)
from qneuro.equivalence.factorization import (
    DenseLinear,
    FactorizedLinear,
    align_factorized_to_dense,
)
from qneuro.equivalence.microcosms import HomogeneousMLP, TwoLayerMLP
from qneuro.equivalence.native_complex import ComplexRealificationMap, NativeComplexMLP
from qneuro.equivalence.trace import paired_transport_trace

EXPERIMENT_ID = "QE-000009"
SCHEMA_VERSION = "1.0.0"

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": "estimator_discovery_against_baselines",
    "description": (
        "Leave-one-family-out comparison of candidate transport-defect estimators against the "
        "preregistered baselines, over the discovery families."
    ),
    "preregistration_id": "ML2-PREREG-001",
    "preregistration_version": "1.0.0",
    "preregistration_document": "docs/ML2_PREREGISTRATION_001.md",
    "profile": "discovery",
    "split": "discovery",
    "outcome_eligible": False,
    "gate": "D",
    "candidates": list(CANDIDATES),
    "baselines": list(BASELINES),
    "training": {
        "optimizers": ["sgd", "adamw"],
        "learning_rates": [0.001, 0.01],
        "weight_decays": [0.0, 0.01],
        "steps": 40,
        "device": "cpu",
    },
    "seeds": {"model": [0, 1, 2], "stream": 21},
    "protocol_deviations": [
        (
            "Analytic microcosms only. Estimator behaviour on realistic models and tasks is "
            "unmeasured."
        ),
        (
            "Univariate log-log fits by design; a richer model would fit discovery better and say "
            "less about generalization, which is the only question Gate D asks."
        ),
    ],
}


def _pairs(model_seed: int) -> list[tuple[str, Any, Any, Any]]:
    """One representative pair per discovery family, at a fixed model seed."""

    built: list[tuple[str, Any, Any, Any]] = []

    torch.manual_seed(model_seed)
    permutation_source = TwoLayerMLP(6, 12, 4)
    permutation_map = HiddenUnitPermutationMap.random(permutation_source, seed=model_seed + 1)
    built.append(
        (
            "permutation",
            permutation_source,
            permutation_map.build_target(permutation_source),
            permutation_map,
        )
    )

    for scale in (2.0, 4.0):
        torch.manual_seed(model_seed)
        scaling_source = HomogeneousMLP(6, 12, 4)
        scaling_map = HomogeneousScalingMap(scale)
        built.append(
            (
                "scaling_orbit",
                scaling_source,
                scaling_map.build_target(scaling_source),
                scaling_map,
            )
        )

    torch.manual_seed(model_seed)
    complex_source = NativeComplexMLP(6, 12, 4)
    complex_map = ComplexRealificationMap()
    built.append(
        (
            "native_complex",
            complex_source,
            complex_map.build_target(complex_source),
            complex_map,
        )
    )

    torch.manual_seed(model_seed)
    dense = DenseLinear(6, 4)
    factorized = FactorizedLinear(6, 4, rank=4)
    align_factorized_to_dense(dense, factorized)
    built.append(("factorization", factorized, dense, FactorizedToDenseMap()))

    return built


def run(config: dict[str, Any]) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True, warn_only=True)
    training = config["training"]

    rows: list[dict[str, Any]] = []
    for model_seed in config["seeds"]["model"]:
        for family, source, target, mapping in _pairs(int(model_seed)):
            for optimizer_name in training["optimizers"]:
                for learning_rate in training["learning_rates"]:
                    for weight_decay in training["weight_decays"]:
                        for transported in (True, False):
                            if transported and not mapping.supports_optimizer_transport:
                                continue
                            trace = paired_transport_trace(
                                source,
                                target,
                                mapping,
                                optimizer_name=optimizer_name,
                                steps=int(training["steps"]),
                                transport_optimizer_state=transported,
                                seed=int(config["seeds"]["stream"]),
                                learning_rate=float(learning_rate),
                                weight_decay=float(weight_decay),
                            )
                            rows.append(
                                {
                                    "family": family,
                                    "map": mapping.name,
                                    "model_seed": int(model_seed),
                                    "optimizer": optimizer_name,
                                    "weight_decay": float(weight_decay),
                                    "optimizer_state_transported": transported,
                                    **{
                                        key: trace[key]
                                        for key in (
                                            "final_divergence",
                                            "one_step_defect",
                                            "cumulative_defect",
                                            "amplified_defect",
                                            "mean_amplification",
                                            "one_step_predictive_divergence",
                                            "total_gradient_norm",
                                            "loss_decrease",
                                            "learning_rate",
                                            "parameter_count",
                                        )
                                    },
                                }
                            )

    evaluation = evaluate_leave_one_family_out(rows, [*CANDIDATES, *BASELINES])
    verdict = gate_d_verdict(evaluation)

    ranked = sorted(
        (
            (name, entry["mean_r2"])
            for name, entry in evaluation["features"].items()
            if entry["mean_r2"] == entry["mean_r2"]
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "profile": config["profile"],
        "split": config["split"],
        "outcome_eligible": False,
        "status": "complete",
        "gate": "D",
        "rows": rows,
        "evaluation": evaluation,
        "gate_d": verdict,
        "ranking_by_mean_heldout_r2": [
            {"feature": name, "mean_r2": value} for name, value in ranked
        ],
        "summary": {
            "rows": len(rows),
            "families": evaluation["families"],
            "best_feature": ranked[0][0] if ranked else None,
            "best_mean_r2": ranked[0][1] if ranked else float("nan"),
            "gate_d_passes": verdict["any_candidate_passes"],
            "passing_candidates": verdict["passing_candidates"],
        },
        "scientific_interpretation": (
            "Discovery evidence on analytic microcosms. Gate D asks whether an accumulated "
            "transport-defect quantity predicts the final predictive gap out of family better than "
            "every preregistered baseline, including raw one-step predictive divergence. The "
            "verdict recorded here is the measured one. A failure is a kill-condition outcome under "
            "section 8 and is reported rather than retried with different features."
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
    print(f"{EXPERIMENT_ID}: {summary['rows']} rows over {len(summary['families'])} families")
    print("  leave-one-family-out mean held-out R^2:")
    for entry in result["ranking_by_mean_heldout_r2"]:
        marker = "candidate" if entry["feature"] in CANDIDATES else "baseline "
        print(f"    {marker}  {entry['feature']:<34} {entry['mean_r2']:+.3f}")
    print("  within-family R^2 (diagnostic, NOT the gate):")
    for name, entry in result["evaluation"]["features"].items():
        marker = "candidate" if name in CANDIDATES else "baseline "
        cells = entry.get("within_family", {})
        rendered = "  ".join(f"{family[:9]}={cells[family]['r2']:+.3f}" for family in sorted(cells))
        print(f"    {marker}  {name:<34} {rendered}")
    print(f"  GATE D: {'PASS' if summary['gate_d_passes'] else 'FAIL'}")
    for name, entry in result["gate_d"]["per_candidate"].items():
        within = result["gate_d"]["within_family_per_candidate"].get(name, {})
        print(
            f"    {name:<22} out-of-family wins: {entry['count']} {entry['families_beating_all_baselines']}"
            f"  |  within-family wins: {within.get('count', 0)} {within.get('families_beating_all_baselines', [])}"
        )


if __name__ == "__main__":
    main()
