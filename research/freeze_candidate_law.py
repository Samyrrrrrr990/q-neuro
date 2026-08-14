"""Freeze one provisional discovery law and the untouched confirmation protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from experiments.run_experiment_zero import ROOT
from qneuro.provenance import file_sha256, require_clean_worktree
from research.computational_laws import FrozenLaw, freeze_best_candidate, frozen_law_from_dict


def _load_candidates(value: dict[str, Any]) -> dict[str, FrozenLaw]:
    candidates = value.get("candidate_laws")
    if not isinstance(candidates, dict):
        raise TypeError("discovery artifact has no candidate_laws mapping")
    return {name: frozen_law_from_dict(item) for name, item in candidates.items()}


def build_artifact(
    discovery: dict[str, Any],
    *,
    discovery_path: Path,
    discovery_sha256: str,
    source_environment: dict[str, Any],
    confirmation_config: dict[str, Any],
    confirmation_config_path: Path,
    confirmation_config_sha256: str,
    code_hashes: dict[str, str],
    frozen_at: str,
    freeze_commit: str,
) -> dict[str, Any]:
    """Validate discovery/confirmation separation and return the immutable law artifact."""

    if discovery.get("status") != "complete" or discovery.get("split") != "discovery":
        raise ValueError("candidate source must be a complete discovery artifact")
    if source_environment.get("git_dirty") is not False:
        raise ValueError("candidate source environment was not clean")
    if confirmation_config.get("split") != "confirmation":
        raise ValueError("confirmation config must declare split: confirmation")
    discovery_families = {str(cell["family"]) for cell in discovery["law_cells"]}
    confirmation_families = set(confirmation_config["families"])
    overlap = discovery_families & confirmation_families
    if overlap:
        raise ValueError(f"discovery/confirmation family overlap: {sorted(overlap)}")
    frozen = freeze_best_candidate(_load_candidates(discovery))
    effects = [float(item["difference"]) for item in discovery["paired_effects"]]
    thresholds = confirmation_config["law_thresholds"]
    return {
        "schema_version": "1.0.0",
        "candidate_id": "QN-LAW-001",
        "status": "frozen_provisional",
        "frozen_at": frozen_at,
        "freeze_commit": freeze_commit,
        "target": "complex_top1_minus_cellwise_best_real_top1",
        "predictors": ["empirical_observed_order_target_mutual_information", "shift_severity"],
        "selection_rule": (
            "Minimum BIC proxy over the six prespecified candidate families, with complexity and "
            "family-name tie breaks implemented by research.computational_laws.freeze_best_candidate."
        ),
        "law": frozen.to_dict(),
        "source": {
            "experiment_id": discovery["experiment_id"],
            "path": discovery_path.as_posix(),
            "sha256": discovery_sha256,
            "source_git_commit": source_environment["git_commit"],
            "profile": discovery["profile"],
            "outcome_eligible": bool(discovery["outcome_eligible"]),
            "general_law_claim_permitted": bool(discovery["general_law_claim_permitted"]),
            "law_cells": len(discovery["law_cells"]),
            "paired_effects": len(effects),
            "positive_effect_cells": sum(value > 0.0 for value in effects),
            "mean_effect": sum(effects) / len(effects),
            "families": sorted(discovery_families),
        },
        "confirmation": {
            "config_path": confirmation_config_path.as_posix(),
            "config_sha256": confirmation_config_sha256,
            "families": sorted(confirmation_families),
            "training_seeds": list(confirmation_config["training"]["seeds"]),
            "world_seeds": list(confirmation_config["dataset"]["world_seeds"]),
            "thresholds": {
                "minimum_r2": float(thresholds["minimum_r2"]),
                "minimum_sign_accuracy": float(thresholds["minimum_sign_accuracy"]),
                "maximum_mae": float(thresholds["maximum_mae"]),
            },
            "code_sha256": code_hashes,
        },
        "scope": {
            "synthetic_nonclinical_only": True,
            "provisional_confirmation_only": True,
            "outcome_e_permitted": False,
            "qn_grand_001_permitted": False,
            "reason": (
                "QN-000040 is a reduced, outcome-ineligible discovery profile and all observed "
                "complex-minus-best-real effects were non-positive."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery",
        type=Path,
        default=ROOT / "experiments" / "results" / "QN-000040" / "metrics.json",
    )
    parser.add_argument(
        "--confirmation-config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "independent_confirmation.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "laws" / "FROZEN_CANDIDATE_001.json",
    )
    args = parser.parse_args()
    require_clean_worktree(ROOT)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen law: {args.output}")
    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    source_environment = json.loads(
        (args.discovery.parent / "environment.json").read_text(encoding="utf-8")
    )
    confirmation_config = yaml.safe_load(args.confirmation_config.read_text(encoding="utf-8"))
    code_paths = [
        ROOT / "research" / "computational_laws.py",
        ROOT / "experiments" / "run_independent_discovery.py",
        ROOT / "experiments" / "run_independent_confirmation.py",
        ROOT / "independent_tasks" / "generators.py",
    ]
    artifact = build_artifact(
        discovery,
        discovery_path=args.discovery.relative_to(ROOT),
        discovery_sha256=file_sha256(args.discovery),
        source_environment=source_environment,
        confirmation_config=confirmation_config,
        confirmation_config_path=args.confirmation_config.relative_to(ROOT),
        confirmation_config_sha256=file_sha256(args.confirmation_config),
        code_hashes={path.relative_to(ROOT).as_posix(): file_sha256(path) for path in code_paths},
        frozen_at=datetime.now(UTC).isoformat(),
        freeze_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_id": artifact["candidate_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
