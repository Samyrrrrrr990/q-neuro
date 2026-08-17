"""Evaluate one frozen provisional law on untouched independent task families."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.run_experiment_zero import ROOT
from experiments.run_independent_discovery import run


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a fast integration profile that cannot be interpreted as confirmation."""

    output = copy.deepcopy(config)
    output["description"] += " [SMOKE PROFILE; NOT CONFIRMATION EVIDENCE]"
    output["profile"] = "smoke_confirmation"
    output["families"] = {
        name: output["families"][name]
        for name in ("machine_fault_diagnosis", "hidden_rule_relational")
    }
    output["dataset"].update(
        train_sizes=[120],
        validation_cases=80,
        test_cases_per_world=50,
        counterfactual_pairs_per_world=10,
        world_seeds=[92009, 92033],
    )
    output["training"].update(seeds=[6101], epochs=2, patience=2)
    output["models"]["names"] = [
        "complex_operator",
        "exact_real_block_operator",
        "state_space",
    ]
    output["models"]["learning_rates"] = {
        name: output["models"]["learning_rates"][name] for name in output["models"]["names"]
    }
    output["protocol_deviations"] = [
        *output["protocol_deviations"],
        "Smoke profile is execution-only and is not confirmation evidence.",
    ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "independent_confirmation.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="reserved for smoke debugging")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        config = smoke_config(config)
    experiment_id, directory, result = run(
        config,
        allow_dirty=args.allow_dirty,
        command_module="experiments.run_independent_confirmation",
    )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "result_directory": str(directory),
                "profile": result["profile"],
                "training_runs": len(result["training_runs"]),
                "records": len(result["records"]),
                "law_cells": len(result["law_cells"]),
                "law_confirmation": result["law_confirmation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
