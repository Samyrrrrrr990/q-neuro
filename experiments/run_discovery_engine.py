"""Unify completed candidates, rank Pareto fronts, flag surprises, and propose mutations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from experiments.run_experiment_zero import ROOT, environment_metadata
from qneuro.discovery import detect_surprises, pareto_frontier
from qneuro.registry import ExperimentRegistry


def _mean(container: dict[str, Any], metric: str) -> float:
    return float(container[metric]["mean"])


def architecture_record(
    source: str,
    model: str,
    summary: dict[str, Any],
    catalog: dict[str, Any],
    generation: int,
) -> dict[str, Any]:
    alias = "density_rank2" if model == "density_dynamics" else model
    genome = catalog["architectures"][alias]
    in_domain = summary[model]["in_domain"]
    shifted = summary[model]["moderate"]["across_worlds"]
    return {
        "candidate_id": model,
        "source_experiment": source,
        "context": "architecture",
        "generation": generation,
        "genome": {**genome, "training": "adamw"},
        "in_domain_top1": _mean(in_domain, "top1"),
        "shifted_top1": _mean(shifted, "top1"),
        "shifted_nll": _mean(shifted, "nll"),
        "shifted_ece": _mean(shifted, "ece"),
        "ambiguity_nll": _mean(in_domain, "ambiguity_pair_nll"),
        "counterfactual_pair_accuracy": _mean(in_domain, "counterfactual_pair_accuracy"),
        "training_seconds": _mean(in_domain, "training_seconds"),
        "parameter_count": _mean(in_domain, "parameter_count"),
        "compute_measure": "CPU training seconds",
    }


def training_record(
    source: str,
    method: str,
    summary: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    values = summary[method]["1000"]
    in_domain = values["in_domain"]
    shifted = values["shifted"]["across_worlds"]
    return {
        "candidate_id": f"complex_operator::{method}",
        "source_experiment": source,
        "context": "training_law",
        "generation": 0,
        "genome": {
            **catalog["architectures"]["complex_operator"],
            "training": catalog["training_laws"][method],
        },
        "in_domain_top1": _mean(in_domain, "top1"),
        "shifted_top1": _mean(shifted, "top1"),
        "shifted_nll": _mean(shifted, "nll"),
        "shifted_ece": _mean(shifted, "ece"),
        "ambiguity_nll": _mean(in_domain, "ambiguity_pair_nll"),
        "counterfactual_pair_accuracy": _mean(in_domain, "counterfactual_pair_accuracy"),
        "training_seconds": _mean(in_domain, "training_seconds"),
        "parameter_count": _mean(in_domain, "deploy_parameter_count"),
        "compute_measure": "CPU training seconds",
    }


def halting_record(
    source: str,
    mode: str,
    summary: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    in_domain = summary[mode]["in_domain"]
    shifted = summary[mode]["shifted"]["across_worlds"]
    architecture = "adaptive_attractor_hard2" if mode == "hard" else "adaptive_attractor"
    return {
        "candidate_id": f"adaptive_attractor::{mode}",
        "source_experiment": source,
        "context": "halting",
        "generation": 1 if mode == "hard" else 0,
        "genome": {**catalog["architectures"][architecture], "training": "frozen_checkpoint"},
        "in_domain_top1": _mean(in_domain, "top1"),
        "shifted_top1": _mean(shifted, "top1"),
        "shifted_nll": _mean(shifted, "nll"),
        "shifted_ece": _mean(shifted, "ece"),
        "ambiguity_nll": None,
        "counterfactual_pair_accuracy": 0.0,
        "training_seconds": _mean(in_domain, "latency_ms_per_case") / 1000.0,
        "parameter_count": 19801.0,
        "compute_measure": "CPU inference seconds per case",
    }


def load_json(experiment_id: str) -> dict[str, Any]:
    return json.loads(
        (ROOT / "experiments" / "results" / experiment_id / "metrics.json").read_text(
            encoding="utf-8"
        )
    )


def run(config: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    environment = environment_metadata()
    registry = ExperimentRegistry(ROOT / "experiments" / "registry.sqlite3")
    experiment_id, result_directory = registry.reserve(config, ROOT / "experiments" / "results")
    print(f"{experiment_id}: writing to {result_directory}", flush=True)
    try:
        (result_directory / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (result_directory / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )
        catalog = yaml.safe_load((ROOT / config["catalog"]).read_text(encoding="utf-8"))
        sources = config["sources"]
        architecture = load_json(sources["architecture_suite"])
        ablations = load_json(sources["ablation_suite"])
        training = load_json(sources["training_laws"])
        halting = load_json(sources["hard_halting"])
        records = [
            architecture_record(
                sources["architecture_suite"], model, architecture["summary"], catalog, 0
            )
            for model in architecture["summary"]
        ]
        existing = {record["candidate_id"] for record in records}
        for model in (
            "real_accumulator",
            "complex_accumulator",
            "complex_magnitude_readout",
            "complex_no_negative",
            "density_rank1",
            "density_rank4",
        ):
            if model not in existing:
                records.append(
                    architecture_record(
                        sources["ablation_suite"], model, ablations["summary"], catalog, 1
                    )
                )
        records.extend(
            training_record(sources["training_laws"], method, training["summary"], catalog)
            for method in training["summary"]
        )
        records.extend(
            halting_record(sources["hard_halting"], mode, halting["summary"], catalog)
            for mode in halting["summary"]
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["context"]].append(record)
        frontiers = {
            context: [
                record["candidate_id"]
                for record in pareto_frontier(
                    values,
                    maximize=tuple(config["pareto"]["maximize"]),
                    minimize=tuple(config["pareto"]["minimize"]),
                )
            ]
            for context, values in grouped.items()
        }
        surprises = detect_surprises(records)
        surprises.extend(
            [
                {
                    "type": "hard_halting_degeneracy",
                    "candidate_id": "adaptive_attractor::hard",
                    "severity": 0.75,
                    "message": "All cases stop at two states; realized savings are not adaptive.",
                },
                {
                    "type": "phase_optimizer_no_frontier_gain",
                    "candidate_id": "complex_operator::phase_gradient",
                    "severity": 0.5,
                    "message": "PGO costs nearly twice multi-objective AdamW and is slightly worse.",
                },
            ]
        )
        inverse_mutations: dict[str, list[str]] = defaultdict(list)
        for parent, children in catalog["mutation_graph"].items():
            for child in children:
                inverse_mutations[child].append(parent)
        for record in records:
            record["parents"] = inverse_mutations.get(record["candidate_id"], [])
            record["pareto"] = record["candidate_id"] in frontiers[record["context"]]
        generated_directory = ROOT / "research" / "discovery" / "generated"
        generated_directory.mkdir(parents=True, exist_ok=True)
        outputs = {
            "candidate_registry.json": records,
            "pareto_frontiers.json": frontiers,
            "surprises.json": surprises,
        }
        artifacts: list[tuple[str, Path]] = []
        for filename, value in outputs.items():
            result_path = result_directory / filename
            generated_path = generated_directory / filename
            serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
            result_path.write_text(serialized, encoding="utf-8")
            generated_path.write_text(serialized, encoding="utf-8")
            artifacts.extend((("discovery", result_path), ("derived_discovery", generated_path)))
        proposals_path = result_directory / "proposals.yaml"
        generated_proposals_path = generated_directory / "proposals.yaml"
        proposal_text = yaml.safe_dump(config["proposals"], sort_keys=False, allow_unicode=False)
        proposals_path.write_text(proposal_text, encoding="utf-8")
        generated_proposals_path.write_text(proposal_text, encoding="utf-8")
        artifacts.extend(
            (("proposal", proposals_path), ("derived_proposal", generated_proposals_path))
        )
        for hypothesis in config["proposals"]:
            registry.register_hypothesis(
                f"H-{hypothesis['id']}", hypothesis["mechanism"], status="proposed"
            )
        for name, genome in catalog["architectures"].items():
            registry.register_architecture(
                name,
                genome["family"],
                genome["state"],
                genome["transition"],
                genome["measurement"],
                genome,
            )
        for source, replication, relationship, notes in (
            (
                "QN-000003",
                "QN-000006",
                "stronger-control replication",
                "Added GRU and paired-real control.",
            ),
            (
                "QN-000006",
                "QN-000008",
                "multi-world confirmation",
                "Five held-out worlds across four severities.",
            ),
            (
                "QN-000014",
                "QN-000016",
                "mechanism ablation",
                "Retrained phase, order, negative-evidence, and density-rank ablations.",
            ),
            (
                "QN-000014",
                "QN-000019",
                "representation follow-up",
                "Frozen hierarchical observable probes.",
            ),
            ("QN-000014", "QN-000023", "compute follow-up", "Realized velocity-based hard exit."),
            ("QN-000014", "QN-000025", "trajectory follow-up", "Evidence-level state audit."),
        ):
            registry.register_replication(source, replication, relationship, notes)
        registry.record_failure(
            "QN-000024",
            "trajectory_analysis",
            "AttributeError",
            "Used nonexistent torch.flatnonzero before replacement with torch.nonzero.",
        )
        results = {
            "experiment_id": experiment_id,
            "status": "complete",
            "candidate_count": len(records),
            "context_counts": {key: len(value) for key, value in grouped.items()},
            "pareto_frontiers": frontiers,
            "surprise_count": len(surprises),
            "proposal_count": len(config["proposals"]),
        }
        metrics_path = result_directory / "metrics.json"
        metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        artifacts.append(("metrics", metrics_path))
        registry.complete(
            experiment_id,
            {
                "discovery": {
                    "candidate_count": {"mean": float(len(records)), "std": 0.0},
                    "surprise_count": {"mean": float(len(surprises)), "std": 0.0},
                    "proposal_count": {"mean": float(len(config["proposals"])), "std": 0.0},
                }
            },
            [
                ("configuration", result_directory / "config.yaml"),
                ("environment", result_directory / "environment.json"),
                *artifacts,
            ],
        )
        return experiment_id, result_directory, results
    except Exception:
        registry.fail(experiment_id)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "configs" / "discovery_engine.yaml",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment_id, result_directory, results = run(config)
    print(json.dumps({"experiment_id": experiment_id, "result_directory": str(result_directory)}))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
