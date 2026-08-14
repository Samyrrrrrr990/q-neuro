"""Research-registry schema and query tests."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from qneuro.registry import ExperimentRegistry


def test_registry_tracks_research_entities_and_ranked_metrics(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.sqlite3")
    experiment_id, result_directory = registry.reserve({"experiment": "test"}, tmp_path / "runs")
    metrics_path = result_directory / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    registry.complete(
        experiment_id,
        {"complex@shift": {"top1": {"mean": 0.7, "std": 0.1}}},
        [("metrics", metrics_path)],
    )
    registry.register_hypothesis("H-001", "Complex order helps", status="tested")
    registry.register_hypothesis("H-001", "Complex order helps", status="tested")
    registry.register_architecture(
        "complex_operator",
        "operator",
        "complex",
        "low_rank_noncommutative",
        "born",
        {"rank": 2},
    )
    registry.register_replication(experiment_id, experiment_id, "self-test", "schema test")
    registry.record_failure(experiment_id, "analysis", "ValueError", "synthetic failure")
    document_hash = hashlib.sha256(b"frozen protocol").hexdigest()
    registry.register_preregistration("PREREG-001", "1.0.0", "docs/protocol.md", document_hash)
    registry.attach_protocol(experiment_id, "PREREG-001", "H-001", ["python", "runner.py"])
    registry.register_trial(
        "TRIAL-001",
        experiment_id,
        "complex_operator",
        "synthetic",
        "world-1",
        "noise",
        0.5,
        11,
        20_000,
        {"training_flops": 100},
        {"top1": 0.7},
        "complete",
    )
    ranked = registry.best_metrics("top1", model_contains="complex")
    assert ranked[0]["experiment_id"] == experiment_id
    assert ranked[0]["mean"] == 0.7

    with pytest.raises(ValueError, match="immutable"):
        registry.register_hypothesis("H-001", "Changed after results", status="tested")
    with pytest.raises(sqlite3.IntegrityError):
        registry.register_trial(
            "TRIAL-001",
            experiment_id,
            "real_operator",
            "synthetic",
            "world-1",
            "noise",
            0.5,
            11,
            20_000,
            {},
            {},
            "complete",
        )


def test_registry_named_grand_experiment_can_be_blocked(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments" / "registry.sqlite3")
    experiment_id, result_directory = registry.reserve_named(
        "QN-GRAND-001", {"stage": "preflight"}, tmp_path / "experiments" / "results"
    )
    artifact = result_directory / "preflight.json"
    artifact.write_text('{"passed": false}\n')
    registry.block(experiment_id, [("preflight", artifact)])
    with sqlite3.connect(registry.path) as connection:
        status = connection.execute(
            "SELECT status FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()[0]
    assert status == "blocked"
    with pytest.raises(FileExistsError):
        registry.reserve_named(
            "QN-GRAND-001", {"stage": "preflight"}, tmp_path / "experiments" / "results"
        )
    with pytest.raises(ValueError, match="QN-GRAND"):
        registry.reserve_named(
            "arbitrary", {"stage": "preflight"}, tmp_path / "experiments" / "results"
        )
