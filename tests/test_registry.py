"""Research-registry schema and query tests."""

from __future__ import annotations

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
    ranked = registry.best_metrics("top1", model_contains="complex")
    assert ranked[0]["experiment_id"] == experiment_id
    assert ranked[0]["mean"] == 0.7
