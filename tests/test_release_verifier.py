from scripts.verify_release import (
    MANIFEST_PATH,
    Recorder,
    artifact_paths,
    load_json,
    verify_manifest,
    verify_semantics,
)


def test_release_semantics_match_frozen_artifacts() -> None:
    recorder = Recorder()

    summary = verify_semantics(recorder)

    assert recorder.passed
    assert summary["discovery_effects"] == 2880
    assert summary["heldout_effects"] == 1920
    assert summary["total_independent_task_effects"] == 4800
    assert summary["grand_status"] == "blocked_before_execution"


def test_release_manifest_covers_unique_existing_artifacts() -> None:
    paths = artifact_paths()

    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)
    assert MANIFEST_PATH.is_file()

    recorder = Recorder()
    verify_manifest(recorder)
    assert recorder.passed


def test_release_scope_stays_nonclinical_and_unexecuted() -> None:
    claims = load_json("research/claims.json")
    decision = load_json("experiments/results/QN-GRAND-001/decision.json")

    assert claims["scope"] == "synthetic and nonclinical computational evidence"
    assert decision["qn_grand_001_executed"] is False
    assert decision["sealed_benchmark_opened"] is False
    assert decision["primary_confirmatory_effect_estimated"] is False
