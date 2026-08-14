from pathlib import Path

from scripts.build_dashboard_data import count_public_artifacts


def test_public_artifact_count_ignores_machine_local_outputs(tmp_path: Path) -> None:
    for name in ("config.yaml", "environment.json", "metrics.json", "VALIDITY.md"):
        (tmp_path / name).write_text("artifact", encoding="utf-8")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "model.pt").write_bytes(b"checkpoint")
    (tmp_path / "run.log").write_text("local log", encoding="utf-8")

    assert count_public_artifacts(tmp_path) == 4
