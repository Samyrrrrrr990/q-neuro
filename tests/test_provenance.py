from __future__ import annotations

from pathlib import Path

import pytest

from qneuro.provenance import (
    artifact_record,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    verify_artifact_record,
)


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_sha256(first) == canonical_sha256(second)


def test_artifact_record_is_relative_and_detects_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "nested" / "result.json"
    artifact.parent.mkdir()
    artifact.write_text('{"result":1}\n', encoding="utf-8")
    record = artifact_record(artifact, tmp_path)
    assert record["path"] == "nested/result.json"
    assert record["sha256"] == file_sha256(artifact)
    assert verify_artifact_record(record, tmp_path)
    artifact.write_text('{"result":2}\n', encoding="utf-8")
    assert not verify_artifact_record(record, tmp_path)


def test_artifact_record_rejects_external_path(tmp_path: Path) -> None:
    external = tmp_path.parent / "external-qneuro-test.txt"
    external.write_text("outside", encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            artifact_record(external, tmp_path)
    finally:
        external.unlink()
