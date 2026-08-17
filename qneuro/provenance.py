"""Portable hashes and clean-source checks for confirmatory experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data with a stable byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty_paths(root: Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("unable to inspect Git worktree") from error
    return [line[3:] for line in output.splitlines() if line]


def require_clean_worktree(root: Path, *, allow_dirty: bool = False) -> list[str]:
    """Reject confirmatory execution from modified source unless explicitly running a smoke test."""

    dirty = git_dirty_paths(root)
    if dirty and not allow_dirty:
        preview = ", ".join(dirty[:5])
        raise RuntimeError(f"confirmatory runs require a clean Git worktree; modified: {preview}")
    return dirty


def dependency_inventory() -> dict[str, str]:
    """Return the installed Python distribution versions in stable name order."""

    packages = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in metadata.distributions()
        if distribution.metadata.get("Name")
    }
    return dict(sorted(packages.items()))


def environment_record(root: Path, *, command: list[str] | None = None) -> dict[str, Any]:
    dirty = git_dirty_paths(root)
    return {
        "git_commit": git_commit(root),
        "git_dirty": bool(dirty),
        "git_dirty_paths": dirty,
        "command": list(command or []),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "dependencies": dependency_inventory(),
    }


def artifact_record(path: Path, root: Path) -> dict[str, str | int]:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("artifact must be inside the repository root") from error
    return {
        "path": relative.as_posix(),
        "size_bytes": resolved_path.stat().st_size,
        "sha256": file_sha256(resolved_path),
    }


def verify_artifact_record(record: dict[str, Any], root: Path) -> bool:
    path = root / str(record["path"])
    return (
        path.is_file()
        and path.stat().st_size == int(record["size_bytes"])
        and file_sha256(path) == str(record["sha256"])
    )
