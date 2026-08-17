"""Lightweight, never-overwriting SQLite experiment registry."""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qneuro.provenance import artifact_record, canonical_sha256

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT UNIQUE,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    result_directory TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id TEXT NOT NULL,
    model TEXT NOT NULL,
    metric TEXT NOT NULL,
    mean REAL,
    std REAL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    experiment_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    parent_hypothesis_id TEXT
);
CREATE TABLE IF NOT EXISTS architectures (
    architecture_id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    state_type TEXT NOT NULL,
    transition_type TEXT NOT NULL,
    measurement_type TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    experiment_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS replications (
    source_experiment_id TEXT NOT NULL,
    replication_experiment_id TEXT NOT NULL,
    relationship TEXT NOT NULL,
    notes TEXT NOT NULL,
    UNIQUE(source_experiment_id, replication_experiment_id),
    FOREIGN KEY(source_experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY(replication_experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS preregistrations (
    preregistration_id TEXT PRIMARY KEY,
    version TEXT NOT NULL,
    document_path TEXT NOT NULL,
    document_sha256 TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_provenance (
    experiment_id TEXT PRIMARY KEY,
    config_sha256 TEXT NOT NULL,
    preregistration_id TEXT,
    hypothesis_id TEXT,
    command_json TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY(preregistration_id) REFERENCES preregistrations(preregistration_id),
    FOREIGN KEY(hypothesis_id) REFERENCES hypotheses(hypothesis_id)
);
CREATE TABLE IF NOT EXISTS artifact_digests (
    experiment_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    UNIQUE(experiment_id, path),
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
CREATE TABLE IF NOT EXISTS run_trials (
    trial_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    architecture TEXT NOT NULL,
    dataset TEXT NOT NULL,
    world TEXT NOT NULL,
    shift_type TEXT NOT NULL,
    severity REAL NOT NULL,
    seed INTEGER NOT NULL,
    parameter_count INTEGER NOT NULL,
    compute_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);
"""


class ExperimentRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def reserve(self, config: dict[str, Any], result_root: Path) -> tuple[str, Path]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO experiments(experiment_id, created_at, status, config_json, result_directory) "
                "VALUES(NULL, ?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), "reserved", json.dumps(config, sort_keys=True), ""),
            )
            numeric_id = int(cursor.lastrowid)
            experiment_id = f"QN-{numeric_id:06d}"
            result_directory = result_root / experiment_id
            if result_directory.exists():
                raise FileExistsError(f"refusing to overwrite {result_directory}")
            result_directory.mkdir(parents=True)
            connection.execute(
                "UPDATE experiments SET experiment_id=?, status=?, result_directory=? WHERE id=?",
                (experiment_id, "running", str(result_directory), numeric_id),
            )
            connection.execute(
                "INSERT INTO experiment_provenance VALUES(?,?,?,?,?)",
                (experiment_id, canonical_sha256(config), None, None, "[]"),
            )
        return experiment_id, result_directory

    def reserve_named(
        self, experiment_id: str, config: dict[str, Any], result_root: Path
    ) -> tuple[str, Path]:
        """Reserve a preregistered named experiment without permitting arbitrary identifiers."""

        if re.fullmatch(r"QN-GRAND-\d{3}", experiment_id) is None:
            raise ValueError("named experiment IDs must match QN-GRAND-NNN")
        result_directory = result_root / experiment_id
        if result_directory.exists():
            raise FileExistsError(f"refusing to overwrite {result_directory}")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments(experiment_id, created_at, status, config_json, "
                "result_directory) VALUES(?,?,?,?,?)",
                (
                    experiment_id,
                    datetime.now(UTC).isoformat(),
                    "running",
                    json.dumps(config, sort_keys=True),
                    str(result_directory),
                ),
            )
            connection.execute(
                "INSERT INTO experiment_provenance VALUES(?,?,?,?,?)",
                (experiment_id, canonical_sha256(config), None, None, "[]"),
            )
        result_directory.mkdir(parents=True)
        return experiment_id, result_directory

    def complete(
        self,
        experiment_id: str,
        summaries: dict[str, dict[str, dict[str, float]]],
        artifacts: list[tuple[str, Path]],
    ) -> None:
        with self._connect() as connection:
            repository_root = self.path.resolve().parents[1]
            for model, metrics in summaries.items():
                for metric, values in metrics.items():
                    connection.execute(
                        "INSERT INTO metrics(experiment_id, model, metric, mean, std) VALUES(?,?,?,?,?)",
                        (experiment_id, model, metric, values.get("mean"), values.get("std")),
                    )
            for kind, path in artifacts:
                record = artifact_record(path, repository_root)
                connection.execute(
                    "INSERT INTO artifacts(experiment_id, kind, path) VALUES(?,?,?)",
                    (experiment_id, kind, record["path"]),
                )
                connection.execute(
                    "INSERT INTO artifact_digests VALUES(?,?,?,?,?)",
                    (
                        experiment_id,
                        kind,
                        record["path"],
                        record["size_bytes"],
                        record["sha256"],
                    ),
                )
            connection.execute(
                "UPDATE experiments SET status=? WHERE experiment_id=?", ("complete", experiment_id)
            )

    def fail(self, experiment_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE experiments SET status=? WHERE experiment_id=?", ("failed", experiment_id)
            )

    def block(self, experiment_id: str, artifacts: list[tuple[str, Path]]) -> None:
        """Record immutable preflight artifacts for a study blocked before outcome access."""

        with self._connect() as connection:
            repository_root = self.path.resolve().parents[1]
            for kind, path in artifacts:
                record = artifact_record(path, repository_root)
                connection.execute(
                    "INSERT INTO artifacts(experiment_id, kind, path) VALUES(?,?,?)",
                    (experiment_id, kind, record["path"]),
                )
                connection.execute(
                    "INSERT INTO artifact_digests VALUES(?,?,?,?,?)",
                    (
                        experiment_id,
                        kind,
                        record["path"],
                        record["size_bytes"],
                        record["sha256"],
                    ),
                )
            connection.execute(
                "UPDATE experiments SET status=? WHERE experiment_id=?", ("blocked", experiment_id)
            )

    def register_hypothesis(
        self,
        hypothesis_id: str,
        statement: str,
        status: str = "open",
        parent_hypothesis_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT statement, status, parent_hypothesis_id FROM hypotheses "
                "WHERE hypothesis_id=?",
                (hypothesis_id,),
            ).fetchone()
            requested = (statement, status, parent_hypothesis_id)
            if existing is not None:
                if existing != requested:
                    raise ValueError(f"hypothesis {hypothesis_id} is immutable")
                return
            connection.execute(
                "INSERT INTO hypotheses VALUES(?,?,?,?,?)",
                (
                    hypothesis_id,
                    statement,
                    status,
                    datetime.now(UTC).isoformat(),
                    parent_hypothesis_id,
                ),
            )

    def register_architecture(
        self,
        architecture_id: str,
        family: str,
        state_type: str,
        transition_type: str,
        measurement_type: str,
        config: dict[str, Any],
    ) -> None:
        config_json = json.dumps(config, sort_keys=True)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT family, state_type, transition_type, measurement_type, config_json "
                "FROM architectures WHERE architecture_id=?",
                (architecture_id,),
            ).fetchone()
            requested = (family, state_type, transition_type, measurement_type, config_json)
            if existing is not None:
                if existing != requested:
                    raise ValueError(f"architecture {architecture_id} is immutable")
                return
            connection.execute(
                "INSERT INTO architectures VALUES(?,?,?,?,?,?)",
                (
                    architecture_id,
                    family,
                    state_type,
                    transition_type,
                    measurement_type,
                    config_json,
                ),
            )

    def register_preregistration(
        self,
        preregistration_id: str,
        version: str,
        document_path: str,
        document_sha256: str,
        status: str = "frozen",
    ) -> None:
        """Register a preregistration once; amendments require a new identifier."""

        requested = (version, document_path, document_sha256, status)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT version, document_path, document_sha256, status FROM preregistrations "
                "WHERE preregistration_id=?",
                (preregistration_id,),
            ).fetchone()
            if existing is not None:
                if existing != requested:
                    raise ValueError(f"preregistration {preregistration_id} is immutable")
                return
            connection.execute(
                "INSERT INTO preregistrations VALUES(?,?,?,?,?,?)",
                (
                    preregistration_id,
                    version,
                    document_path,
                    document_sha256,
                    status,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def attach_protocol(
        self,
        experiment_id: str,
        preregistration_id: str,
        hypothesis_id: str,
        command: list[str],
    ) -> None:
        """Attach frozen protocol identity and invocation before recording outcomes."""

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE experiment_provenance SET preregistration_id=?, hypothesis_id=?, "
                "command_json=? WHERE experiment_id=? AND preregistration_id IS NULL",
                (preregistration_id, hypothesis_id, json.dumps(command), experiment_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"protocol for {experiment_id} is missing or already attached")

    def register_trial(
        self,
        trial_id: str,
        experiment_id: str,
        architecture: str,
        dataset: str,
        world: str,
        shift_type: str,
        severity: float,
        seed: int,
        parameter_count: int,
        compute: dict[str, Any],
        metrics: dict[str, Any],
        status: str,
    ) -> None:
        """Write one immutable architecture/world/shift/seed trial."""

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO run_trials VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    trial_id,
                    experiment_id,
                    architecture,
                    dataset,
                    world,
                    shift_type,
                    float(severity),
                    int(seed),
                    int(parameter_count),
                    json.dumps(compute, sort_keys=True),
                    json.dumps(metrics, sort_keys=True),
                    status,
                ),
            )

    def record_failure(
        self,
        experiment_id: str,
        stage: str,
        error_type: str,
        message: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO failures VALUES(?,?,?,?,?)",
                (
                    experiment_id,
                    stage,
                    error_type,
                    message,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def register_replication(
        self,
        source_experiment_id: str,
        replication_experiment_id: str,
        relationship: str,
        notes: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO replications VALUES(?,?,?,?)",
                (source_experiment_id, replication_experiment_id, relationship, notes),
            )

    def best_metrics(
        self,
        metric: str,
        *,
        lower_is_better: bool = False,
        model_contains: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Query ranked registered metrics without embedding experiment-specific schemas."""

        ordering = "ASC" if lower_is_better else "DESC"
        query = (
            "SELECT experiment_id, model, metric, mean, std FROM metrics "
            "WHERE metric=? AND mean IS NOT NULL"
        )
        parameters: list[Any] = [metric]
        if model_contains is not None:
            query += " AND model LIKE ?"
            parameters.append(f"%{model_contains}%")
        query += f" ORDER BY mean {ordering} LIMIT ?"
        parameters.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "experiment_id": row[0],
                "model": row[1],
                "metric": row[2],
                "mean": row[3],
                "std": row[4],
            }
            for row in rows
        ]
