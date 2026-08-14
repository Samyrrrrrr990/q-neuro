"""Lightweight, never-overwriting SQLite experiment registry."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
        return experiment_id, result_directory

    def complete(
        self,
        experiment_id: str,
        summaries: dict[str, dict[str, dict[str, float]]],
        artifacts: list[tuple[str, Path]],
    ) -> None:
        with self._connect() as connection:
            for model, metrics in summaries.items():
                for metric, values in metrics.items():
                    connection.execute(
                        "INSERT INTO metrics(experiment_id, model, metric, mean, std) VALUES(?,?,?,?,?)",
                        (experiment_id, model, metric, values.get("mean"), values.get("std")),
                    )
            for kind, path in artifacts:
                connection.execute(
                    "INSERT INTO artifacts(experiment_id, kind, path) VALUES(?,?,?)",
                    (experiment_id, kind, str(path)),
                )
            connection.execute(
                "UPDATE experiments SET status=? WHERE experiment_id=?", ("complete", experiment_id)
            )

    def fail(self, experiment_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE experiments SET status=? WHERE experiment_id=?", ("failed", experiment_id)
            )

    def register_hypothesis(
        self,
        hypothesis_id: str,
        statement: str,
        status: str = "open",
        parent_hypothesis_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO hypotheses VALUES(?,?,?,?,?)",
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
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO architectures VALUES(?,?,?,?,?,?)",
                (
                    architecture_id,
                    family,
                    state_type,
                    transition_type,
                    measurement_type,
                    json.dumps(config, sort_keys=True),
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
