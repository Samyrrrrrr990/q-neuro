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
