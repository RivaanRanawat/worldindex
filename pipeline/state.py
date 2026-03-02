import json
import sqlite3
from pathlib import Path
from typing import Any

from pipeline.models import PipelineTaskRecord, TaskStatus


class PipelineState:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def create_task(
        self,
        task_name: str,
        status: TaskStatus = "pending",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO pipeline_state (
                    task_name,
                    status,
                    retry_count,
                    checkpoint,
                    error_message,
                    started_at,
                    completed_at,
                    updated_at
                )
                VALUES (?, ?, 0, NULL, NULL, NULL, NULL, CURRENT_TIMESTAMP)
                """,
                (task_name, status),
            )

    def update_status(
        self,
        task_name: str,
        status: TaskStatus,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_state
                SET status = ?,
                    error_message = ?,
                    started_at = CASE
                        WHEN ? = 'running' THEN CURRENT_TIMESTAMP
                        WHEN ? = 'pending' THEN NULL
                        ELSE started_at
                    END,
                    completed_at = CASE
                        WHEN ? = 'completed' THEN CURRENT_TIMESTAMP
                        WHEN ? != 'completed' THEN NULL
                        ELSE completed_at
                    END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_name = ?
                """,
                (
                    status,
                    error_message if status == "failed" else None,
                    status,
                    status,
                    status,
                    status,
                    task_name,
                ),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown task: {task_name}")

    def get_status(self, task_name: str) -> PipelineTaskRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    task_name,
                    status,
                    retry_count,
                    checkpoint,
                    error_message,
                    started_at,
                    completed_at,
                    updated_at
                FROM pipeline_state
                WHERE task_name = ?
                """,
                (task_name,),
            ).fetchone()
        if row is None:
            return None
        return PipelineTaskRecord.model_validate(dict(row))

    def get_all_tasks(self) -> list[PipelineTaskRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    task_name,
                    status,
                    retry_count,
                    checkpoint,
                    error_message,
                    started_at,
                    completed_at,
                    updated_at
                FROM pipeline_state
                ORDER BY task_name
                """
            ).fetchall()
        return [PipelineTaskRecord.model_validate(dict(row)) for row in rows]

    def get_pending_tasks(self) -> list[PipelineTaskRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    task_name,
                    status,
                    retry_count,
                    checkpoint,
                    error_message,
                    started_at,
                    completed_at,
                    updated_at
                FROM pipeline_state
                WHERE status IN ('pending', 'failed')
                ORDER BY task_name
                """
            ).fetchall()
        return [PipelineTaskRecord.model_validate(dict(row)) for row in rows]

    def increment_retry(self, task_name: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_state
                SET retry_count = retry_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_name = ?
                """,
                (task_name,),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown task: {task_name}")
            row = connection.execute(
                """
                SELECT retry_count
                FROM pipeline_state
                WHERE task_name = ?
                """,
                (task_name,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown task: {task_name}")
        return int(row["retry_count"])

    def set_checkpoint(
        self,
        task_name: str,
        checkpoint: str | None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pipeline_state
                SET checkpoint = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE task_name = ?
                """,
                (checkpoint, task_name),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown task: {task_name}")

    def get_checkpoint(self, task_name: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT checkpoint
                FROM pipeline_state
                WHERE task_name = ?
                """,
                (task_name,),
            ).fetchone()
        if row is None:
            return None
        value = row["checkpoint"]
        return None if value is None else str(value)

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        payload = json.dumps(value, default=str)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO run_metadata (metadata_key, metadata_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(metadata_key) DO UPDATE SET
                    metadata_value = excluded.metadata_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, payload),
            )

    def get_metadata(self, key: str) -> Any | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT metadata_value
                FROM run_metadata
                WHERE metadata_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(str(row["metadata_value"]))

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    task_name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    checkpoint TEXT,
                    error_message TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_metadata (
                    metadata_key TEXT PRIMARY KEY,
                    metadata_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection
