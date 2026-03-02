from __future__ import annotations

import sqlite3
from pathlib import Path

from pipeline.state import PipelineState


def test_pipeline_state_tracks_transitions_and_metadata(tmp_path: Path) -> None:
    state = PipelineState(tmp_path / "state" / "pipeline.sqlite3")

    state.create_task("extract")
    state.create_task("compress")

    assert [task.task_name for task in state.get_all_tasks()] == ["compress", "extract"]
    assert [task.task_name for task in state.get_pending_tasks()] == ["compress", "extract"]

    state.update_status("extract", "running")
    state.set_checkpoint("extract", "12")
    assert [task.task_name for task in state.get_pending_tasks()] == ["compress"]

    retry_count = state.increment_retry("extract")
    state.update_status("extract", "failed", "boom")
    failed_task = state.get_status("extract")

    assert retry_count == 1
    assert failed_task is not None
    assert failed_task.status == "failed"
    assert failed_task.retry_count == 1
    assert failed_task.checkpoint == "12"
    assert failed_task.error_message == "boom"

    state.set_metadata("summary", {"clips": 12})
    assert state.get_metadata("summary") == {"clips": 12}

    state.update_status("extract", "completed")
    completed_task = state.get_status("extract")
    assert completed_task is not None
    assert completed_task.status == "completed"
    assert completed_task.error_message is None
    assert completed_task.completed_at is not None

    with sqlite3.connect(state.db_path) as connection:
        task_row = connection.execute(
            """
            SELECT status, retry_count, checkpoint
            FROM pipeline_state
            WHERE task_name = 'extract'
            """
        ).fetchone()
        metadata_row = connection.execute(
            """
            SELECT metadata_value
            FROM run_metadata
            WHERE metadata_key = 'summary'
            """
        ).fetchone()

    assert task_row == ("completed", 1, "12")
    assert metadata_row == ('{"clips": 12}',)
