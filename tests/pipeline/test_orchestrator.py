from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from api.config import ServingConfig
from compression import CompressionPipelineConfig
from extraction import ExtractionConfig
from index import IndexConfig
from ingestion.config import DatasetConfig
from pipeline.config import PipelineConfig
from pipeline.orchestrator import PipelineOrchestrator

_EXTRACTION_CHECKPOINT_KEY = "single_node_extraction"


def _build_dataset_config(repo_id: str = "tests/demo__4") -> DatasetConfig:
    return DatasetConfig.model_validate(
        {
            "repo_id": repo_id,
            "image_key": "observation.images.main",
            "source_fps": 15,
            "target_fps": 4,
            "robot_type": "testbot",
            "language_key": "language_instruction",
            "clip_length": 64,
            "clip_stride": 32,
        }
    )


def _build_pipeline_config(tmp_path: Path) -> PipelineConfig:
    state_dir = tmp_path / "state"
    extraction_output_dir = tmp_path / "raw"
    compression_output_dir = tmp_path / "compressed"

    extraction_config = ExtractionConfig(
        dataset_configs=[_build_dataset_config()],
        model_id="fake:6x4",
        device="cpu",
        batch_size=1,
        queue_depth=2,
        flush_size=2,
        output_dir=extraction_output_dir,
        checkpoint_db=state_dir / "extraction.sqlite3",
    )
    compression_config = CompressionPipelineConfig(
        raw_dir=extraction_output_dir,
        output_dir=compression_output_dir,
        checkpoint_db=state_dir / "compression.sqlite3",
        sample_size=16,
        pca_dim=8,
        n_centroids=4,
        n_bits=2,
        random_seed=0,
    )
    index_config = IndexConfig(
        compressed_dir=compression_config.shard_dir,
        output_path=compression_config.faiss_index_path,
        hnsw_m=16,
        ef_construction=32,
        ef_search=32,
    )

    return PipelineConfig(
        datasets=[tmp_path / "dataset.yaml"],
        extraction=extraction_config,
        compression=compression_config,
        index=index_config,
        serving=ServingConfig(data_dir=compression_output_dir),
        output_dir=tmp_path / "artifacts",
        pipeline_db=state_dir / "pipeline.sqlite3",
    )


def _task_by_name(status: dict[str, Any], task_name: str) -> dict[str, Any]:
    for task in status["tasks"]:
        if task["task_name"] == task_name:
            return task
    raise KeyError(task_name)


def _write_extraction_checkpoint(checkpoint_db: Path, clip_index: int) -> None:
    checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(checkpoint_db) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_checkpoint (
                checkpoint_key TEXT PRIMARY KEY,
                clip_index INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO extraction_checkpoint (checkpoint_key, clip_index, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(checkpoint_key) DO UPDATE SET
                clip_index = excluded.clip_index,
                updated_at = CURRENT_TIMESTAMP
            """,
            (_EXTRACTION_CHECKPOINT_KEY, clip_index),
        )


def test_extract_stage_resumes_from_saved_checkpoint_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_pipeline_config(tmp_path)
    orchestrator = PipelineOrchestrator(config)
    observed_checkpoints: list[int | None] = []

    def fake_run_extraction(current_config: ExtractionConfig, checkpoint: int | None = None) -> int:
        observed_checkpoints.append(checkpoint)
        if checkpoint is None:
            _write_extraction_checkpoint(current_config.checkpoint_db, 1)
            raise RuntimeError("simulated extraction crash")

        assert checkpoint == 1
        _write_extraction_checkpoint(current_config.checkpoint_db, 3)
        return 3

    monkeypatch.setattr("pipeline.orchestrator.run_extraction", fake_run_extraction)

    with pytest.raises(RuntimeError, match="simulated extraction crash"):
        orchestrator.run_stage("extract")

    failed_status = orchestrator.get_status()
    failed_task = _task_by_name(failed_status, "extract")
    assert failed_task["status"] == "failed"
    assert failed_task["retry_count"] == 1
    assert failed_task["checkpoint"] == "1"
    assert failed_status["total_clips_processed"] == 2

    completed_status = orchestrator.run_stage("extract")
    completed_task = _task_by_name(completed_status, "extract")
    assert completed_task["status"] == "completed"
    assert completed_task["checkpoint"] == "3"
    assert completed_status["total_clips_processed"] == 4
    assert observed_checkpoints == [None, 1]
