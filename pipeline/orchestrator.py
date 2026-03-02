import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import uvicorn

from api.server import create_app
from compression import (
    TokenCompressor,
    discover_raw_batches,
    read_clip_from_shard,
    read_coarse_vectors_from_shard,
    run_compression_pipeline,
)
from extraction import run_extraction
from index import IndexBuilder, run_index_build
from pipeline.config import PipelineConfig
from pipeline.models import ArtifactSize, PipelineStatusSummary, SampleQueryResult, ValidationReport
from pipeline.state import PipelineState

_EXTRACTION_STAGE = "extract"
_COMPRESSION_STAGE = "compress"
_INDEX_STAGE = "build_index"
_STAGE_ORDER = (_EXTRACTION_STAGE, _COMPRESSION_STAGE, _INDEX_STAGE)
_EXTRACTION_CHECKPOINT_KEY = "single_node_extraction"
_COMPRESSION_LAST_SHARD_KEY = "compression:last_completed_shard_id"
_COMPRESSION_INDEXED_KEY = "compression:indexed_shard_id"


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._state = PipelineState(config.pipeline_db)
        self._logger = structlog.get_logger(__name__).bind(component="pipeline_orchestrator")

    def run_full_pipeline(self) -> dict[str, Any]:
        self._ensure_tasks()
        self._mark_run_started()

        for stage_name in _STAGE_ORDER:
            task = self._state.get_status(stage_name)
            if task is not None and task.status == "completed":
                self._logger.info("stage_skipped", stage=stage_name, reason="already_completed")
                continue
            self._run_stage_internal(stage_name)

        return self.get_status()

    def run_stage(self, stage_name: str) -> dict[str, Any]:
        normalized_stage = self._normalize_stage_name(stage_name)
        self._ensure_tasks()
        self._mark_run_started()
        self._prepare_manual_rerun(normalized_stage)
        self._run_stage_internal(normalized_stage)
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        self._ensure_tasks()
        tasks_by_name = {task.task_name: task for task in self._state.get_all_tasks()}
        ordered_tasks = [tasks_by_name[stage_name] for stage_name in _STAGE_ORDER if stage_name in tasks_by_name]
        summary = PipelineStatusSummary(
            tasks=ordered_tasks,
            total_clips_processed=self._count_processed_clips(),
            estimated_seconds_remaining=self._estimate_seconds_remaining(ordered_tasks),
        )
        return summary.model_dump(mode="json")

    def validate(self) -> ValidationReport:
        recall_at_10 = self._validate_index_recall()
        mean_cosine_similarity, sampled_clip_count = self._validate_compression_similarity()
        sample_queries = self._sample_queries()
        artifact_sizes = self._storage_summary()
        report = ValidationReport(
            recall_at_10=recall_at_10,
            mean_cosine_similarity=mean_cosine_similarity,
            sampled_clip_count=sampled_clip_count,
            sample_queries=sample_queries,
            artifact_sizes=artifact_sizes,
        )
        self._logger.info(
            "validation_complete",
            recall_at_10=report.recall_at_10,
            mean_cosine_similarity=report.mean_cosine_similarity,
            sampled_clip_count=report.sampled_clip_count,
        )
        return report

    def serve(self) -> None:
        uvicorn.run(
            create_app(config=self.config.serving),
            host=self.config.serving.host,
            port=self.config.serving.port,
        )

    def _run_stage_internal(self, stage_name: str) -> None:
        checkpoint = self._state.get_checkpoint(stage_name)
        self._state.update_status(stage_name, "running")
        started_at = time.perf_counter()
        self._logger.info("stage_started", stage=stage_name, checkpoint=checkpoint)

        try:
            new_checkpoint = self._execute_stage(stage_name, checkpoint)
            persisted_checkpoint = new_checkpoint if new_checkpoint is not None else self._read_stage_checkpoint(stage_name)
            if persisted_checkpoint is not None:
                self._state.set_checkpoint(stage_name, persisted_checkpoint)
            self._state.update_status(stage_name, "completed")
            duration_seconds = round(time.perf_counter() - started_at, 3)
            self._state.set_metadata(f"{stage_name}:duration_seconds", duration_seconds)
            self._state.set_metadata("last_successful_stage", stage_name)
            self._logger.info(
                "stage_completed",
                stage=stage_name,
                duration_seconds=duration_seconds,
                checkpoint=persisted_checkpoint,
            )
        except Exception as exc:
            duration_seconds = round(time.perf_counter() - started_at, 3)
            self._state.increment_retry(stage_name)
            persisted_checkpoint = self._read_stage_checkpoint(stage_name)
            if persisted_checkpoint is not None:
                self._state.set_checkpoint(stage_name, persisted_checkpoint)
            self._state.update_status(stage_name, "failed", str(exc))
            self._state.set_metadata(f"{stage_name}:duration_seconds", duration_seconds)
            self._logger.error(
                "stage_failed",
                stage=stage_name,
                duration_seconds=duration_seconds,
                checkpoint=persisted_checkpoint,
                error=str(exc),
            )
            raise

    def _execute_stage(
        self,
        stage_name: str,
        checkpoint: str | None,
    ) -> str | None:
        if stage_name == _EXTRACTION_STAGE:
            resume_from = None if checkpoint is None else int(checkpoint)
            final_clip_index = run_extraction(self.config.extraction, checkpoint=resume_from)
            return str(final_clip_index)

        if stage_name == _COMPRESSION_STAGE:
            run_compression_pipeline(
                self.config.compression,
                checkpoint=checkpoint,
                build_index=False,
            )
            return self._read_stage_checkpoint(_COMPRESSION_STAGE)

        if stage_name == _INDEX_STAGE:
            index_path = run_index_build(self.config.index, checkpoint=checkpoint)
            self._mark_compression_index_built()
            return str(index_path)

        raise ValueError(f"Unknown stage: {stage_name}")

    def _prepare_manual_rerun(self, stage_name: str) -> None:
        task = self._state.get_status(stage_name)
        if task is None or task.status != "completed":
            return

        start_index = _STAGE_ORDER.index(stage_name)
        for affected_stage in _STAGE_ORDER[start_index:]:
            self._reset_internal_stage_state(affected_stage)
            self._state.update_status(affected_stage, "pending")
            self._state.set_checkpoint(affected_stage, None)

    def _reset_internal_stage_state(self, stage_name: str) -> None:
        if stage_name == _EXTRACTION_STAGE and self.config.extraction.checkpoint_db.exists():
            self.config.extraction.checkpoint_db.unlink()
            return

        if stage_name == _COMPRESSION_STAGE and self.config.compression.checkpoint_db.exists():
            self.config.compression.checkpoint_db.unlink()

    def _ensure_tasks(self) -> None:
        for stage_name in _STAGE_ORDER:
            self._state.create_task(stage_name)

    def _mark_run_started(self) -> None:
        self._state.set_metadata("run_started_at", datetime.now(timezone.utc).isoformat())

    def _normalize_stage_name(self, stage_name: str) -> str:
        normalized = stage_name.strip().lower().replace("-", "_")
        if normalized not in _STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage_name}")
        return normalized

    def _read_stage_checkpoint(self, stage_name: str) -> str | None:
        if stage_name == _EXTRACTION_STAGE:
            if not self.config.extraction.checkpoint_db.exists():
                return None
            with sqlite3.connect(self.config.extraction.checkpoint_db) as connection:
                row = connection.execute(
                    """
                    SELECT clip_index
                    FROM extraction_checkpoint
                    WHERE checkpoint_key = ?
                    """,
                    (_EXTRACTION_CHECKPOINT_KEY,),
                ).fetchone()
            if row is None:
                return None
            return str(int(row[0]))

        if stage_name == _COMPRESSION_STAGE:
            return self._read_compression_checkpoint(_COMPRESSION_LAST_SHARD_KEY)

        if stage_name == _INDEX_STAGE and self.config.index.output_path.exists():
            return str(self.config.index.output_path)

        return None

    def _read_compression_checkpoint(self, key: str) -> str | None:
        if not self.config.compression.checkpoint_db.exists():
            return None

        with sqlite3.connect(self.config.compression.checkpoint_db) as connection:
            row = connection.execute(
                """
                SELECT checkpoint_value
                FROM compression_checkpoint
                WHERE checkpoint_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def _mark_compression_index_built(self) -> None:
        shard_paths = sorted(self.config.compression.shard_dir.glob("shard_*.widx"))
        if not shard_paths:
            return

        final_shard_id = int(shard_paths[-1].stem.rsplit("_", maxsplit=1)[-1])
        self.config.compression.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.config.compression.checkpoint_db) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS compression_checkpoint (
                    checkpoint_key TEXT PRIMARY KEY,
                    checkpoint_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO compression_checkpoint (checkpoint_key, checkpoint_value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(checkpoint_key) DO UPDATE SET
                    checkpoint_value = excluded.checkpoint_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (_COMPRESSION_INDEXED_KEY, str(final_shard_id)),
            )

    def _count_processed_clips(self) -> int:
        if self.config.compression.consolidated_metadata_path.exists():
            return int(pl.read_parquet(self.config.compression.consolidated_metadata_path).height)

        metadata_paths = sorted(self.config.extraction.output_dir.glob("metadata_*.parquet"))
        if metadata_paths:
            return int(sum(pl.read_parquet(path).height for path in metadata_paths))

        extraction_checkpoint = self._read_stage_checkpoint(_EXTRACTION_STAGE)
        if extraction_checkpoint is None:
            return 0
        return max(int(extraction_checkpoint) + 1, 0)

    def _estimate_seconds_remaining(self, tasks: list[Any]) -> float | None:
        run_started_at = self._state.get_metadata("run_started_at")
        if run_started_at is None:
            return None

        progress_units = 0.0
        for task in tasks:
            if task.status == "completed":
                progress_units += 1.0
            elif task.status == "running":
                progress_units += 0.5

        if progress_units == 0.0:
            return None

        started_at = datetime.fromisoformat(str(run_started_at))
        elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
        remaining_units = len(tasks) - progress_units
        if remaining_units <= 0.0:
            return 0.0
        return round((elapsed_seconds / progress_units) * remaining_units, 3)

    def _validate_index_recall(self) -> float:
        coarse_vectors = self._load_all_coarse_vectors()
        builder = IndexBuilder(
            hnsw_m=self.config.index.hnsw_m,
            ef_construction=self.config.index.ef_construction,
            ef_search=self.config.index.ef_search,
        )
        return builder.validate_index(self.config.index.output_path, coarse_vectors)

    def _validate_compression_similarity(self) -> tuple[float, int]:
        metadata = pl.read_parquet(self.config.compression.consolidated_metadata_path)
        if metadata.height == 0:
            raise ValueError("No clips available for compression validation")

        sample_size = min(100, metadata.height)
        rng = np.random.default_rng(0)
        sampled_indices = rng.choice(metadata.height, size=sample_size, replace=False)
        sampled_rows = [metadata.row(int(row_index), named=True) for row_index in sampled_indices.tolist()]
        clip_lookup = self._build_raw_clip_lookup({int(row["clip_index"]) for row in sampled_rows})
        compressor = TokenCompressor.load(self.config.compression.compressor_dir)
        if compressor.pca_mean is None or compressor.pca_components is None:
            raise RuntimeError("Loaded compressor is missing PCA artifacts")

        raw_batch_cache: dict[Path, np.ndarray] = {}
        similarities: list[float] = []
        for row in sampled_rows:
            clip_index = int(row["clip_index"])
            token_path, token_offset = clip_lookup[clip_index]
            token_batch = raw_batch_cache.get(token_path)
            if token_batch is None:
                token_batch = np.load(token_path, mmap_mode="r")
                raw_batch_cache[token_path] = token_batch

            raw_tokens = np.asarray(token_batch[token_offset], dtype=np.float32)
            shard_path = self.config.compression.shard_dir / f"shard_{int(row['shard_id']):08d}.widx"
            compressed_clip = read_clip_from_shard(shard_path, int(row["shard_offset"]))
            projected_tokens = (raw_tokens - compressor.pca_mean) @ compressor.pca_components.T
            similarities.append(
                self._average_cosine_similarity(
                    projected_tokens,
                    compressor.decompress_clip(compressed_clip),
                )
            )

        return float(np.mean(similarities)), sample_size

    def _sample_queries(self) -> list[SampleQueryResult]:
        import faiss

        coarse_vectors = self._load_all_coarse_vectors()
        if coarse_vectors.shape[0] == 0:
            return []

        metadata = pl.read_parquet(self.config.compression.consolidated_metadata_path)
        metadata_rows = [metadata.row(index, named=True) for index in range(metadata.height)]
        normalized_vectors = self._normalize_rows(coarse_vectors)
        query_count = min(3, normalized_vectors.shape[0])
        top_k = min(5, normalized_vectors.shape[0])
        query_indices = np.random.default_rng(0).choice(
            normalized_vectors.shape[0],
            size=query_count,
            replace=False,
        )

        faiss_index = faiss.read_index(str(self.config.index.output_path))
        if hasattr(faiss_index, "hnsw"):
            faiss_index.hnsw.efSearch = self.config.index.ef_search

        results: list[SampleQueryResult] = []
        for query_index in query_indices.tolist():
            query = np.ascontiguousarray(normalized_vectors[query_index : query_index + 1])
            _, neighbor_indices = faiss_index.search(query, top_k)
            top_clip_ids = [int(value) for value in neighbor_indices[0].tolist() if value >= 0]
            top_episode_ids = [str(metadata_rows[clip_id]["episode_id"]) for clip_id in top_clip_ids]
            results.append(
                SampleQueryResult(
                    query_clip_id=int(query_index),
                    top_clip_ids=top_clip_ids,
                    top_episode_ids=top_episode_ids,
                )
            )

        return results

    def _storage_summary(self) -> list[ArtifactSize]:
        artifact_paths = [
            self.config.extraction.output_dir,
            self.config.extraction.checkpoint_db,
            self.config.compression.output_dir,
            self.config.compression.checkpoint_db,
            self.config.index.output_path,
            self.config.pipeline_db,
        ]
        summary: list[ArtifactSize] = []
        for path in artifact_paths:
            if not path.exists():
                continue
            summary.append(ArtifactSize(path=path, bytes=self._path_size(path)))
        return summary

    def _load_all_coarse_vectors(self) -> np.ndarray:
        shard_paths = sorted(self.config.compression.shard_dir.glob("shard_*.widx"))
        if not shard_paths:
            raise FileNotFoundError(f"No compressed shards found in {self.config.compression.shard_dir}")
        return np.ascontiguousarray(
            np.concatenate(
                [read_coarse_vectors_from_shard(shard_path) for shard_path in shard_paths],
                axis=0,
            ).astype(np.float32, copy=False)
        )

    def _build_raw_clip_lookup(self, clip_indices: set[int]) -> dict[int, tuple[Path, int]]:
        remaining = set(clip_indices)
        lookup: dict[int, tuple[Path, int]] = {}

        for raw_batch in discover_raw_batches(self.config.compression.raw_dir):
            if not remaining:
                break

            clip_series = pl.read_parquet(raw_batch.metadata_path).get_column("clip_index")
            for offset, clip_index in enumerate(clip_series.to_list()):
                current_clip_index = int(clip_index)
                if current_clip_index not in remaining:
                    continue
                lookup[current_clip_index] = (raw_batch.token_path, offset)
                remaining.remove(current_clip_index)
                if not remaining:
                    break

        if remaining:
            missing = ", ".join(str(value) for value in sorted(remaining))
            raise KeyError(f"Missing raw clip tokens for clip indices: {missing}")

        return lookup

    def _path_size(self, path: Path) -> int:
        if path.is_file():
            return int(path.stat().st_size)
        return int(sum(child.stat().st_size for child in path.rglob("*") if child.is_file()))

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        return np.ascontiguousarray(matrix / safe_norms, dtype=np.float32)

    @staticmethod
    def _average_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        numerator = np.sum(a * b, axis=1)
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        safe_denominator = np.where(denominator == 0.0, 1.0, denominator)
        return float(np.mean(numerator / safe_denominator))
