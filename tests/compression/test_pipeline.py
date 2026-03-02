from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sklearn")

from compression import CompressionPipelineConfig, read_clip_from_shard, run_compression_pipeline


def _make_raw_tokens(seed: int, clip_count: int, tokens_per_clip: int, input_dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, input_dim)).astype(np.float32)
    assignments = rng.integers(0, centers.shape[0], size=(clip_count, tokens_per_clip))
    noise = rng.normal(scale=0.01, size=(clip_count, tokens_per_clip, input_dim)).astype(np.float32)
    return np.ascontiguousarray(centers[assignments] + noise)


def _write_raw_batch(raw_dir: Path, start_clip_index: int, tokens: np.ndarray) -> None:
    end_clip_index = start_clip_index + tokens.shape[0] - 1
    token_path = raw_dir / f"tokens_{start_clip_index:08d}_{end_clip_index:08d}.npy"
    metadata_path = raw_dir / f"metadata_{start_clip_index:08d}_{end_clip_index:08d}.parquet"

    np.save(token_path, tokens.astype(np.float32), allow_pickle=False)
    metadata = pl.DataFrame(
        {
            "clip_index": list(range(start_clip_index, end_clip_index + 1)),
            "episode_id": [f"episode_{value}" for value in range(start_clip_index, end_clip_index + 1)],
            "dataset_name": ["demo"] * tokens.shape[0],
            "robot_type": ["testbot"] * tokens.shape[0],
            "clip_start_frame": [value * 8 for value in range(start_clip_index, end_clip_index + 1)],
            "clip_end_frame": [value * 8 + 7 for value in range(start_clip_index, end_clip_index + 1)],
            "timestamp_start": [float(value) for value in range(start_clip_index, end_clip_index + 1)],
            "timestamp_end": [float(value) + 0.875 for value in range(start_clip_index, end_clip_index + 1)],
            "language_instruction": [f"task_{value}" for value in range(start_clip_index, end_clip_index + 1)],
            "num_original_frames": [8] * tokens.shape[0],
        }
    )
    metadata.write_parquet(metadata_path)


def test_run_compression_pipeline_reuses_existing_outputs_on_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_batch(raw_dir, 0, _make_raw_tokens(seed=0, clip_count=2, tokens_per_clip=12, input_dim=32))
    _write_raw_batch(raw_dir, 2, _make_raw_tokens(seed=1, clip_count=1, tokens_per_clip=12, input_dim=32))

    config = CompressionPipelineConfig(
        raw_dir=raw_dir,
        output_dir=tmp_path / "compressed",
        checkpoint_db=tmp_path / "state" / "compression.sqlite3",
        sample_size=24,
        pca_dim=8,
        n_centroids=4,
        n_bits=2,
        random_seed=0,
    )

    index_path = run_compression_pipeline(config)

    assert index_path == config.faiss_index_path
    assert config.compressor_dir.exists()
    assert config.checkpoint_db.exists()
    assert config.faiss_index_path.exists()

    shard_paths = sorted(config.shard_dir.glob("*.widx"))
    assert [path.name for path in shard_paths] == ["shard_00000000.widx", "shard_00000001.widx"]

    with sqlite3.connect(config.checkpoint_db) as connection:
        checkpoint_rows = dict(
            connection.execute(
                """
                SELECT checkpoint_key, checkpoint_value
                FROM compression_checkpoint
                """
            ).fetchall()
        )
    assert checkpoint_rows == {
        "compression:training_complete": "1",
        "compression:last_completed_shard_id": "1",
        "compression:indexed_shard_id": "1",
    }

    recovered_clip = read_clip_from_shard(shard_paths[1], 0)
    assert recovered_clip.centroid_ids.shape == (12,)
    assert recovered_clip.quantized_residuals.shape == (12, 2)
    assert recovered_clip.coarse_vector.shape == (8,)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("completed shards should not be rewritten on rerun")

    monkeypatch.setattr("compression.pipeline.write_compressed_shard", fail_if_called)
    second_index_path = run_compression_pipeline(config)

    consolidated = pl.read_parquet(config.consolidated_metadata_path)
    assert second_index_path == config.faiss_index_path
    assert consolidated["clip_index"].to_list() == [0, 1, 2]
    assert consolidated["shard_id"].to_list() == [0, 0, 1]
    assert consolidated["shard_offset"].to_list() == [0, 1, 0]
