from __future__ import annotations

import multiprocessing
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from extraction.pipeline import (
    ExtractionConfig,
    SharedClipDescriptor,
    _read_checkpoint,
    encoder_fn,
    producer_fn,
    run_extraction,
    writer_fn,
)
from ingestion.config import DatasetConfig
from tests.extraction.spawn_fakes import DEFAULT_CLIP_SHAPE, FakeClipFormer, FakeVisionModel

_DEFAULT_TOKEN_SHAPE = (8, 4)


def _build_config(repo_id: str = "tests/demo") -> DatasetConfig:
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


def _mp_context() -> Any:
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _enqueue_shared_memory_clip(
    queue: Any,
    clip: np.ndarray,
    metadata: dict[str, Any],
) -> SharedClipDescriptor:
    shared_memory = SharedMemory(create=True, size=clip.nbytes)
    shared_array = np.ndarray(clip.shape, dtype=clip.dtype, buffer=shared_memory.buf)
    shared_array[...] = clip
    descriptor = SharedClipDescriptor(
        shm_name=shared_memory.name,
        shape=tuple(clip.shape),
        dtype=str(clip.dtype),
        metadata=metadata,
    )
    queue.put(descriptor)
    resource_tracker.unregister(shared_memory._name, "shared_memory")
    shared_memory.close()
    return descriptor


def _load_metadata_frames(output_dir: Path) -> pl.DataFrame:
    metadata_paths = sorted(output_dir.glob("metadata_*.parquet"))
    return pl.concat([pl.read_parquet(path) for path in metadata_paths], how="vertical")


def test_producer_fn_skips_resume_offset_and_uses_shared_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = _mp_context().Queue()
    config = _build_config()

    monkeypatch.setattr(
        "extraction.pipeline._build_clip_former",
        lambda current_config, clip_former_factory=None: FakeClipFormer(current_config, clip_count=4),
    )

    producer_fn([config], queue, start_clip_index=2)

    clip_indices: list[int] = []
    descriptors: list[SharedClipDescriptor] = []
    while True:
        item = queue.get()
        if item is None:
            break
        descriptor = SharedClipDescriptor(*item)
        descriptors.append(descriptor)
        clip_indices.append(int(descriptor.metadata["clip_index"]))

    assert clip_indices == [2, 3]

    for descriptor in descriptors:
        shared_memory = SharedMemory(name=descriptor.shm_name)
        shared_array = np.ndarray(
            descriptor.shape,
            dtype=np.dtype(descriptor.dtype),
            buffer=shared_memory.buf,
        )
        assert shared_array.shape == DEFAULT_CLIP_SHAPE
        shared_memory.close()
        shared_memory.unlink()

    queue.close()
    queue.join_thread()


def test_encoder_fn_unlinks_shared_memory_and_upcasts_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    q1 = _mp_context().Queue()
    q2 = _mp_context().Queue()

    descriptor_a = _enqueue_shared_memory_clip(
        q1,
        np.ones(DEFAULT_CLIP_SHAPE, dtype=np.float32),
        {"clip_index": 0, "dataset_name": "demo"},
    )
    descriptor_b = _enqueue_shared_memory_clip(
        q1,
        np.full(DEFAULT_CLIP_SHAPE, fill_value=2.0, dtype=np.float32),
        {"clip_index": 1, "dataset_name": "demo"},
    )
    q1.put(None)

    monkeypatch.setattr(
        "extraction.pipeline._load_encoder_model",
        lambda model_id, device, model_loader=None: FakeVisionModel(token_shape=(16, 8)),
    )

    encoder_fn(q1, q2, model_id="fake", device="cpu", batch_size=2)

    encoded_batches: list[tuple[np.ndarray, dict[str, Any]]] = []
    while True:
        item = q2.get()
        if item is None:
            break
        encoded_batches.append(item)

    assert len(encoded_batches) == 2
    assert all(tokens.dtype == np.float32 for tokens, _ in encoded_batches)
    assert all(tokens.shape == (16, 8) for tokens, _ in encoded_batches)
    assert [metadata["clip_index"] for _, metadata in encoded_batches] == [0, 1]

    with pytest.raises(FileNotFoundError):
        SharedMemory(name=descriptor_a.shm_name)
    with pytest.raises(FileNotFoundError):
        SharedMemory(name=descriptor_b.shm_name)

    q1.close()
    q2.close()
    q1.join_thread()
    q2.join_thread()


def test_writer_fn_flushes_remaining_buffer_and_updates_checkpoint(tmp_path: Path) -> None:
    q2 = _mp_context().Queue()
    output_dir = tmp_path / "artifacts"
    checkpoint_db = tmp_path / "checkpoint.sqlite3"

    for clip_index in range(3):
        q2.put(
            (
                np.full(_DEFAULT_TOKEN_SHAPE, fill_value=float(clip_index), dtype=np.float16),
                {
                    "clip_index": clip_index,
                    "episode_id": "demo_episode_0",
                    "dataset_name": "demo",
                    "robot_type": "testbot",
                    "clip_start_frame": clip_index * 4,
                    "clip_end_frame": clip_index * 4 + 3,
                    "timestamp_start": float(clip_index),
                    "timestamp_end": float(clip_index) + 0.75,
                    "language_instruction": f"task_{clip_index}",
                    "num_original_frames": 4,
                },
            )
        )
    q2.put(None)

    writer_fn(q2, output_dir, checkpoint_db, flush_size=2)

    token_paths = sorted(output_dir.glob("tokens_*.npy"))
    metadata_paths = sorted(output_dir.glob("metadata_*.parquet"))
    assert [path.name for path in token_paths] == [
        "tokens_00000000_00000001.npy",
        "tokens_00000002_00000002.npy",
    ]
    assert [path.name for path in metadata_paths] == [
        "metadata_00000000_00000001.parquet",
        "metadata_00000002_00000002.parquet",
    ]

    first_tokens = np.load(token_paths[0])
    second_tokens = np.load(token_paths[1])
    assert first_tokens.shape == (2, *_DEFAULT_TOKEN_SHAPE)
    assert second_tokens.shape == (1, *_DEFAULT_TOKEN_SHAPE)
    assert first_tokens.dtype == np.float32
    assert second_tokens.dtype == np.float32

    metadata_frame = _load_metadata_frames(output_dir)
    assert metadata_frame.columns == [
        "clip_index",
        "episode_id",
        "dataset_name",
        "robot_type",
        "clip_start_frame",
        "clip_end_frame",
        "timestamp_start",
        "timestamp_end",
        "language_instruction",
        "num_original_frames",
    ]
    assert metadata_frame["clip_index"].to_list() == [0, 1, 2]
    assert _read_checkpoint(checkpoint_db) == 2

    q2.close()
    q2.join_thread()


def test_run_extraction_writes_expected_batches_end_to_end(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")

    output_dir = tmp_path / "output"
    checkpoint_db = tmp_path / "checkpoint.sqlite3"
    extraction_config = ExtractionConfig(
        dataset_configs=[_build_config("tests/alpha__4"), _build_config("tests/bravo__3")],
        model_id="fake:6x5",
        device="cpu",
        batch_size=2,
        queue_depth=2,
        flush_size=3,
        clip_former_factory="tests.extraction.spawn_fakes:build_fake_clip_former",
        model_loader="tests.extraction.spawn_fakes:load_fake_encoder_model",
        output_dir=output_dir,
        checkpoint_db=checkpoint_db,
    )

    final_checkpoint = run_extraction(extraction_config)

    assert final_checkpoint == 6
    token_paths = sorted(output_dir.glob("tokens_*.npy"))
    assert [path.name for path in token_paths] == [
        "tokens_00000000_00000002.npy",
        "tokens_00000003_00000005.npy",
        "tokens_00000006_00000006.npy",
    ]
    assert np.load(token_paths[0]).shape == (3, 6, 5)
    assert np.load(token_paths[1]).shape == (3, 6, 5)
    assert np.load(token_paths[2]).shape == (1, 6, 5)

    metadata_frame = _load_metadata_frames(output_dir)
    assert metadata_frame["clip_index"].to_list() == list(range(7))
    assert metadata_frame["dataset_name"].to_list() == [
        "alpha",
        "alpha",
        "alpha",
        "alpha",
        "bravo",
        "bravo",
        "bravo",
    ]


def test_run_extraction_resumes_without_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")

    output_dir = tmp_path / "output"
    checkpoint_db = tmp_path / "checkpoint.sqlite3"
    extraction_config = ExtractionConfig(
        dataset_configs=[_build_config("tests/resume__5")],
        model_id="fake:4x3",
        device="cpu",
        batch_size=2,
        queue_depth=2,
        flush_size=2,
        clip_former_factory="tests.extraction.spawn_fakes:build_fake_clip_former",
        model_loader="tests.extraction.spawn_fakes:load_fake_encoder_model",
        output_dir=output_dir,
        checkpoint_db=checkpoint_db,
    )

    assert run_extraction(extraction_config) == 4

    resumed_config = extraction_config.model_copy(
        update={
            "dataset_configs": [_build_config("tests/resume__8")],
        }
    )
    assert run_extraction(resumed_config) == 7

    token_paths = sorted(output_dir.glob("tokens_*.npy"))
    assert [path.name for path in token_paths] == [
        "tokens_00000000_00000001.npy",
        "tokens_00000002_00000003.npy",
        "tokens_00000004_00000004.npy",
        "tokens_00000005_00000006.npy",
        "tokens_00000007_00000007.npy",
    ]

    metadata_frame = _load_metadata_frames(output_dir)
    clip_indices = metadata_frame["clip_index"].to_list()
    assert clip_indices == list(range(8))
    assert len(clip_indices) == len(set(clip_indices))
