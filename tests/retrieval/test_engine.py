from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

pytest.importorskip("faiss")

from compression import TokenCompressor, write_compressed_shard
from index import IndexBuilder
from retrieval import BoundingBox, QueryEncoderConfig, RetrievalConfig, RetrievalEngine


class _StubQueryEncoder:
    def __init__(self, encoded_by_brightness: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        self._encoded_by_brightness = encoded_by_brightness

    def encode_image(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        brightness = int(np.asarray(image, dtype=np.uint8).mean())
        return self._encoded_by_brightness[brightness]


def test_retrieval_engine_supports_image_spatial_trajectory_and_transition_queries(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    compressor_dir = tmp_path / "compressor"
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()

    rng = np.random.default_rng(3)
    training_tokens = rng.normal(size=(4_096, 4)).astype(np.float32)
    compressor = TokenCompressor(pca_dim=4, n_centroids=32)
    compressor.train(training_tokens)
    compressor.save(compressor_dir)

    base_patch = np.tile(np.asarray([[0.0, 2.0, 0.0, 0.0]], dtype=np.float32), (256, 1))
    clip_transition_start = base_patch.copy()
    clip_transition_start[0] = np.asarray([6.0, 0.0, 0.0, 0.0], dtype=np.float32)

    clip_transition_end = base_patch.copy()
    clip_transition_end[0] = np.asarray([0.0, 0.0, 6.0, 0.0], dtype=np.float32)

    clip_similar = np.tile(np.asarray([[0.0, 0.0, 4.0, 0.0]], dtype=np.float32), (256, 1))
    clip_far = np.tile(np.asarray([[-4.0, 0.0, 0.0, 0.0]], dtype=np.float32), (256, 1))

    compressed_clips = [
        compressor.compress_clip(clip_transition_start),
        compressor.compress_clip(clip_transition_end),
        compressor.compress_clip(clip_similar),
        compressor.compress_clip(clip_far),
    ]
    write_compressed_shard(compressed_clips, shard_dir / "shard_00000000.widx")

    metadata = pl.DataFrame(
        {
            "episode_id": [
                "episode_transition",
                "episode_transition",
                "episode_similar",
                "episode_far",
            ],
            "dataset_name": ["droid", "droid", "droid", "droid"],
            "robot_type": ["franka", "franka", "franka", "franka"],
            "timestamp_start": [0.0, 2.0, 0.0, 0.0],
            "timestamp_end": [1.0, 3.0, 1.0, 1.0],
            "language_instruction": ["pick", "place", "stir", "push"],
            "shard_id": [0, 0, 0, 0],
            "shard_offset": [0, 1, 2, 3],
        }
    )
    metadata_path = tmp_path / "clips.parquet"
    metadata.write_parquet(metadata_path)

    index_path = tmp_path / "coarse.faiss"
    IndexBuilder(ef_search=64).build_faiss_index(shard_dir, index_path)

    config = RetrievalConfig(
        faiss_index_path=index_path,
        metadata_path=metadata_path,
        shard_dir=shard_dir,
        compressor_dir=compressor_dir,
        query_encoder=QueryEncoderConfig(
            pca_artifact_dir=compressor_dir,
            spatial_grid_size=16,
        ),
        trajectory_embedding_dir=trajectory_dir,
        maxsim_candidate_batch_size=2,
        maxsim_use_gpu=False,
        trajectory_frame_search_k=16,
        coarse_search_k=4,
    )

    engine = RetrievalEngine(
        config,
        query_encoder=_StubQueryEncoder(
            {
                0: (
                    compressor.decompress_clip(compressed_clips[0])[:256],
                    compressed_clips[0].coarse_vector,
                ),
                255: (
                    compressor.decompress_clip(compressed_clips[1])[:256],
                    compressed_clips[1].coarse_vector,
                ),
            }
        ),
    )

    query_sequence = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    similar_sequence = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    far_sequence = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.save(trajectory_dir / "episode_query.npy", query_sequence, allow_pickle=False)
    np.save(trajectory_dir / "episode_similar.npy", similar_sequence, allow_pickle=False)
    np.save(trajectory_dir / "episode_far.npy", far_sequence, allow_pickle=False)

    dark_image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    bright_image = Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8))

    image_results = engine.search_image(dark_image, top_k=2, coarse_k=4)
    assert image_results[0].clip_id == 0

    spatial_results = engine.search_spatial(
        dark_image,
        BoundingBox(row_start=0, row_end=0, col_start=0, col_end=0),
        top_k=2,
    )
    assert spatial_results[0].clip_id == 0

    transition_results = engine.search_transition(dark_image, bright_image, top_k=1, max_gap_sec=5.0)
    assert transition_results[0].clip_id == 1
    assert transition_results[0].episode_id == "episode_transition"

    trajectory_results = engine.search_trajectory("episode_query", top_k=2)
    assert trajectory_results[0].episode_id == "episode_similar"
    assert trajectory_results[0].dtw_distance < trajectory_results[1].dtw_distance
    assert trajectory_results[0].alignment_path
