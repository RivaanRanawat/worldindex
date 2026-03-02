from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from compression import CompressedClip, write_compressed_shard
from index import IndexBuilder


def _make_dummy_clip(coarse_vector: np.ndarray) -> CompressedClip:
    return CompressedClip(
        centroid_ids=np.zeros(8, dtype=np.uint16),
        quantized_residuals=np.zeros((8, 2), dtype=np.uint8),
        coarse_vector=np.asarray(coarse_vector, dtype=np.float32),
    )


def test_build_faiss_index_and_validate_recall(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rng = np.random.default_rng(0)

    centers = rng.normal(size=(20, 128)).astype(np.float32)
    coarse_vectors = np.concatenate(
        [
            centers[cluster_index] + rng.normal(scale=0.01, size=(10, 128)).astype(np.float32)
            for cluster_index in range(centers.shape[0])
        ],
        axis=0,
    )

    first_shard = [_make_dummy_clip(vector) for vector in coarse_vectors[:100]]
    second_shard = [_make_dummy_clip(vector) for vector in coarse_vectors[100:]]
    write_compressed_shard(first_shard, shard_dir / "shard_00000000.widx")
    write_compressed_shard(second_shard, shard_dir / "shard_00000001.widx")

    builder = IndexBuilder(ef_search=256)
    index_path = tmp_path / "coarse_hnsw.faiss"

    recovered_vectors = builder.build_faiss_index(shard_dir, index_path)
    recall = builder.validate_index(index_path, recovered_vectors)

    assert index_path.exists()
    assert recovered_vectors.shape == (200, 128)
    assert recall > 0.95
