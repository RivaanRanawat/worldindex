from pathlib import Path

import numpy as np

from compression import CompressedClip, write_compressed_shard
from retrieval import CompressedShardReader


def _make_clip(value: int) -> CompressedClip:
    return CompressedClip(
        centroid_ids=np.full(4, value, dtype=np.uint16),
        quantized_residuals=np.full((4, 1), value, dtype=np.uint8),
        coarse_vector=np.full(4, float(value), dtype=np.float32),
    )


def test_shard_reader_caches_open_memmaps_by_shard_id(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    write_compressed_shard([_make_clip(1), _make_clip(2)], shard_dir / "shard_00000000.widx")
    write_compressed_shard([_make_clip(3)], shard_dir / "shard_00000001.widx")

    reader = CompressedShardReader(shard_dir)

    first_clip = reader.get_clip(0, 0)
    second_clip = reader.get_clip(0, 1)
    assert reader.open_shard_ids == (0,)

    third_clip = reader.get_clip(1, 0)
    assert reader.open_shard_ids == (0, 1)

    np.testing.assert_array_equal(first_clip.centroid_ids, np.full(4, 1, dtype=np.uint16))
    np.testing.assert_array_equal(second_clip.centroid_ids, np.full(4, 2, dtype=np.uint16))
    np.testing.assert_array_equal(third_clip.centroid_ids, np.full(4, 3, dtype=np.uint16))
