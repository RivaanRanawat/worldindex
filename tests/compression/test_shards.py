from __future__ import annotations

from pathlib import Path

import numpy as np

from compression import (
    CompressedClip,
    read_clip_from_shard,
    read_shard_header,
    shard_record_size,
    write_compressed_shard,
)


def _make_clip(value: int) -> CompressedClip:
    return CompressedClip(
        centroid_ids=np.full(8, value, dtype=np.uint16),
        quantized_residuals=np.full((8, 2), value, dtype=np.uint8),
        coarse_vector=np.full(8, float(value), dtype=np.float32),
    )


def test_write_and_read_compressed_shard_support_random_access(tmp_path: Path) -> None:
    clips = [_make_clip(1), _make_clip(2), _make_clip(3)]
    shard_path = tmp_path / "demo.widx"

    write_compressed_shard(clips, shard_path)

    header = read_shard_header(shard_path)
    middle_clip = read_clip_from_shard(shard_path, 1)
    expected_record_size = shard_record_size(
        token_count=8,
        residual_bytes_per_token=2,
        coarse_dim=8,
    )

    assert header.clip_count == 3
    assert header.record_size == expected_record_size
    assert shard_path.stat().st_size == 32 + (header.clip_count * expected_record_size)
    np.testing.assert_array_equal(middle_clip.centroid_ids, clips[1].centroid_ids)
    np.testing.assert_array_equal(middle_clip.quantized_residuals, clips[1].quantized_residuals)
    np.testing.assert_array_equal(middle_clip.coarse_vector, clips[1].coarse_vector)
