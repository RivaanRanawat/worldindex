from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from compression.models import CompressedClip, ShardHeader

SHARD_MAGIC = b"WIDXSHD1"
SHARD_SUFFIX = ".widx"
_SHARD_HEADER_STRUCT = struct.Struct("<8s6I")


def shard_record_size(token_count: int, residual_bytes_per_token: int, coarse_dim: int) -> int:
    return (
        token_count * np.dtype(np.uint16).itemsize
        + token_count * residual_bytes_per_token
        + coarse_dim * np.dtype(np.float32).itemsize
    )


def iter_shard_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob(f"*{SHARD_SUFFIX}") if path.is_file())


def write_compressed_shard(clips: list[CompressedClip], output_path: Path) -> None:
    if not clips:
        raise ValueError("clips must contain at least one CompressedClip")

    token_count = int(clips[0].centroid_ids.shape[0])
    residual_bytes_per_token = int(clips[0].quantized_residuals.shape[1])
    pca_dim = residual_bytes_per_token * 4
    coarse_dim = int(clips[0].coarse_vector.shape[0])
    record_size = shard_record_size(token_count, residual_bytes_per_token, coarse_dim)
    header = ShardHeader(
        clip_count=len(clips),
        token_count=token_count,
        pca_dim=pca_dim,
        residual_bytes_per_token=residual_bytes_per_token,
        coarse_dim=coarse_dim,
        record_size=record_size,
    )

    for clip in clips:
        if clip.centroid_ids.shape != (token_count,):
            raise ValueError("all clips in a shard must share the same token_count")
        if clip.quantized_residuals.shape != (token_count, residual_bytes_per_token):
            raise ValueError("all clips in a shard must share the same packed residual shape")
        if clip.coarse_vector.shape != (coarse_dim,):
            raise ValueError("all clips in a shard must share the same coarse vector width")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

    with temp_output_path.open("wb") as shard_file:
        shard_file.write(
            _SHARD_HEADER_STRUCT.pack(
                SHARD_MAGIC,
                header.clip_count,
                header.token_count,
                header.pca_dim,
                header.residual_bytes_per_token,
                header.coarse_dim,
                header.record_size,
            )
        )
        for clip in clips:
            shard_file.write(np.asarray(clip.centroid_ids, dtype=np.uint16).tobytes(order="C"))
            shard_file.write(np.asarray(clip.quantized_residuals, dtype=np.uint8).tobytes(order="C"))
            shard_file.write(np.asarray(clip.coarse_vector, dtype=np.float32).tobytes(order="C"))

    temp_output_path.replace(output_path)


def read_shard_header(shard_path: Path) -> ShardHeader:
    with shard_path.open("rb") as shard_file:
        raw_header = shard_file.read(_SHARD_HEADER_STRUCT.size)

    if len(raw_header) != _SHARD_HEADER_STRUCT.size:
        raise ValueError(f"{shard_path} is too small to contain a valid shard header")

    magic, clip_count, token_count, pca_dim, residual_bytes_per_token, coarse_dim, record_size = (
        _SHARD_HEADER_STRUCT.unpack(raw_header)
    )
    if magic != SHARD_MAGIC:
        raise ValueError(f"{shard_path} does not contain a WorldIndex shard")

    return ShardHeader(
        clip_count=clip_count,
        token_count=token_count,
        pca_dim=pca_dim,
        residual_bytes_per_token=residual_bytes_per_token,
        coarse_dim=coarse_dim,
        record_size=record_size,
    )


def read_clip_from_shard(shard_path: Path, clip_index: int) -> CompressedClip:
    header = read_shard_header(shard_path)
    if clip_index < 0 or clip_index >= header.clip_count:
        raise IndexError(f"clip_index {clip_index} is out of range for shard {shard_path}")

    records = np.memmap(
        shard_path,
        mode="r",
        dtype=header.record_dtype(),
        offset=_SHARD_HEADER_STRUCT.size,
        shape=(header.clip_count,),
    )
    record = records[clip_index]
    return CompressedClip(
        centroid_ids=np.asarray(record["centroid_ids"]),
        quantized_residuals=np.asarray(record["quantized_residuals"]),
        coarse_vector=np.asarray(record["coarse_vector"]),
    )


def read_coarse_vectors_from_shard(shard_path: Path) -> np.ndarray:
    header = read_shard_header(shard_path)
    records = np.memmap(
        shard_path,
        mode="r",
        dtype=header.record_dtype(),
        offset=_SHARD_HEADER_STRUCT.size,
        shape=(header.clip_count,),
    )
    return np.asarray(records["coarse_vector"], dtype=np.float32)
