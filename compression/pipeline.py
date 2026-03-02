from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field

from compression.shards import write_compressed_shard
from compression.token_compressor import TokenCompressor


class CompressionPipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_dir: Path
    output_dir: Path
    sample_size: int = Field(default=500_000, gt=0)
    pca_dim: int = Field(default=128, gt=0)
    n_centroids: int = Field(default=32768, gt=0, le=np.iinfo(np.uint16).max)
    n_bits: int = Field(default=2, gt=0)
    random_seed: int = 0

    @property
    def compressor_dir(self) -> Path:
        return self.output_dir / "compression_model"

    @property
    def shard_dir(self) -> Path:
        return self.output_dir / "shards"

    @property
    def shard_metadata_dir(self) -> Path:
        return self.output_dir / "metadata"

    @property
    def consolidated_metadata_path(self) -> Path:
        return self.output_dir / "clips.parquet"

    @classmethod
    def from_yaml(cls, config_path: Path) -> CompressionPipelineConfig:
        raw_config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw_config, dict):
            raise TypeError(f"Expected mapping in {config_path}, found {type(raw_config).__name__}")
        return cls.model_validate(raw_config)


class RawBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_path: Path
    metadata_path: Path
    clip_count: int = Field(gt=0)
    token_count: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)


def discover_raw_batches(raw_dir: Path) -> list[RawBatch]:
    token_paths = sorted(path for path in raw_dir.glob("tokens_*.npy") if path.is_file())
    batches: list[RawBatch] = []

    for token_path in token_paths:
        metadata_path = raw_dir / token_path.name.replace("tokens_", "metadata_").replace(".npy", ".parquet")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file for token batch {token_path}")

        token_batch = np.load(token_path, mmap_mode="r")
        if token_batch.ndim != 3:
            raise ValueError(f"Expected 3D token batch in {token_path}, found shape {token_batch.shape}")

        clip_count, tokens_per_clip, embedding_dim = token_batch.shape
        batches.append(
            RawBatch(
                token_path=token_path,
                metadata_path=metadata_path,
                clip_count=int(clip_count),
                token_count=int(clip_count * tokens_per_clip),
                embedding_dim=int(embedding_dim),
            )
        )

    return batches


def sample_training_tokens(
    raw_batches: list[RawBatch],
    sample_size: int,
    random_seed: int = 0,
) -> np.ndarray:
    if not raw_batches:
        raise ValueError("raw_batches must contain at least one batch")

    total_tokens = sum(batch.token_count for batch in raw_batches)
    target_sample_size = min(sample_size, total_tokens)
    expected_counts = np.asarray(
        [target_sample_size * batch.token_count / total_tokens for batch in raw_batches],
        dtype=np.float64,
    )
    base_counts = np.floor(expected_counts).astype(np.int64)
    remainder = int(target_sample_size - int(base_counts.sum()))

    if remainder > 0:
        fractional_order = np.argsort(-(expected_counts - base_counts))
        for batch_index in fractional_order[:remainder]:
            base_counts[batch_index] += 1

    rng = np.random.default_rng(random_seed)
    sample_parts: list[np.ndarray] = []

    for batch, sample_count in zip(raw_batches, base_counts.tolist(), strict=True):
        if sample_count == 0:
            continue

        token_batch = np.load(batch.token_path, mmap_mode="r")
        flattened = token_batch.reshape(batch.token_count, batch.embedding_dim)
        sample_indices = rng.choice(batch.token_count, size=sample_count, replace=False)
        sample_parts.append(np.asarray(flattened[sample_indices], dtype=np.float32))

    return np.ascontiguousarray(np.concatenate(sample_parts, axis=0).astype(np.float32, copy=False))


def run_compression_pipeline(config: CompressionPipelineConfig) -> Path:
    logger = structlog.get_logger(__name__).bind(component="compression_pipeline")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.shard_dir.mkdir(parents=True, exist_ok=True)
    config.shard_metadata_dir.mkdir(parents=True, exist_ok=True)

    raw_batches = discover_raw_batches(config.raw_dir)
    if not raw_batches:
        raise FileNotFoundError(f"No extraction token batches found in {config.raw_dir}")

    if config.compressor_dir.exists():
        compressor = TokenCompressor.load(config.compressor_dir)
        logger.info("loaded_existing_compressor", compressor_dir=str(config.compressor_dir))
    else:
        sampled_tokens = sample_training_tokens(raw_batches, config.sample_size, config.random_seed)
        compressor = TokenCompressor(
            pca_dim=config.pca_dim,
            n_centroids=config.n_centroids,
            n_bits=config.n_bits,
        )
        compressor.train(sampled_tokens)
        compressor.save(config.compressor_dir)
        logger.info("trained_compressor", compressor_dir=str(config.compressor_dir))

    for shard_id, raw_batch in enumerate(raw_batches):
        shard_path = config.shard_dir / f"shard_{shard_id:08d}.widx"
        shard_metadata_path = config.shard_metadata_dir / f"shard_{shard_id:08d}.parquet"

        token_batch = np.load(raw_batch.token_path, mmap_mode="r")
        metadata_frame = pl.read_parquet(raw_batch.metadata_path)
        if metadata_frame.height != raw_batch.clip_count:
            raise ValueError(
                f"metadata rows ({metadata_frame.height}) do not match clip count ({raw_batch.clip_count})"
            )

        compressed_clips = [
            compressor.compress_clip(np.asarray(token_batch[clip_index], dtype=np.float32))
            for clip_index in range(raw_batch.clip_count)
        ]
        write_compressed_shard(compressed_clips, shard_path)

        shard_offsets = pl.Series("shard_offset", np.arange(raw_batch.clip_count, dtype=np.int64))
        enriched_metadata = metadata_frame.with_columns(
            pl.lit(shard_id, dtype=pl.Int64).alias("shard_id"),
            shard_offsets,
        )
        _write_parquet_atomically(enriched_metadata, shard_metadata_path)
        logger.info(
            "compressed_shard_written",
            shard_id=shard_id,
            clip_count=raw_batch.clip_count,
            shard_path=str(shard_path),
        )

    _consolidate_metadata(config)

    logger.info(
        "compression_pipeline_complete",
        shard_count=len(raw_batches),
        metadata_path=str(config.consolidated_metadata_path),
    )
    return config.consolidated_metadata_path


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: poetry run python scripts/compress.py <config.yaml>")

    config = CompressionPipelineConfig.from_yaml(Path(args[0]))
    metadata_path = run_compression_pipeline(config)
    print({"metadata_path": str(metadata_path)})


def _consolidate_metadata(config: CompressionPipelineConfig) -> None:
    metadata_paths = sorted(
        path for path in config.shard_metadata_dir.glob("shard_*.parquet") if path.is_file()
    )
    if not metadata_paths:
        raise FileNotFoundError(f"No shard metadata found in {config.shard_metadata_dir}")

    consolidated = pl.concat([pl.read_parquet(path) for path in metadata_paths], how="vertical")
    _write_parquet_atomically(consolidated, config.consolidated_metadata_path)


def _write_parquet_atomically(frame: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    frame.write_parquet(temp_output_path)
    temp_output_path.replace(output_path)
