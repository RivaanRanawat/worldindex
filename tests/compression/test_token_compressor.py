from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")
pytest.importorskip("sklearn")

from compression import TokenCompressor


def _make_cluster_source(input_dim: int, cluster_count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(4, input_dim)).astype(np.float32)
    weights = rng.normal(size=(cluster_count, 4)).astype(np.float32)
    return np.ascontiguousarray(weights @ basis)


def _sample_clustered_tokens(
    centers: np.ndarray,
    row_count: int,
    input_dim: int,
    *,
    seed: int,
    noise_scale: float = 0.01,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, centers.shape[0], size=row_count)
    noise = rng.normal(scale=noise_scale, size=(row_count, input_dim)).astype(np.float32)
    return np.ascontiguousarray(centers[assignments] + noise)


def _make_clip_batch(
    centers: np.ndarray,
    clip_count: int,
    tokens_per_clip: int,
    input_dim: int,
    *,
    seed: int,
    noise_scale: float = 0.01,
) -> np.ndarray:
    return _sample_clustered_tokens(
        centers,
        clip_count * tokens_per_clip,
        input_dim,
        seed=seed,
        noise_scale=noise_scale,
    ).reshape(clip_count, tokens_per_clip, input_dim)


def _average_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    return float(np.mean(numerator / denominator))


def test_compression_quality_exceeds_target_for_projected_tokens() -> None:
    centers = _make_cluster_source(64, 8, seed=0)
    training_tokens = _sample_clustered_tokens(centers, 1_024, 64, seed=1)
    evaluation_clips = _make_clip_batch(centers, 100, 16, 64, seed=2)
    compressor = TokenCompressor(pca_dim=8, n_centroids=8)

    compressor.train(training_tokens)

    similarities: list[float] = []
    for raw_clip in evaluation_clips:
        compressed = compressor.compress_clip(raw_clip)
        decompressed = compressor.decompress_clip(compressed)
        projected = (raw_clip - compressor.pca_mean) @ compressor.pca_components.T
        similarities.append(_average_cosine_similarity(projected, decompressed))

    assert float(np.mean(similarities)) > 0.97


def test_save_load_and_roundtrip_keep_error_bounded(tmp_path: Path) -> None:
    centers = _make_cluster_source(64, 8, seed=3)
    training_tokens = _sample_clustered_tokens(centers, 768, 64, seed=4)
    raw_clip = _make_clip_batch(centers, 1, 32, 64, seed=5)[0]
    compressor = TokenCompressor(pca_dim=8, n_centroids=8)

    compressor.train(training_tokens)
    compressed = compressor.compress_clip(raw_clip)
    decompressed = compressor.decompress_clip(compressed)

    projected = (raw_clip - compressor.pca_mean) @ compressor.pca_components.T
    mean_absolute_error = float(np.mean(np.abs(projected - decompressed)))
    assert compressed.centroid_ids.dtype == np.uint16
    assert compressed.quantized_residuals.dtype == np.uint8
    assert compressed.quantized_residuals.shape == (32, 2)
    assert compressed.coarse_vector.dtype == np.float32
    assert mean_absolute_error < 0.1

    save_dir = tmp_path / "compressor"
    compressor.save(save_dir)
    reloaded = TokenCompressor.load(save_dir)
    reloaded_decompressed = reloaded.decompress_clip(compressed)

    np.testing.assert_allclose(reloaded_decompressed, decompressed, atol=1e-6)


def test_default_pca_falls_back_to_256_when_variance_retention_is_too_low() -> None:
    rng = np.random.default_rng(4)
    training_tokens = rng.normal(size=(260, 300)).astype(np.float32)
    compressor = TokenCompressor(n_centroids=32)

    compressor.train(training_tokens)

    assert compressor.pca_dim == 256
