from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import structlog

from compression.models import CompressedClip

_PCA_COMPONENTS_FILENAME = "pca_components.npy"
_PCA_MEAN_FILENAME = "pca_mean.npy"
_CENTROIDS_FILENAME = "centroids.npy"
_QUANTILE_THRESHOLDS_FILENAME = "quantile_thresholds.npy"
_QUANTIZATION_LEVELS_FILENAME = "quantization_levels.npy"
_EXPLAINED_VARIANCE_FILENAME = "explained_variance_ratio.npy"
_INPUT_DIM_FILENAME = "input_dim.npy"


class TokenCompressor:
    def __init__(
        self,
        pca_dim: int = 128,
        n_centroids: int = 32768,
        n_bits: int = 2,
    ) -> None:
        if pca_dim <= 0:
            raise ValueError("pca_dim must be positive")
        if n_centroids <= 0:
            raise ValueError("n_centroids must be positive")
        if n_centroids > np.iinfo(np.uint16).max:
            raise ValueError("n_centroids must fit inside uint16 centroid ids")
        if n_bits != 2:
            raise ValueError("TokenCompressor currently supports only 2-bit residual quantization")

        self.pca_dim = pca_dim
        self.n_centroids = n_centroids
        self.n_bits = n_bits
        self.input_dim: int | None = None
        self.pca_components: np.ndarray | None = None
        self.pca_mean: np.ndarray | None = None
        self.centroids: np.ndarray | None = None
        self.quantile_thresholds: np.ndarray | None = None
        self.quantization_levels: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None
        self._centroid_index: Any | None = None
        self._logger = structlog.get_logger(__name__).bind(component="token_compressor")

    def train(self, sample_tokens: np.ndarray) -> None:
        from sklearn.cluster import MiniBatchKMeans

        training_tokens = np.asarray(sample_tokens, dtype=np.float32)
        if training_tokens.ndim != 2:
            raise ValueError("sample_tokens must be a 2D array")
        if training_tokens.shape[0] < 2:
            raise ValueError("sample_tokens must contain at least two rows")

        self.input_dim = int(training_tokens.shape[1])

        target_pca_dim = min(self.pca_dim, training_tokens.shape[0], training_tokens.shape[1])
        pca = self._fit_pca(training_tokens, target_pca_dim)
        explained_variance = float(np.sum(pca.explained_variance_ratio_))

        if (
            self.pca_dim == 128
            and target_pca_dim == 128
            and explained_variance < 0.95
            and min(training_tokens.shape[0], training_tokens.shape[1]) >= 256
        ):
            self._logger.info(
                "pca_variance_below_target",
                explained_variance=explained_variance,
                requested_pca_dim=128,
                fallback_pca_dim=256,
            )
            pca = self._fit_pca(training_tokens, 256)
            explained_variance = float(np.sum(pca.explained_variance_ratio_))

        self.pca_dim = int(pca.n_components_)
        self.pca_components = np.ascontiguousarray(pca.components_.astype(np.float32, copy=False))
        self.pca_mean = np.ascontiguousarray(pca.mean_.astype(np.float32, copy=False))
        self.explained_variance_ratio = np.ascontiguousarray(
            pca.explained_variance_ratio_.astype(np.float32, copy=False)
        )
        projected = self._project(training_tokens)

        effective_centroids = min(self.n_centroids, projected.shape[0])
        if effective_centroids != self.n_centroids:
            self._logger.info(
                "reducing_centroid_count",
                requested_centroids=self.n_centroids,
                effective_centroids=effective_centroids,
            )
            self.n_centroids = effective_centroids

        kmeans = MiniBatchKMeans(
            n_clusters=self.n_centroids,
            batch_size=min(10_000, projected.shape[0]),
            n_init="auto",
            random_state=0,
        )
        kmeans.fit(projected)

        self.centroids = np.ascontiguousarray(kmeans.cluster_centers_.astype(np.float32, copy=False))
        cluster_assignments = np.asarray(kmeans.predict(projected), dtype=np.intp)
        residuals = projected - self.centroids[cluster_assignments]
        self.quantile_thresholds = np.ascontiguousarray(
            np.percentile(residuals, [25, 50, 75], axis=0).astype(np.float32)
        )
        quantized = self._quantize_residuals(residuals)
        self.quantization_levels = self._compute_quantization_levels(residuals, quantized)
        self._centroid_index = None

        self._logger.info(
            "training_complete",
            input_dim=self.input_dim,
            pca_dim=self.pca_dim,
            explained_variance=explained_variance,
            n_centroids=self.n_centroids,
        )

    def compress_clip(self, raw_tokens: np.ndarray) -> CompressedClip:
        projected = self._project(raw_tokens)
        centroid_index = self._get_centroid_index()
        _, nearest_centroid_ids = centroid_index.search(projected, 1)
        centroid_ids = np.ascontiguousarray(nearest_centroid_ids[:, 0].astype(np.uint16, copy=False))
        centroids = self._require_centroids()[centroid_ids.astype(np.intp)]
        residuals = projected - centroids
        quantized_residuals = self._pack_2bit(self._quantize_residuals(residuals))
        coarse_vector = np.ascontiguousarray(projected.mean(axis=0, dtype=np.float32))

        return CompressedClip(
            centroid_ids=centroid_ids,
            quantized_residuals=quantized_residuals,
            coarse_vector=coarse_vector,
        )

    def decompress_clip(self, compressed: CompressedClip) -> np.ndarray:
        centroids = self._require_centroids()[compressed.centroid_ids.astype(np.intp)]
        quantized = self._unpack_2bit(compressed.quantized_residuals)
        dequantized_residuals = self._require_quantization_levels()[quantized, np.arange(self.pca_dim)]
        return np.ascontiguousarray((centroids + dequantized_residuals).astype(np.float32, copy=False))

    def save(self, directory: Path) -> None:
        self._require_trained()
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _PCA_COMPONENTS_FILENAME, self.pca_components, allow_pickle=False)
        np.save(directory / _PCA_MEAN_FILENAME, self.pca_mean, allow_pickle=False)
        np.save(directory / _CENTROIDS_FILENAME, self.centroids, allow_pickle=False)
        np.save(directory / _QUANTILE_THRESHOLDS_FILENAME, self.quantile_thresholds, allow_pickle=False)
        np.save(directory / _QUANTIZATION_LEVELS_FILENAME, self.quantization_levels, allow_pickle=False)
        np.save(directory / _EXPLAINED_VARIANCE_FILENAME, self.explained_variance_ratio, allow_pickle=False)
        np.save(
            directory / _INPUT_DIM_FILENAME,
            np.asarray([self.input_dim], dtype=np.int32),
            allow_pickle=False,
        )

    @classmethod
    def load(cls, directory: Path) -> TokenCompressor:
        pca_components = np.load(directory / _PCA_COMPONENTS_FILENAME)
        centroids = np.load(directory / _CENTROIDS_FILENAME)
        compressor = cls(
            pca_dim=int(pca_components.shape[0]),
            n_centroids=int(centroids.shape[0]),
            n_bits=2,
        )
        compressor.input_dim = int(np.load(directory / _INPUT_DIM_FILENAME)[0])
        compressor.pca_components = np.ascontiguousarray(pca_components.astype(np.float32, copy=False))
        compressor.pca_mean = np.ascontiguousarray(
            np.load(directory / _PCA_MEAN_FILENAME).astype(np.float32, copy=False)
        )
        compressor.centroids = np.ascontiguousarray(centroids.astype(np.float32, copy=False))
        compressor.quantile_thresholds = np.ascontiguousarray(
            np.load(directory / _QUANTILE_THRESHOLDS_FILENAME).astype(np.float32, copy=False)
        )
        compressor.quantization_levels = np.ascontiguousarray(
            np.load(directory / _QUANTIZATION_LEVELS_FILENAME).astype(np.float32, copy=False)
        )
        compressor.explained_variance_ratio = np.ascontiguousarray(
            np.load(directory / _EXPLAINED_VARIANCE_FILENAME).astype(np.float32, copy=False)
        )
        return compressor

    def _fit_pca(self, training_tokens: np.ndarray, n_components: int) -> Any:
        from sklearn.decomposition import PCA

        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=0,
        )
        pca.fit(training_tokens)
        return pca

    def _project(self, tokens: np.ndarray) -> np.ndarray:
        self._require_pca_ready()
        raw_tokens = np.asarray(tokens, dtype=np.float32)
        if raw_tokens.ndim != 2:
            raise ValueError("tokens must be a 2D array")
        if raw_tokens.shape[1] != self.input_dim:
            raise ValueError(
                f"expected tokens with width {self.input_dim}, received {raw_tokens.shape[1]}"
            )

        centered = raw_tokens - self._require_pca_mean()
        projected = centered @ self._require_pca_components().T
        return np.ascontiguousarray(projected.astype(np.float32, copy=False))

    def _quantize_residuals(self, residuals: np.ndarray) -> np.ndarray:
        thresholds = self._require_quantile_thresholds()
        clipped = np.asarray(residuals, dtype=np.float32)
        quantized = (clipped > thresholds[0]).astype(np.uint8)
        quantized += (clipped > thresholds[1]).astype(np.uint8)
        quantized += (clipped > thresholds[2]).astype(np.uint8)
        return np.ascontiguousarray(quantized)

    def _compute_quantization_levels(
        self,
        residuals: np.ndarray,
        quantized: np.ndarray,
    ) -> np.ndarray:
        thresholds = self._require_quantile_thresholds()
        levels = np.empty((1 << self.n_bits, self.pca_dim), dtype=np.float32)

        for dim in range(self.pca_dim):
            dim_residuals = residuals[:, dim]
            dim_quantized = quantized[:, dim]
            for level in range(levels.shape[0]):
                bucket_values = dim_residuals[dim_quantized == level]
                if bucket_values.size == 0:
                    levels[level, dim] = self._fallback_bucket_value(thresholds[:, dim], level)
                    continue
                levels[level, dim] = float(np.median(bucket_values))

        return np.ascontiguousarray(levels)

    def _fallback_bucket_value(self, thresholds: np.ndarray, level: int) -> float:
        if level == 0:
            return float(thresholds[0])
        if level == 1:
            return float((thresholds[0] + thresholds[1]) / 2.0)
        if level == 2:
            return float((thresholds[1] + thresholds[2]) / 2.0)
        return float(thresholds[2])

    def _get_centroid_index(self) -> Any:
        import faiss

        if self._centroid_index is None:
            centroid_index = faiss.IndexFlatL2(self.pca_dim)
            centroid_index.add(self._require_centroids())
            self._centroid_index = centroid_index
        return self._centroid_index

    def _pack_2bit(self, values: np.ndarray) -> np.ndarray:
        quantized = np.asarray(values, dtype=np.uint8)
        if quantized.ndim != 2:
            raise ValueError("values must be a 2D uint8 array")
        if quantized.shape[1] % 4 != 0:
            raise ValueError("the projected token width must be divisible by 4 for 2-bit packing")

        grouped = quantized.reshape(quantized.shape[0], -1, 4)
        packed = (
            grouped[:, :, 0]
            | (grouped[:, :, 1] << 2)
            | (grouped[:, :, 2] << 4)
            | (grouped[:, :, 3] << 6)
        )
        return np.ascontiguousarray(packed.astype(np.uint8, copy=False))

    def _unpack_2bit(self, packed: np.ndarray) -> np.ndarray:
        packed_residuals = np.asarray(packed, dtype=np.uint8)
        if packed_residuals.ndim != 2:
            raise ValueError("packed residuals must be a 2D uint8 array")

        unpacked = np.empty((packed_residuals.shape[0], packed_residuals.shape[1] * 4), dtype=np.uint8)
        unpacked[:, 0::4] = packed_residuals & 0b00000011
        unpacked[:, 1::4] = (packed_residuals >> 2) & 0b00000011
        unpacked[:, 2::4] = (packed_residuals >> 4) & 0b00000011
        unpacked[:, 3::4] = (packed_residuals >> 6) & 0b00000011
        return np.ascontiguousarray(unpacked[:, : self.pca_dim])

    def _require_trained(self) -> None:
        if (
            self.input_dim is None
            or self.pca_components is None
            or self.pca_mean is None
            or self.centroids is None
            or self.quantile_thresholds is None
            or self.quantization_levels is None
        ):
            raise RuntimeError("TokenCompressor must be trained or loaded before use")

    def _require_pca_ready(self) -> None:
        if self.input_dim is None or self.pca_components is None or self.pca_mean is None:
            raise RuntimeError("TokenCompressor must be trained through the PCA stage before projection")

    def _require_pca_components(self) -> np.ndarray:
        if self.pca_components is None:
            raise RuntimeError("TokenCompressor is missing PCA components")
        return self.pca_components

    def _require_pca_mean(self) -> np.ndarray:
        if self.pca_mean is None:
            raise RuntimeError("TokenCompressor is missing PCA mean")
        return self.pca_mean

    def _require_centroids(self) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("TokenCompressor is missing centroids")
        return self.centroids

    def _require_quantile_thresholds(self) -> np.ndarray:
        if self.quantile_thresholds is None:
            raise RuntimeError("TokenCompressor is missing quantile thresholds")
        return self.quantile_thresholds

    def _require_quantization_levels(self) -> np.ndarray:
        if self.quantization_levels is None:
            raise RuntimeError("TokenCompressor is missing quantization levels")
        return self.quantization_levels
