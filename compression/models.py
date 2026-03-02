from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CompressedClip(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    centroid_ids: np.ndarray
    quantized_residuals: np.ndarray
    coarse_vector: np.ndarray

    @field_validator("centroid_ids")
    @classmethod
    def _validate_centroid_ids(cls, value: Any) -> np.ndarray:
        centroid_ids = np.asarray(value, dtype=np.uint16)
        if centroid_ids.ndim != 1:
            raise ValueError("centroid_ids must be a 1D uint16 array")
        return np.ascontiguousarray(centroid_ids)

    @field_validator("quantized_residuals")
    @classmethod
    def _validate_quantized_residuals(cls, value: Any) -> np.ndarray:
        quantized_residuals = np.asarray(value, dtype=np.uint8)
        if quantized_residuals.ndim != 2:
            raise ValueError("quantized_residuals must be a 2D uint8 array")
        return np.ascontiguousarray(quantized_residuals)

    @field_validator("coarse_vector")
    @classmethod
    def _validate_coarse_vector(cls, value: Any) -> np.ndarray:
        coarse_vector = np.asarray(value, dtype=np.float32)
        if coarse_vector.ndim != 1:
            raise ValueError("coarse_vector must be a 1D float32 array")
        return np.ascontiguousarray(coarse_vector)

    def model_post_init(self, __context: Any) -> None:
        if self.quantized_residuals.shape[0] != self.centroid_ids.shape[0]:
            raise ValueError("centroid_ids and quantized_residuals must have the same token count")


class ShardHeader(BaseModel):
    model_config = ConfigDict(frozen=True)

    clip_count: int = Field(ge=0)
    token_count: int = Field(gt=0)
    pca_dim: int = Field(gt=0)
    residual_bytes_per_token: int = Field(gt=0)
    coarse_dim: int = Field(gt=0)
    record_size: int = Field(gt=0)

    @property
    def centroid_ids_nbytes(self) -> int:
        return self.token_count * np.dtype(np.uint16).itemsize

    @property
    def quantized_residuals_nbytes(self) -> int:
        return self.token_count * self.residual_bytes_per_token

    @property
    def coarse_vector_nbytes(self) -> int:
        return self.coarse_dim * np.dtype(np.float32).itemsize

    def record_dtype(self) -> np.dtype[Any]:
        return np.dtype(
            [
                ("centroid_ids", np.uint16, (self.token_count,)),
                (
                    "quantized_residuals",
                    np.uint8,
                    (self.token_count, self.residual_bytes_per_token),
                ),
                ("coarse_vector", np.float32, (self.coarse_dim,)),
            ]
        )

    def model_post_init(self, __context: Any) -> None:
        expected_record_size = (
            self.centroid_ids_nbytes
            + self.quantized_residuals_nbytes
            + self.coarse_vector_nbytes
        )
        if self.record_size != expected_record_size:
            raise ValueError(
                f"record_size={self.record_size} does not match expected {expected_record_size}"
            )
