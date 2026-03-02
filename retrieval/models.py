from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


_DEFAULT_MODEL_ID = "facebook/vjepa2-vith-fpc64-256"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    row_start: int = Field(ge=0, le=15)
    row_end: int = Field(ge=0, le=15)
    col_start: int = Field(ge=0, le=15)
    col_end: int = Field(ge=0, le=15)

    @model_validator(mode="after")
    def _validate_bounds(self) -> BoundingBox:
        if self.row_end < self.row_start:
            raise ValueError("row_end must be greater than or equal to row_start")
        if self.col_end < self.col_start:
            raise ValueError("col_end must be greater than or equal to col_start")
        return self

    def to_patch_indices(self, grid_size: int = 16) -> list[int]:
        indices: list[int] = []
        for row in range(self.row_start, self.row_end + 1):
            row_offset = row * grid_size
            for col in range(self.col_start, self.col_end + 1):
                indices.append(row_offset + col)
        return indices


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    clip_id: int = Field(ge=0)
    episode_id: str
    dataset_name: str
    robot_type: str
    score: float
    timestamp_start: float
    timestamp_end: float
    language_instruction: str | None = None


class TrajectoryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_id: str
    dataset_name: str
    robot_type: str
    dtw_distance: float
    alignment_path: list[tuple[int, int]]


class QueryEncoderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    pca_artifact_dir: Path
    model_id: str = _DEFAULT_MODEL_ID
    device: str = "cpu"
    clip_length: int = Field(default=64, gt=0)
    temporal_positions: int = Field(default=32, gt=0)
    spatial_grid_size: int = Field(default=16, gt=0)

    @property
    def spatial_token_count(self) -> int:
        return self.spatial_grid_size * self.spatial_grid_size

    @property
    def token_count(self) -> int:
        return self.temporal_positions * self.spatial_token_count

    @property
    def midpoint_index(self) -> int:
        return self.temporal_positions // 2


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    faiss_index_path: Path
    metadata_path: Path
    shard_dir: Path
    compressor_dir: Path
    query_encoder: QueryEncoderConfig
    trajectory_embedding_dir: Path | None = None
    maxsim_candidate_batch_size: int = Field(default=25, gt=0)
    maxsim_use_gpu: bool = True
    trajectory_frame_search_k: int = Field(default=8, gt=0)
    coarse_search_k: int = Field(default=100, gt=0)
