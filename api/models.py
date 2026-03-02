from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from retrieval.models import (
    BoundingBox as RetrievalBoundingBox,
    EpisodeDetails,
    SearchResult,
    TrajectoryResult,
    TransitionResult,
)


class BoundingBox(BaseModel):
    """Inclusive patch bounds on the 16x16 spatial grid extracted from a query frame."""

    model_config = ConfigDict(frozen=True)

    row_start: int = Field(ge=0, le=15, description="First included patch row.")
    row_end: int = Field(ge=0, le=15, description="Last included patch row.")
    col_start: int = Field(ge=0, le=15, description="First included patch column.")
    col_end: int = Field(ge=0, le=15, description="Last included patch column.")

    @model_validator(mode="after")
    def _validate_bounds(self) -> BoundingBox:
        if self.row_end < self.row_start:
            raise ValueError("row_end must be greater than or equal to row_start")
        if self.col_end < self.col_start:
            raise ValueError("col_end must be greater than or equal to col_start")
        return self

    def to_retrieval_bbox(self) -> RetrievalBoundingBox:
        return RetrievalBoundingBox(
            row_start=self.row_start,
            row_end=self.row_end,
            col_start=self.col_start,
            col_end=self.col_end,
        )


class ImageSearchRequest(BaseModel):
    """Query parameters for whole-image visual retrieval."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=10, gt=0, description="Number of reranked results to return.")
    coarse_k: int = Field(default=100, gt=0, description="Number of coarse FAISS candidates to rerank.")


class SpatialSearchRequest(BaseModel):
    """Parameters for patch-localized retrieval over a query image."""

    model_config = ConfigDict(frozen=True)

    bbox: BoundingBox = Field(description="Inclusive patch region in the 16x16 token grid.")
    top_k: int = Field(default=10, gt=0, description="Number of reranked results to return.")


class TrajectorySearchRequest(BaseModel):
    """JSON body for retrieving episodes with similar trajectory embeddings."""

    model_config = ConfigDict(frozen=True)

    episode_id: str = Field(min_length=1, description="Episode identifier to use as the trajectory query.")
    top_k: int = Field(default=10, gt=0, description="Number of similar episodes to return.")


class TransitionSearchRequest(BaseModel):
    """Query parameters for searching state transitions within a time gap."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=10, gt=0, description="Number of transitions to return.")
    max_gap_seconds: float = Field(
        default=30.0,
        ge=0.0,
        description="Maximum allowed time gap between the two matched states.",
    )


class SearchResponse(RootModel[list[SearchResult]]):
    """Array response for image and spatial search."""


class TransitionSearchResponse(RootModel[list[TransitionResult]]):
    """Array response for transition search."""


class TrajectorySearchResponse(RootModel[list[TrajectoryResult]]):
    """Array response for trajectory search."""


class EpisodeDetailsResponse(EpisodeDetails):
    """Episode-level metadata derived from the clip catalog."""


class HealthResponse(BaseModel):
    """Runtime and model readiness information for the serving process."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="High-level readiness status.")
    indexed_clips: int = Field(ge=0, description="Number of clips currently loaded into the coarse index.")
    model_id: str = Field(description="V-JEPA 2 model identifier used for query encoding.")
    device: str = Field(description="Execution device currently serving encodes.")
    uptime_seconds: float = Field(ge=0.0, description="Seconds since the process loaded its retrieval engine.")
