from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TaskStatus = Literal["pending", "running", "completed", "failed"]


class PipelineTaskRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_name: str
    status: TaskStatus
    retry_count: int = Field(ge=0)
    checkpoint: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime


class PipelineStatusSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    tasks: list[PipelineTaskRecord]
    total_clips_processed: int = Field(ge=0)
    estimated_seconds_remaining: float | None = Field(default=None, ge=0.0)


class SampleQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_clip_id: int = Field(ge=0)
    top_clip_ids: list[int]
    top_episode_ids: list[str]


class ArtifactSize(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    bytes: int = Field(ge=0)


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    recall_at_10: float = Field(ge=0.0)
    mean_cosine_similarity: float = Field(ge=-1.0, le=1.0)
    sampled_clip_count: int = Field(ge=0)
    sample_queries: list[SampleQueryResult]
    artifact_sizes: list[ArtifactSize]
