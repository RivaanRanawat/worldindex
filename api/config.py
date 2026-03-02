from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from retrieval.models import QueryEncoderConfig, RetrievalConfig

_DEFAULT_MODEL_ID = "facebook/vjepa2-vith-fpc64-256"


class ServingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str = Field(default="127.0.0.1", description="Interface to bind the FastAPI server to.")
    port: int = Field(default=8000, ge=1, le=65_535, description="TCP port for the FastAPI server.")
    data_dir: Path = Field(description="Directory containing WorldIndex serving artifacts.")
    model_id: str = Field(default=_DEFAULT_MODEL_ID, description="Frozen V-JEPA 2 checkpoint identifier.")
    device: str = Field(
        default="cpu",
        pattern="^(cpu|cuda)$",
        description="Execution device for V-JEPA 2 inference.",
    )
    max_concurrent_queries: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of concurrent V-JEPA 2 encodes. Defaults to 1 on CUDA and CPU core count on CPU.",
    )

    @property
    def faiss_index_path(self) -> Path:
        return self.data_dir / "coarse_hnsw.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.data_dir / "clips.parquet"

    @property
    def shard_dir(self) -> Path:
        return self.data_dir / "shards"

    @property
    def compressor_dir(self) -> Path:
        return self.data_dir / "compression_model"

    @property
    def trajectory_embedding_dir(self) -> Path | None:
        trajectory_dir = self.data_dir / "trajectory_embeddings"
        return trajectory_dir if trajectory_dir.exists() else None

    @property
    def resolved_max_concurrent_queries(self) -> int:
        if self.max_concurrent_queries is not None:
            return self.max_concurrent_queries
        if self.device == "cuda":
            return 1

        return max(1, os.cpu_count() or 1)

    def to_retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            faiss_index_path=self.faiss_index_path,
            metadata_path=self.metadata_path,
            shard_dir=self.shard_dir,
            compressor_dir=self.compressor_dir,
            query_encoder=QueryEncoderConfig(
                pca_artifact_dir=self.compressor_dir,
                model_id=self.model_id,
                device=self.device,
            ),
            trajectory_embedding_dir=self.trajectory_embedding_dir,
            maxsim_use_gpu=self.device == "cuda",
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> ServingConfig:
        raw_config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw_config, dict):
            raise TypeError(f"Expected mapping in {config_path}, found {type(raw_config).__name__}")
        return cls.model_validate(raw_config)
