from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from api.config import ServingConfig
from compression import CompressionPipelineConfig
from extraction import ExtractionConfig
from index import IndexConfig
from ingestion.config import DatasetConfig


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    datasets: list[Path] = Field(min_length=1)
    extraction: ExtractionConfig
    compression: CompressionPipelineConfig
    index: IndexConfig
    serving: ServingConfig
    output_dir: Path
    pipeline_db: Path

    @classmethod
    def from_yaml(cls, config_path: Path) -> PipelineConfig:
        raw_config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw_config, dict):
            raise TypeError(f"Expected mapping in {config_path}, found {type(raw_config).__name__}")
        enriched = _expand_pipeline_config(raw_config, config_path.parent)
        return cls.model_validate(enriched)


def _expand_pipeline_config(raw_config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if "datasets" not in raw_config:
        raise KeyError("pipeline config is missing the datasets field")

    dataset_paths = [_resolve_path(base_dir, Path(value)) for value in raw_config["datasets"]]
    dataset_configs = [DatasetConfig.from_yaml(path) for path in dataset_paths]

    default_output_dir = (base_dir.parent / "artifacts" / "pipeline").resolve()
    output_dir = _resolve_path(base_dir, Path(raw_config.get("output_dir", default_output_dir)))
    pipeline_db = _resolve_path(
        base_dir,
        Path(raw_config.get("pipeline_db", output_dir / "checkpoints" / "pipeline.sqlite3")),
    )

    extraction_config = dict(raw_config.get("extraction") or {})
    compression_config = dict(raw_config.get("compression") or {})
    index_config = dict(raw_config.get("index") or {})
    serving_config = dict(raw_config.get("serving") or {})

    extraction_config["dataset_configs"] = dataset_configs
    extraction_config["output_dir"] = _resolve_path(
        base_dir,
        Path(extraction_config.get("output_dir", output_dir / "extraction")),
    )
    extraction_config["checkpoint_db"] = _resolve_path(
        base_dir,
        Path(extraction_config.get("checkpoint_db", output_dir / "checkpoints" / "extraction.sqlite3")),
    )

    compression_config["raw_dir"] = _resolve_path(
        base_dir,
        Path(compression_config.get("raw_dir", extraction_config["output_dir"])),
    )
    compression_config["output_dir"] = _resolve_path(
        base_dir,
        Path(compression_config.get("output_dir", output_dir / "compression")),
    )
    compression_config["checkpoint_db"] = _resolve_path(
        base_dir,
        Path(compression_config.get("checkpoint_db", output_dir / "checkpoints" / "compression.sqlite3")),
    )

    index_config["compressed_dir"] = _resolve_path(
        base_dir,
        Path(index_config.get("compressed_dir", compression_config["output_dir"] / "shards")),
    )
    index_config["output_path"] = _resolve_path(
        base_dir,
        Path(index_config.get("output_path", compression_config["output_dir"] / "coarse_hnsw.faiss")),
    )

    serving_config["data_dir"] = _resolve_path(
        base_dir,
        Path(serving_config.get("data_dir", compression_config["output_dir"])),
    )

    return {
        "datasets": dataset_paths,
        "extraction": extraction_config,
        "compression": compression_config,
        "index": index_config,
        "serving": serving_config,
        "output_dir": output_dir,
        "pipeline_db": pipeline_db,
    }


def _resolve_path(base_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
