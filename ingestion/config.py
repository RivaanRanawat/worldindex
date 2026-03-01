from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    repo_id: str
    image_key: str
    source_fps: int = Field(gt=0)
    target_fps: int = Field(default=4, gt=0)
    robot_type: str
    language_key: str | None = None
    clip_length: int = Field(default=64, gt=0)
    clip_stride: int = Field(default=32, gt=0)

    @property
    def dataset_name(self) -> str:
        return self.repo_id.rsplit("/", maxsplit=1)[-1]

    @classmethod
    def from_yaml(cls, config_path: Path) -> DatasetConfig:
        raw_config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(raw_config, dict):
            raise TypeError(f"Expected mapping in {config_path}, found {type(raw_config).__name__}")
        return cls.model_validate(raw_config)


def load_all_configs(config_dir: Path) -> list[DatasetConfig]:
    config_paths = sorted(
        path
        for path in config_dir.iterdir()
        if path.is_file() and path.suffix in {".yaml", ".yml"}
    )
    return [DatasetConfig.from_yaml(path) for path in config_paths]
