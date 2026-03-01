from __future__ import annotations

import os
from typing import Any

import numpy as np

from ingestion.config import DatasetConfig

DEFAULT_CLIP_SHAPE = (64, 3, 4, 4)


class FakeMetadata:
    def __init__(self, dataset_name: str, robot_type: str, clip_number: int) -> None:
        self._payload = {
            "episode_id": f"{dataset_name}_episode_0",
            "dataset_name": dataset_name,
            "robot_type": robot_type,
            "clip_start_frame": clip_number * 4,
            "clip_end_frame": clip_number * 4 + 3,
            "timestamp_start": float(clip_number),
            "timestamp_end": float(clip_number) + 0.75,
            "language_instruction": f"task_{clip_number}",
            "num_original_frames": 4,
        }

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        assert mode == "python"
        return dict(self._payload)


class FakeClipFormer:
    def __init__(
        self,
        config: DatasetConfig,
        clip_count: int,
        dataset_name: str | None = None,
        clip_shape: tuple[int, ...] = DEFAULT_CLIP_SHAPE,
    ) -> None:
        self._config = config
        self._clip_count = clip_count
        self._clip_shape = clip_shape
        self._dataset_name = dataset_name or config.dataset_name

    def iter_clips(self) -> Any:
        for clip_number in range(self._clip_count):
            pixel_values = np.full(
                self._clip_shape,
                fill_value=float(clip_number),
                dtype=np.float32,
            )
            yield (
                {"pixel_values": pixel_values},
                FakeMetadata(
                    dataset_name=self._dataset_name,
                    robot_type=self._config.robot_type,
                    clip_number=clip_number,
                ),
            )


class FakeVisionModel:
    def __init__(self, token_shape: tuple[int, int]) -> None:
        self._token_shape = token_shape

    def eval(self) -> FakeVisionModel:
        return self

    def to(self, device: str) -> FakeVisionModel:
        del device
        return self

    def get_vision_features(self, batch: Any) -> Any:
        import torch

        batch_size = int(batch.shape[0])
        outputs = torch.empty(
            (batch_size, *self._token_shape),
            dtype=batch.dtype,
            device=batch.device,
        )
        base_values = batch.float().mean(dim=tuple(range(1, batch.ndim)))
        for batch_index in range(batch_size):
            outputs[batch_index].fill_(float(base_values[batch_index].item() + batch_index))
        return outputs


class LimitedRealClipFormer:
    def __init__(self, config: DatasetConfig, clip_limit: int) -> None:
        from ingestion import ClipFormer

        self._clip_former = ClipFormer(config)
        self._clip_limit = clip_limit

    def iter_clips(self) -> Any:
        yielded = 0
        for clip in self._clip_former.iter_clips():
            yield clip
            yielded += 1
            if yielded >= self._clip_limit:
                return


def build_fake_clip_former(config: DatasetConfig) -> FakeClipFormer:
    return FakeClipFormer(
        config,
        clip_count=_parse_clip_count(config.repo_id),
        dataset_name=_parse_dataset_label(config.repo_id),
    )


def build_limited_real_clip_former(config: DatasetConfig) -> LimitedRealClipFormer:
    clip_limit = int(os.environ.get("WORLDINDEX_REAL_EXTRACTION_CLIP_LIMIT", "20"))
    return LimitedRealClipFormer(config, clip_limit=clip_limit)


def load_fake_encoder_model(model_id: str, device: str) -> FakeVisionModel:
    del device
    return FakeVisionModel(token_shape=_parse_token_shape(model_id))


def load_full_shape_fake_encoder_model(model_id: str, device: str) -> FakeVisionModel:
    del model_id, device
    return FakeVisionModel(token_shape=(8192, 1280))


def _parse_clip_count(repo_id: str) -> int:
    suffix = repo_id.rsplit("/", maxsplit=1)[-1]
    _, clip_count = suffix.rsplit("__", maxsplit=1)
    return int(clip_count)


def _parse_dataset_label(repo_id: str) -> str:
    suffix = repo_id.rsplit("/", maxsplit=1)[-1]
    label, _ = suffix.rsplit("__", maxsplit=1)
    return label


def _parse_token_shape(model_id: str) -> tuple[int, int]:
    _, dimensions = model_id.split(":", maxsplit=1)
    rows, cols = dimensions.split("x", maxsplit=1)
    return int(rows), int(cols)
