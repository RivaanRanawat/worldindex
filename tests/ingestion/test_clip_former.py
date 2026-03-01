from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PIL import Image

from ingestion import ClipFormer, DatasetConfig, load_all_configs


def _make_image(frame_value: int) -> Image.Image:
    return Image.new("RGB", (32, 32), color=(frame_value % 256, 0, 0))


def _assign_nested_value(target: dict[str, Any], key_path: str, value: Any) -> None:
    current = target
    keys = key_path.split(".")
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


class FakeStreamingDataset:
    def __init__(
        self,
        episode_lengths: list[int],
        *,
        image_key: str,
        language_key: str | None = None,
        language_values: dict[int, str] | None = None,
    ) -> None:
        self.episode_data_index = {"from": [], "to": []}
        self._samples: list[dict[str, Any]] = []

        offset = 0
        for episode_index, episode_length in enumerate(episode_lengths):
            self.episode_data_index["from"].append(offset)
            instruction = None if language_values is None else language_values.get(episode_index)
            for frame_index in range(episode_length):
                sample: dict[str, Any] = {"episode_index": episode_index, "frame_index": frame_index}
                _assign_nested_value(sample, image_key, _make_image(frame_index))
                if language_key is not None and instruction is not None:
                    _assign_nested_value(sample, language_key, instruction)
                self._samples.append(sample)
                offset += 1
            self.episode_data_index["to"].append(offset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._samples[index]


class FakeIterableStreamingDataset:
    def __init__(
        self,
        episode_lengths: list[int],
        *,
        image_key: str,
        language_key: str | None = None,
        language_values: dict[int, str] | None = None,
    ) -> None:
        self.num_episodes = len(episode_lengths)
        self._samples: list[dict[str, Any]] = []

        for episode_index, episode_length in enumerate(episode_lengths):
            instruction = None if language_values is None else language_values.get(episode_index)
            for frame_index in range(episode_length):
                sample: dict[str, Any] = {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    image_key: _make_image(frame_index),
                }
                if language_key is not None and instruction is not None:
                    sample[language_key] = instruction
                self._samples.append(sample)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._samples)


class FakeIterableStreamingDatasetWithoutEpisodeCount:
    def __init__(
        self,
        episode_lengths: list[int],
        *,
        image_key: str,
        include_frame_index: bool = True,
    ) -> None:
        self._samples: list[dict[str, Any]] = []

        for episode_index, episode_length in enumerate(episode_lengths):
            for frame_index in range(episode_length):
                sample: dict[str, Any] = {
                    "episode_index": episode_index,
                    image_key: _make_image(frame_index),
                }
                if include_frame_index:
                    sample["frame_index"] = frame_index
                self._samples.append(sample)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._samples)


class FakeMakeFrameStreamingDataset:
    def __init__(self, episode_lengths: list[int], *, image_key: str) -> None:
        self.hf_dataset: list[dict[str, Any]] = []
        self.make_frame_calls = 0

        for episode_index, episode_length in enumerate(episode_lengths):
            for frame_index in range(episode_length):
                self.hf_dataset.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        image_key: _make_image(frame_index),
                    }
                )

    def make_frame(self, iterator: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        self.make_frame_calls += 1
        yield from iterator


class RecordingVideoProcessor:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def __call__(self, frames: list[Image.Image], *, return_tensors: str) -> dict[str, np.ndarray]:
        assert return_tensors == "pt"
        frame_values = [frame.getpixel((0, 0))[0] for frame in frames]
        self.calls.append(frame_values)

        pixel_values = np.zeros((1, len(frames), 3, 256, 256), dtype=np.float32)
        for frame_position, frame_value in enumerate(frame_values):
            pixel_values[0, frame_position, :, :, :] = frame_value

        return {"pixel_values": pixel_values}


def _build_config(**overrides: Any) -> DatasetConfig:
    base_config = {
        "repo_id": "lerobot/droid",
        "image_key": "observation.images.exterior_image_1_left",
        "source_fps": 15,
        "target_fps": 4,
        "robot_type": "franka",
        "language_key": "language_instruction",
        "clip_length": 64,
        "clip_stride": 32,
    }
    base_config.update(overrides)
    return DatasetConfig.model_validate(base_config)


def test_load_all_configs_returns_sorted_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "b.yaml").write_text(
        "repo_id: lerobot/b\nimage_key: observation.images.image_0\nsource_fps: 5\nrobot_type: widowx\n"
    )
    (tmp_path / "a.yaml").write_text(
        "repo_id: lerobot/a\nimage_key: observation.images.image_0\nsource_fps: 15\nrobot_type: franka\n"
    )

    configs = load_all_configs(tmp_path)

    assert [config.repo_id for config in configs] == ["lerobot/a", "lerobot/b"]


def test_project_dataset_configs_load() -> None:
    configs = load_all_configs(Path("config/datasets"))

    assert [config.dataset_name for config in configs] == [
        "aloha_sim_insertion_scripted_image",
        "bridge",
        "droid_1.0.1",
    ]


def test_iter_clips_subsamples_to_target_fps_and_pads_short_episode() -> None:
    config = _build_config()
    dataset = FakeStreamingDataset(
        [100],
        image_key=config.image_key,
        language_key=config.language_key,
        language_values={0: "pick up the mug"},
    )
    processor = RecordingVideoProcessor()
    clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

    clips = list(clip_former.iter_clips())

    assert len(clips) == 1
    processed, metadata = clips[0]
    assert processed["pixel_values"].shape == (64, 3, 256, 256)
    assert metadata.episode_id == "lerobot/droid_0"
    assert metadata.dataset_name == "droid"
    assert metadata.robot_type == "franka"
    assert metadata.language_instruction == "pick up the mug"
    assert metadata.num_original_frames == 27
    assert metadata.clip_start_frame == 0
    assert metadata.clip_end_frame == processor.calls[0][26]
    assert processor.calls[0][26] == processor.calls[0][-1]


def test_iter_clips_emits_overlapping_windows_for_long_episode() -> None:
    config = _build_config(language_key=None)
    dataset = FakeStreamingDataset([400], image_key=config.image_key)
    processor = RecordingVideoProcessor()
    clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

    clips = list(clip_former.iter_clips())

    assert len(clips) == 3
    assert [metadata.num_original_frames for _, metadata in clips] == [64, 64, 43]
    assert [metadata.clip_start_frame for _, metadata in clips] == [0, 120, 240]
    assert [metadata.episode_id for _, metadata in clips] == [
        "lerobot/droid_0",
        "lerobot/droid_0",
        "lerobot/droid_0",
    ]


# Enable when distributed added

# def test_iter_clips_for_episode_returns_only_requested_episode() -> None:
#     config = _build_config()
#     dataset = FakeStreamingDataset(
#         [100, 400],
#         image_key=config.image_key,
#         language_key=config.language_key,
#         language_values={0: "first task", 1: "second task"},
#     )
#     processor = RecordingVideoProcessor()
#     clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

#     clips = list(clip_former.iter_clips_for_episode("lerobot/droid_1"))

#     assert len(clips) == 3
#     assert all(metadata.episode_id == "lerobot/droid_1" for _, metadata in clips)
#     assert all(metadata.language_instruction == "second task" for _, metadata in clips)


# def test_iter_clips_for_episode_supports_iterable_streaming_dataset() -> None:
#     config = _build_config()
#     dataset = FakeIterableStreamingDataset(
#         [100, 100],
#         image_key=config.image_key,
#         language_key=config.language_key,
#         language_values={0: "first task", 1: "second task"},
#     )
#     processor = RecordingVideoProcessor()
#     clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

#     clips = list(clip_former.iter_clips_for_episode("lerobot/droid_1"))

#     assert len(clips) == 1
#     processed, metadata = clips[0]
#     assert processed["pixel_values"].shape == (64, 3, 256, 256)
#     assert metadata.episode_id == "lerobot/droid_1"
#     assert metadata.language_instruction == "second task"
#     assert metadata.num_original_frames == 27


# def test_iter_clips_for_episode_rejects_missing_episode_for_unknown_stream_length() -> None:
#     config = _build_config(language_key=None)
#     dataset = FakeIterableStreamingDatasetWithoutEpisodeCount(
#         [100, 100],
#         image_key=config.image_key,
#     )
#     processor = RecordingVideoProcessor()
#     clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

#     with pytest.raises(ValueError):
#         list(clip_former.iter_clips_for_episode("lerobot/droid_5"))


# def test_iter_clips_for_episode_rejects_invalid_identifier() -> None:
#     config = _build_config(language_key=None)
#     dataset = FakeStreamingDataset([100], image_key=config.image_key)
#     processor = RecordingVideoProcessor()
#     clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

#     with pytest.raises(ValueError):
#         list(clip_former.iter_clips_for_episode("lerobot/bridge_0"))


def test_padding_repeats_final_subsampled_frame_for_tiny_episode() -> None:
    config = _build_config(language_key=None)
    dataset = FakeStreamingDataset([40], image_key=config.image_key)
    processor = RecordingVideoProcessor()
    clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

    clips = list(clip_former.iter_clips())

    assert len(clips) == 1
    processed, metadata = clips[0]
    assert processed["pixel_values"].shape == (64, 3, 256, 256)
    assert metadata.num_original_frames == 11
    assert processor.calls[0][10] == processor.calls[0][-1]
    assert np.all(processed["pixel_values"][-1] == processor.calls[0][10])


def test_iter_clips_resets_fallback_frame_indices_for_each_streamed_episode() -> None:
    config = _build_config(
        language_key=None,
        source_fps=4,
        target_fps=4,
        clip_length=2,
        clip_stride=2,
    )
    dataset = FakeIterableStreamingDatasetWithoutEpisodeCount(
        [2, 2],
        image_key=config.image_key,
        include_frame_index=False,
    )
    processor = RecordingVideoProcessor()
    clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

    clips = list(clip_former.iter_clips())

    assert [metadata.clip_start_frame for _, metadata in clips] == [0, 0]
    assert [metadata.clip_end_frame for _, metadata in clips] == [1, 1]


def test_iter_clips_consumes_make_frame_stream_once() -> None:
    config = _build_config(language_key=None)
    dataset = FakeMakeFrameStreamingDataset([100], image_key=config.image_key)
    processor = RecordingVideoProcessor()
    clip_former = ClipFormer(config, dataset=dataset, video_processor=processor)

    clips = list(clip_former.iter_clips())

    assert len(clips) == 1
    assert dataset.make_frame_calls == 1
