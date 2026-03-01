from __future__ import annotations
import math
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from typing import Any
import structlog
from pydantic import BaseModel, ConfigDict
from ingestion.config import DatasetConfig
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from transformers import AutoVideoProcessor

MODEL_REPO_ID = "facebook/vjepa2-vith-fpc64-256"


class ClipMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_id: str
    dataset_name: str
    robot_type: str
    clip_start_frame: int
    clip_end_frame: int
    timestamp_start: float
    timestamp_end: float
    language_instruction: str | None = None
    num_original_frames: int


class ClipFormer:
    def __init__(
        self,
        config: DatasetConfig,
        dataset: Any | None = None,
        video_processor: Any | None = None,
    ) -> None:
        self.config = config
        self._logger = structlog.get_logger(__name__).bind(repo_id=config.repo_id)
        self._dataset = dataset if dataset is not None else StreamingLeRobotDataset(
            self.config.repo_id,
            streaming=True,
            shuffle=False,
            buffer_size=1,
            max_num_shards=1,
        )
        self._video_processor = (
            video_processor if video_processor is not None else AutoVideoProcessor.from_pretrained(MODEL_REPO_ID)
        )
        self._episode_ranges = self._build_episode_ranges()

    def iter_clips(self) -> Iterator[tuple[dict[str, Any], ClipMetadata]]:
        if self._episode_ranges is not None:
            for episode_index in range(len(self._episode_ranges)):
                episode_frames, frame_numbers, language_instruction = self._load_episode(episode_index)
                yield from self._emit_episode_clips(
                    episode_index=episode_index,
                    episode_frames=episode_frames,
                    frame_numbers=frame_numbers,
                    language_instruction=language_instruction,
                )
            return

        for episode_index, episode_frames, frame_numbers, language_instruction in self._iter_streamed_episodes():
            yield from self._emit_episode_clips(
                episode_index=episode_index,
                episode_frames=episode_frames,
                frame_numbers=frame_numbers,
                language_instruction=language_instruction,
            )

    def _build_episode_ranges(self) -> list[tuple[int, int]] | None:
        episode_data_index = getattr(self._dataset, "episode_data_index", None)
        if not isinstance(episode_data_index, Mapping):
            return None

        episode_starts = [int(v) for v in episode_data_index["from"]]
        episode_stops = [int(v) for v in episode_data_index["to"]]
        return list(zip(episode_starts, episode_stops, strict=True))

    def _emit_episode_clips(
        self,
        episode_index: int,
        episode_frames: Sequence[Any],
        frame_numbers: Sequence[int],
        language_instruction: str | None,
    ) -> Iterator[tuple[dict[str, Any], ClipMetadata]]:
        ordered_pairs = sorted(zip(frame_numbers, episode_frames, strict=True), key=lambda item: item[0])
        ordered_frame_numbers = [frame_number for frame_number, _ in ordered_pairs]
        ordered_frames = [frame for _, frame in ordered_pairs]

        subsampled_frames, subsampled_frame_numbers = self._subsample_episode(
            ordered_frames,
            ordered_frame_numbers,
        )
        if not subsampled_frames:
            return

        episode_id = f"{self.config.repo_id}_{episode_index}"
        self._logger.debug(
            "prepared_episode",
            episode_id=episode_id,
            source_frames=len(ordered_frames),
            subsampled_frames=len(subsampled_frames),
        )

        first_frame = ordered_frame_numbers[0]
        for clip_frames, clip_frame_numbers, original_count in self._clip_windows(
            subsampled_frames,
            subsampled_frame_numbers,
        ):
            processed = self._preprocess_clip(clip_frames)
            start_frame = clip_frame_numbers[0]
            end_frame = clip_frame_numbers[original_count - 1]
            metadata = ClipMetadata(
                episode_id=episode_id,
                dataset_name=self.config.dataset_name,
                robot_type=self.config.robot_type,
                clip_start_frame=start_frame,
                clip_end_frame=end_frame,
                timestamp_start=(start_frame - first_frame) / self.config.source_fps,
                timestamp_end=(end_frame - first_frame) / self.config.source_fps,
                language_instruction=language_instruction,
                num_original_frames=original_count,
            )
            yield processed, metadata

    def _load_episode(self, episode_index: int) -> tuple[list[Any], list[int], str | None]:
        start, stop = self._episode_ranges[episode_index]
        frames: list[Any] = []
        frame_numbers: list[int] = []
        language_instruction: str | None = None

        for local_index, row_index in enumerate(range(start, stop)):
            sample = self._dataset[row_index]
            frames.append(self._get_nested_value(sample, self.config.image_key))
            frame_numbers.append(int(sample.get("frame_index", local_index)))

            if language_instruction is None and self.config.language_key:
                language_value = self._get_nested_value(sample, self.config.language_key)
                language_instruction = None if language_value is None else str(language_value)

        return frames, frame_numbers, language_instruction

    def _iter_streamed_episodes(self) -> Iterator[tuple[int, list[Any], list[int], str | None]]:
        current_episode_index: int | None = None
        frames: list[Any] = []
        frame_numbers: list[int] = []
        language_instruction: str | None = None

        for sample in self._iter_stream_samples():
            sample_episode_index = int(sample["episode_index"])

            if current_episode_index is None:
                current_episode_index = sample_episode_index
            elif sample_episode_index != current_episode_index:
                yield current_episode_index, frames, frame_numbers, language_instruction
                current_episode_index = sample_episode_index
                frames = []
                frame_numbers = []
                language_instruction = None

            frames.append(self._get_nested_value(sample, self.config.image_key))
            frame_numbers.append(int(sample.get("frame_index", len(frame_numbers))))

            if language_instruction is None and self.config.language_key:
                language_value = self._get_nested_value(sample, self.config.language_key)
                language_instruction = None if language_value is None else str(language_value)

        if current_episode_index is not None:
            yield current_episode_index, frames, frame_numbers, language_instruction

    def _iter_stream_samples(self) -> Iterator[Mapping[str, Any]]:
        if hasattr(self._dataset, "hf_dataset") and hasattr(self._dataset, "make_frame"):
            source_iterator = iter(self._dataset.hf_dataset)
            while True:
                try:
                    first_sample = next(source_iterator)
                except StopIteration:
                    return

                pending_iterator = chain([first_sample], source_iterator)
                try:
                    yield from self._dataset.make_frame(pending_iterator)
                except RuntimeError as error:
                    if str(error) == "generator raised StopIteration":
                        return
                    raise
                source_iterator = pending_iterator

        yield from self._dataset

    def _subsample_episode(
        self,
        frames: Sequence[Any],
        frame_numbers: Sequence[int],
    ) -> tuple[list[Any], list[int]]:
        if not frames:
            return [], []

        base_frame = frame_numbers[0]
        frame_timestamps = [
            (frame_number - base_frame) / self.config.source_fps for frame_number in frame_numbers
        ]
        target_frame_count = int(math.floor(frame_timestamps[-1] * self.config.target_fps)) + 1

        source_cursor = 0
        selected_positions: list[int] = []
        for target_index in range(target_frame_count):
            target_timestamp = target_index / self.config.target_fps
            while (
                source_cursor + 1 < len(frame_timestamps)
                and frame_timestamps[source_cursor + 1] <= target_timestamp
            ):
                source_cursor += 1

            chosen_position = source_cursor
            if source_cursor + 1 < len(frame_timestamps):
                previous_delta = abs(frame_timestamps[source_cursor] - target_timestamp)
                next_delta = abs(frame_timestamps[source_cursor + 1] - target_timestamp)
                if next_delta < previous_delta:
                    chosen_position = source_cursor + 1

            selected_positions.append(chosen_position)

        subsampled_frames = [frames[position] for position in selected_positions]
        subsampled_frame_numbers = [frame_numbers[position] for position in selected_positions]
        return subsampled_frames, subsampled_frame_numbers

    def _clip_windows(
        self,
        frames: Sequence[Any],
        frame_numbers: Sequence[int],
    ) -> Iterator[tuple[list[Any], list[int], int]]:
        start = 0
        total_frames = len(frames)
        while start < total_frames:
            end = min(start + self.config.clip_length, total_frames)
            window_frames = list(frames[start:end])
            window_frame_numbers = list(frame_numbers[start:end])
            original_count = len(window_frames)

            if original_count < self.config.clip_length:
                pad_count = self.config.clip_length - original_count
                window_frames.extend([window_frames[-1]] * pad_count)
                window_frame_numbers.extend([window_frame_numbers[-1]] * pad_count)
                yield window_frames, window_frame_numbers, original_count
                return

            yield window_frames, window_frame_numbers, original_count
            if end >= total_frames:
                return
            start += self.config.clip_stride

    def _preprocess_clip(self, clip_frames: Sequence[Any]) -> dict[str, Any]:
        processor_input: Any = list(clip_frames)
        if hasattr(clip_frames[0], "dim"):
            from torch import stack

            processor_input = stack(list(clip_frames))

        processed = dict(self._video_processor(processor_input, return_tensors="pt"))
        pixel_values = processed.get("pixel_values")
        if pixel_values is None:
            pixel_values = processed.pop("pixel_values_videos")
            processed["pixel_values"] = pixel_values
        if len(pixel_values.shape) == 5 and pixel_values.shape[0] == 1:
            processed["pixel_values"] = pixel_values.squeeze(0)
        return processed

    @staticmethod
    def _get_nested_value(sample: Mapping[str, Any], key_path: str) -> Any:
        if key_path in sample:
            return sample[key_path]
        current = sample
        for key in key_path.split("."):
            current = current[key]
        return current
