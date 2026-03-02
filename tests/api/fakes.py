from __future__ import annotations

import threading
import time
from io import BytesIO

from PIL import Image

from retrieval.models import EpisodeDetails, SearchResult, TrajectoryResult, TransitionResult


class FakeEngine:
    def __init__(self, search_delay_seconds: float = 0.0) -> None:
        self.indexed_clip_count = 6
        self.model_id = "fake-vjepa2"
        self.device = "cpu"
        self.search_delay_seconds = search_delay_seconds
        self.max_active_image_searches = 0
        self._active_image_searches = 0
        self._lock = threading.Lock()

    def search_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        coarse_k: int = 100,
    ) -> list[SearchResult]:
        del image
        del coarse_k
        self._enter_image_search()
        try:
            time.sleep(self.search_delay_seconds)
            return self._make_search_results(top_k)
        finally:
            self._exit_image_search()

    def search_spatial(
        self,
        image: Image.Image,
        bbox: object,
        top_k: int = 10,
    ) -> list[SearchResult]:
        del image
        del bbox
        return self._make_search_results(top_k)

    def search_trajectory(
        self,
        episode_id: str,
        top_k: int = 10,
    ) -> list[TrajectoryResult]:
        return [
            TrajectoryResult(
                episode_id=f"{episode_id}-match-{index}",
                dataset_name="droid",
                robot_type="franka",
                dtw_distance=0.1 + index,
                alignment_path=[(0, 0), (1, 1)],
            )
            for index in range(top_k)
        ]

    def search_transition(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        top_k: int = 10,
        max_gap_seconds: float = 30.0,
    ) -> list[TransitionResult]:
        del image_a
        del image_b
        del max_gap_seconds
        return [
            TransitionResult(
                clip_id=index,
                episode_id=f"transition-{index}",
                dataset_name="bridge",
                robot_type="ur5",
                score=0.95 - (index * 0.05),
                timestamp_start=float(index),
                timestamp_end=float(index + 1),
                language_instruction="move mug",
            )
            for index in range(top_k)
        ]

    def get_episode_details(self, episode_id: str) -> EpisodeDetails:
        if episode_id == "missing":
            raise KeyError(f"Unknown episode_id: {episode_id}")

        return EpisodeDetails(
            episode_id=episode_id,
            dataset_name="droid",
            robot_type="franka",
            clip_count=3,
            timestamp_start=0.0,
            timestamp_end=12.0,
            language_instruction="pick up the cube",
        )

    def _enter_image_search(self) -> None:
        with self._lock:
            self._active_image_searches += 1
            self.max_active_image_searches = max(self.max_active_image_searches, self._active_image_searches)

    def _exit_image_search(self) -> None:
        with self._lock:
            self._active_image_searches -= 1

    def _make_search_results(self, top_k: int) -> list[SearchResult]:
        return [
            SearchResult(
                clip_id=index,
                episode_id=f"episode-{index}",
                dataset_name="droid",
                robot_type="franka",
                score=1.0 - (index * 0.1),
                timestamp_start=float(index),
                timestamp_end=float(index + 1),
                language_instruction="pick up the block",
            )
            for index in range(top_k)
        ]


def make_test_image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(32, 64, 128)).save(buffer, format="PNG")
    return buffer.getvalue()
