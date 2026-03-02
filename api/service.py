from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from io import BytesIO
from typing import TYPE_CHECKING, Any, TypeVar

from PIL import Image, UnidentifiedImageError

from api.config import ServingConfig
from api.models import HealthResponse
from retrieval.models import BoundingBox, EpisodeDetails, SearchResult, TrajectoryResult, TransitionResult

if TYPE_CHECKING:
    from retrieval.engine import RetrievalEngine

_MIN_IMAGE_DIMENSION = 16

T = TypeVar("T")


def build_service(
    config: ServingConfig,
    engine: RetrievalEngine | None = None,
) -> RetrievalService:
    if engine is None:
        from retrieval.engine import RetrievalEngine

        resolved_engine = RetrievalEngine(config.to_retrieval_config())
    else:
        resolved_engine = engine
    return RetrievalService(
        engine=resolved_engine,
        max_concurrent_queries=config.resolved_max_concurrent_queries,
    )


def decode_image_bytes(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Image payload is empty")

    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Unsupported image format. Use JPEG, PNG, or WebP.") from exc

    rgb_image = image.convert("RGB")
    if rgb_image.width < _MIN_IMAGE_DIMENSION or rgb_image.height < _MIN_IMAGE_DIMENSION:
        raise ValueError("Image dimensions must be at least 16x16 pixels")
    return rgb_image


class RetrievalService:
    def __init__(
        self,
        engine: RetrievalEngine,
        max_concurrent_queries: int,
    ) -> None:
        self._engine = engine
        self._query_semaphore = asyncio.Semaphore(max_concurrent_queries)
        self._started_at = time.perf_counter()

    async def search_image(
        self,
        image: Image.Image,
        top_k: int,
        coarse_k: int,
    ) -> list[SearchResult]:
        return await self._run_limited(self._engine.search_image, image, top_k, coarse_k)

    async def search_spatial(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        top_k: int,
    ) -> list[SearchResult]:
        return await self._run_limited(self._engine.search_spatial, image, bbox, top_k)

    async def search_trajectory(
        self,
        episode_id: str,
        top_k: int,
    ) -> list[TrajectoryResult]:
        return await asyncio.to_thread(self._engine.search_trajectory, episode_id, top_k)

    async def search_transition(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        top_k: int,
        max_gap_seconds: float,
    ) -> list[TransitionResult]:
        return await self._run_limited(
            self._engine.search_transition,
            image_a,
            image_b,
            top_k,
            max_gap_seconds,
        )

    async def get_episode_details(self, episode_id: str) -> EpisodeDetails:
        return await asyncio.to_thread(self._engine.get_episode_details, episode_id)

    def health(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            indexed_clips=self._engine.indexed_clip_count,
            model_id=self._engine.model_id,
            device=self._engine.device,
            uptime_seconds=time.perf_counter() - self._started_at,
        )

    async def _run_limited(
        self,
        fn: Callable[..., T],
        *args: Any,
    ) -> T:
        async with self._query_semaphore:
            return await asyncio.to_thread(fn, *args)
