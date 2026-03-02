from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, Awaitable, Callable, TypeVar

import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from PIL import Image
from pydantic import ValidationError

from api.config import ServingConfig
from api.models import (
    BoundingBox,
    EpisodeDetailsResponse,
    HealthResponse,
    ImageSearchRequest,
    SearchResponse,
    SpatialSearchRequest,
    TrajectorySearchRequest,
    TrajectorySearchResponse,
    TransitionSearchRequest,
    TransitionSearchResponse,
)
from api.service import RetrievalService, build_service, decode_image_bytes

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_LOGGER = structlog.get_logger(__name__).bind(component="api_server")

T = TypeVar("T")


def create_app(
    config: ServingConfig | None = None,
    service: RetrievalService | None = None,
) -> FastAPI:
    if config is None and service is None:
        raise ValueError("create_app requires either a ServingConfig or a prebuilt RetrievalService")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        if not hasattr(app.state, "service"):
            if service is not None:
                app.state.service = service
            elif config is not None:
                app.state.service = await asyncio.to_thread(build_service, config)
            else:
                raise RuntimeError("Unable to initialize retrieval service")
        yield

    app = FastAPI(
        title="WorldIndex API",
        version="0.1.0",
        lifespan=lifespan,
    )
    if service is not None:
        app.state.service = service

    @app.middleware("http")
    async def log_request_timing(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
        start_time = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            _LOGGER.info(
                "request_complete",
                method=request.method,
                path=request.url.path,
                status_code=500 if response is None else response.status_code,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 3),
            )

    @app.post("/search/image", response_model=SearchResponse)
    async def search_image(
        request: Request,
        image: Annotated[UploadFile, File(description="Query image in JPEG, PNG, or WebP format.")],
        params: Annotated[ImageSearchRequest, Depends()],
    ) -> SearchResponse:
        query_image = await _read_upload_image(image)
        results = await _handle_errors(
            _get_service(request).search_image(
                query_image,
                params.top_k,
                params.coarse_k,
            )
        )
        return SearchResponse(root=results)

    @app.post("/search/spatial", response_model=SearchResponse)
    async def search_spatial(
        request: Request,
        image: Annotated[UploadFile, File(description="Query image in JPEG, PNG, or WebP format.")],
        bbox: Annotated[
            str,
            Form(description="JSON object with row_start, row_end, col_start, and col_end."),
        ],
        top_k: Annotated[int, Query(gt=0)] = 10,
    ) -> SearchResponse:
        query_image = await _read_upload_image(image)
        try:
            payload = SpatialSearchRequest(
                bbox=BoundingBox.model_validate_json(bbox),
                top_k=top_k,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

        results = await _handle_errors(
            _get_service(request).search_spatial(
                query_image,
                payload.bbox.to_retrieval_bbox(),
                payload.top_k,
            )
        )
        return SearchResponse(root=results)

    @app.post("/search/trajectory", response_model=TrajectorySearchResponse)
    async def search_trajectory(
        request: Request,
        payload: TrajectorySearchRequest,
    ) -> TrajectorySearchResponse:
        results = await _handle_errors(
            _get_service(request).search_trajectory(payload.episode_id, payload.top_k)
        )
        return TrajectorySearchResponse(root=results)

    @app.post("/search/transition", response_model=TransitionSearchResponse)
    async def search_transition(
        request: Request,
        state_a: Annotated[UploadFile, File(description="The starting state image.")],
        state_b: Annotated[UploadFile, File(description="The ending state image.")],
        params: Annotated[TransitionSearchRequest, Depends()],
    ) -> TransitionSearchResponse:
        state_a_image = await _read_upload_image(state_a)
        state_b_image = await _read_upload_image(state_b)
        results = await _handle_errors(
            _get_service(request).search_transition(
                state_a_image,
                state_b_image,
                params.top_k,
                params.max_gap_seconds,
            )
        )
        return TransitionSearchResponse(root=results)

    @app.get("/episodes/{episode_id}", response_model=EpisodeDetailsResponse)
    async def get_episode_details(
        request: Request,
        episode_id: str,
    ) -> EpisodeDetailsResponse:
        details = await _handle_errors(_get_service(request).get_episode_details(episode_id))
        return EpisodeDetailsResponse.model_validate(details.model_dump())

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return _get_service(request).health()

    return app


def _get_service(request: Request) -> RetrievalService:
    return request.app.state.service


async def _read_upload_image(upload: UploadFile) -> Image.Image:
    content_type = None if upload.content_type is None else upload.content_type.lower()
    if content_type is not None and content_type not in _ALLOWED_CONTENT_TYPES and content_type != "application/octet-stream":
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG, PNG, or WebP.")

    try:
        return decode_image_bytes(await upload.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _handle_errors(operation: Awaitable[T]) -> T:
    try:
        return await operation
    except KeyError as exc:
        detail = exc.args[0] if exc.args else "Requested resource was not found"
        raise HTTPException(status_code=404, detail=detail) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail="Model ran out of memory while serving the request") from exc
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise HTTPException(
                status_code=503,
                detail="Model ran out of memory while serving the request",
            ) from exc
        raise
