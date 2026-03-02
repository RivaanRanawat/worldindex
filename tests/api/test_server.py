from __future__ import annotations

import asyncio

import httpx
from fastapi.testclient import TestClient

from api.server import create_app
from api.service import RetrievalService
from tests.api.fakes import FakeEngine, make_test_image_bytes


def test_image_search_endpoint_returns_enriched_results() -> None:
    service = RetrievalService(FakeEngine(), max_concurrent_queries=2)
    client = TestClient(create_app(service=service))

    response = client.post(
        "/search/image",
        params={"top_k": 2, "coarse_k": 4},
        files={"image": ("query.png", make_test_image_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["episode_id"] == "episode-0"
    assert payload[0]["dataset_name"] == "droid"
    assert payload[0]["robot_type"] == "franka"
    assert payload[0]["timestamp_start"] == 0.0
    assert payload[0]["timestamp_end"] == 1.0
    assert payload[0]["language_instruction"] == "pick up the block"


def test_concurrency_limit_serializes_image_queries() -> None:
    engine = FakeEngine(search_delay_seconds=0.05)
    service = RetrievalService(engine, max_concurrent_queries=1)
    app = create_app(service=service)

    async def run_requests() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await asyncio.gather(
                *[
                    client.post(
                        "/search/image",
                        params={"top_k": 1, "coarse_k": 2},
                        files={"image": ("query.png", make_test_image_bytes(), "image/png")},
                    )
                    for _ in range(5)
                ]
            )

    responses = asyncio.run(run_requests())

    assert all(response.status_code == 200 for response in responses)
    assert engine.max_active_image_searches == 1


def test_health_endpoint_reports_indexed_clip_count() -> None:
    service = RetrievalService(FakeEngine(), max_concurrent_queries=2)
    client = TestClient(create_app(service=service))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["indexed_clips"] == 6
    assert payload["model_id"] == "fake-vjepa2"
    assert payload["device"] == "cpu"


def test_episode_endpoint_returns_404_for_unknown_episode() -> None:
    service = RetrievalService(FakeEngine(), max_concurrent_queries=2)
    client = TestClient(create_app(service=service))

    response = client.get("/episodes/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown episode_id: missing"
