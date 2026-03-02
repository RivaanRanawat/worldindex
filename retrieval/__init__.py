from typing import Any

from retrieval.dtw import DTWMatcher
from retrieval.engine import RetrievalEngine
from retrieval.maxsim import MaxSimScorer
from retrieval.models import (
    BoundingBox,
    EpisodeDetails,
    QueryEncoderConfig,
    RetrievalConfig,
    SearchResult,
    TrajectoryResult,
    TransitionResult,
)
from retrieval.shard_reader import CompressedShardReader

__all__ = [
    "BoundingBox",
    "CompressedShardReader",
    "DTWMatcher",
    "EpisodeDetails",
    "MaxSimScorer",
    "QueryEncoder",
    "QueryEncoderConfig",
    "RetrievalConfig",
    "RetrievalEngine",
    "SearchResult",
    "TrajectoryResult",
    "TransitionResult",
]


def __getattr__(name: str) -> Any:
    if name == "QueryEncoder":
        from retrieval.query_encoder import QueryEncoder

        return QueryEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
