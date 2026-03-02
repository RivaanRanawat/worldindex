from __future__ import annotations

import numpy as np

from retrieval import DTWMatcher


def test_dtw_prefers_time_warped_sequence_over_unrelated_sequence() -> None:
    matcher = DTWMatcher()
    query = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    time_warped = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    unrelated = np.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    warped_distance = matcher.dtw_distance(query, time_warped, window=2)
    unrelated_distance = matcher.dtw_distance(query, unrelated, window=2)

    assert warped_distance < 1e-4
    assert unrelated_distance > warped_distance + 0.5


def test_lb_keogh_stays_below_dtw_distance_and_ranking_sorts_by_distance() -> None:
    matcher = DTWMatcher()
    rng = np.random.default_rng(1)

    for _ in range(5):
        query = rng.normal(size=(6, 8)).astype(np.float32)
        candidate = rng.normal(size=(7, 8)).astype(np.float32)
        lower_bound = matcher.lb_keogh(query, candidate, window=2)
        distance = matcher.dtw_distance(query, candidate, window=2)
        assert lower_bound <= distance + 1e-5

    query = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidate_near = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    candidate_far = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, -1.0],
            [-1.0, -1.0],
        ],
        dtype=np.float32,
    )

    ranked = matcher.rank_by_trajectory(
        query,
        {
            "near": candidate_near,
            "far": candidate_far,
        },
        top_k=2,
    )

    assert [episode_id for episode_id, _ in ranked] == ["near", "far"]
