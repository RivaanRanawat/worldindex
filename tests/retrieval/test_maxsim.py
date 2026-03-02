import numpy as np

from compression import TokenCompressor
from retrieval import MaxSimScorer


def test_maxsim_reranks_exact_match_first_and_is_batch_invariant() -> None:
    rng = np.random.default_rng(0)
    training_tokens = rng.normal(size=(512, 4)).astype(np.float32)
    compressor = TokenCompressor(pca_dim=4, n_centroids=16)
    compressor.train(training_tokens)

    raw_clip_a = np.vstack(
        [
            np.tile(np.asarray([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32), (128, 1)),
            np.tile(np.asarray([[0.0, 3.0, 0.0, 0.0]], dtype=np.float32), (128, 1)),
        ]
    )
    raw_clip_b = np.vstack(
        [
            np.tile(np.asarray([[-3.0, 0.0, 0.0, 0.0]], dtype=np.float32), (128, 1)),
            np.tile(np.asarray([[0.0, -3.0, 0.0, 0.0]], dtype=np.float32), (128, 1)),
        ]
    )
    raw_clip_c = np.tile(np.asarray([[0.0, 0.0, 3.0, 0.0]], dtype=np.float32), (256, 1))

    candidate_clips = [
        compressor.compress_clip(raw_clip_a),
        compressor.compress_clip(raw_clip_b),
        compressor.compress_clip(raw_clip_c),
    ]
    query_tokens = compressor.decompress_clip(candidate_clips[0])

    single_batch_scores = MaxSimScorer(
        candidate_batch_size=3,
        use_gpu_if_available=False,
    ).score_candidates(query_tokens, candidate_clips, compressor)
    chunked_scores = MaxSimScorer(
        candidate_batch_size=1,
        use_gpu_if_available=False,
    ).score_candidates(query_tokens, candidate_clips, compressor)

    assert single_batch_scores[0][0] == 0
    assert single_batch_scores[0][1] > single_batch_scores[1][1]
    assert single_batch_scores == chunked_scores
