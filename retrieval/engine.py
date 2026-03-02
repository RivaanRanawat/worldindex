from __future__ import annotations

from typing import Any

import faiss
import numpy as np
import polars as pl
import structlog
from PIL import Image

from compression import TokenCompressor
from retrieval.dtw import DTWMatcher
from retrieval.maxsim import MaxSimScorer
from retrieval.models import BoundingBox, RetrievalConfig, SearchResult, TrajectoryResult
from retrieval.query_encoder import QueryEncoder
from retrieval.shard_reader import CompressedShardReader


class RetrievalEngine:
    def __init__(
        self,
        config: RetrievalConfig,
        query_encoder: QueryEncoder | None = None,
        maxsim_scorer: MaxSimScorer | None = None,
        dtw_matcher: DTWMatcher | None = None,
    ) -> None:
        self.config = config
        self._logger = structlog.get_logger(__name__).bind(component="retrieval_engine")
        self._faiss_index = faiss.read_index(str(self.config.faiss_index_path))
        self._metadata = pl.read_parquet(self.config.metadata_path)
        if "clip_id" not in self._metadata.columns:
            self._metadata = self._metadata.with_row_index("clip_id")
        self._metadata_rows = [self._metadata.row(index, named=True) for index in range(self._metadata.height)]
        self._episode_lookup = self._build_episode_lookup(self._metadata_rows)
        self._compressor = TokenCompressor.load(self.config.compressor_dir)
        self._shard_reader = CompressedShardReader(self.config.shard_dir)
        self._query_encoder = query_encoder if query_encoder is not None else QueryEncoder(self.config.query_encoder)
        self._maxsim_scorer = (
            maxsim_scorer
            if maxsim_scorer is not None
            else MaxSimScorer(
                candidate_batch_size=self.config.maxsim_candidate_batch_size,
                use_gpu_if_available=self.config.maxsim_use_gpu,
            )
        )
        self._dtw_matcher = dtw_matcher if dtw_matcher is not None else DTWMatcher()
        self._trajectory_index: faiss.Index | None = None
        self._trajectory_frame_episode_ids: list[str] = []

    def search_image(
        self,
        image: Image.Image,
        top_k: int = 10,
        coarse_k: int = 100,
    ) -> list[SearchResult]:
        query_tokens, coarse_vector = self._query_encoder.encode_image(image)
        return self._search_by_tokens(query_tokens, coarse_vector, top_k, coarse_k)

    def search_spatial(
        self,
        image: Image.Image,
        bbox: BoundingBox,
        top_k: int = 10,
    ) -> list[SearchResult]:
        query_tokens, coarse_vector = self._query_encoder.encode_image(image)
        selected_indices = bbox.to_patch_indices(self.config.query_encoder.spatial_grid_size)
        spatial_tokens = np.ascontiguousarray(query_tokens[selected_indices], dtype=np.float32)
        return self._search_by_tokens(
            spatial_tokens,
            coarse_vector,
            top_k,
            self.config.coarse_search_k,
        )

    def search_trajectory(
        self,
        episode_id: str,
        top_k: int = 10,
    ) -> list[TrajectoryResult]:
        query_sequence = self._load_trajectory_sequence(episode_id)
        self._ensure_trajectory_index()

        frame_search_k = min(self.config.trajectory_frame_search_k, len(self._trajectory_frame_episode_ids))
        if frame_search_k == 0:
            return []

        normalized_query = self._normalize_rows(query_sequence)
        _, neighbor_indices = self._trajectory_index.search(normalized_query, frame_search_k)

        candidate_episode_ids = {
            self._trajectory_frame_episode_ids[int(index)]
            for index in neighbor_indices.reshape(-1).tolist()
            if index >= 0 and self._trajectory_frame_episode_ids[int(index)] != episode_id
        }
        if not candidate_episode_ids:
            return []

        candidate_sequences = {
            candidate_episode_id: self._load_trajectory_sequence(candidate_episode_id)
            for candidate_episode_id in sorted(candidate_episode_ids)
        }
        ranked = self._dtw_matcher.rank_by_trajectory(query_sequence, candidate_sequences, top_k=top_k)

        results: list[TrajectoryResult] = []
        for candidate_episode_id, distance in ranked:
            candidate_sequence = candidate_sequences[candidate_episode_id]
            window = self._dtw_matcher.window_for_sequences(query_sequence, candidate_sequence, 0.2)
            alignment_path = self._dtw_matcher.alignment_path(query_sequence, candidate_sequence, window)
            dataset_name, robot_type = self._episode_lookup.get(candidate_episode_id, ("", ""))
            results.append(
                TrajectoryResult(
                    episode_id=candidate_episode_id,
                    dataset_name=dataset_name,
                    robot_type=robot_type,
                    dtw_distance=distance,
                    alignment_path=alignment_path,
                )
            )

        return results

    def search_transition(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        top_k: int = 10,
        max_gap_sec: float = 30.0,
    ) -> list[SearchResult]:
        ranked_a = self._scored_image_candidates(image_a, self.config.coarse_search_k)
        ranked_b = self._scored_image_candidates(image_b, self.config.coarse_search_k)

        by_episode_a = self._group_by_episode(ranked_a)
        by_episode_b = self._group_by_episode(ranked_b)
        shared_episode_ids = sorted(set(by_episode_a) & set(by_episode_b))

        episode_rankings: list[tuple[float, int, dict[str, Any]]] = []
        for episode_id in shared_episode_ids:
            best_match: tuple[float, int, dict[str, Any]] | None = None
            for score_a, clip_id_a, row_a in by_episode_a[episode_id]:
                for score_b, clip_id_b, row_b in by_episode_b[episode_id]:
                    gap = float(row_b["timestamp_start"]) - float(row_a["timestamp_end"])
                    if gap < 0.0 or gap > max_gap_sec:
                        continue

                    combined_score = score_a + score_b
                    candidate = (combined_score, clip_id_b, row_b)
                    if best_match is None or candidate[0] > best_match[0]:
                        best_match = candidate

            if best_match is not None:
                episode_rankings.append(best_match)

        episode_rankings.sort(key=lambda item: item[0], reverse=True)
        return [
            self._build_search_result(clip_id=clip_id, row=row, score=score)
            for score, clip_id, row in episode_rankings[:top_k]
        ]

    def _search_by_tokens(
        self,
        query_tokens: np.ndarray,
        coarse_vector: np.ndarray,
        top_k: int,
        coarse_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidate_clip_ids = self._coarse_search(coarse_vector, coarse_k)
        if not candidate_clip_ids:
            return []

        candidate_rows = [self._metadata_rows[clip_id] for clip_id in candidate_clip_ids]
        candidate_clips = [
            self._shard_reader.get_clip(int(row["shard_id"]), int(row["shard_offset"]))
            for row in candidate_rows
        ]
        reranked = self._maxsim_scorer.score_candidates(query_tokens, candidate_clips, self._compressor)

        return [
            self._build_search_result(
                clip_id=candidate_clip_ids[candidate_index],
                row=candidate_rows[candidate_index],
                score=score,
            )
            for candidate_index, score in reranked[:top_k]
        ]

    def _coarse_search(self, coarse_vector: np.ndarray, coarse_k: int) -> list[int]:
        if coarse_k <= 0:
            raise ValueError("coarse_k must be positive")

        query = np.asarray(coarse_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self._faiss_index.d:
            raise ValueError(f"Expected coarse vectors of width {self._faiss_index.d}, received {query.shape[1]}")

        normalized_query = self._normalize_rows(query)
        search_k = min(coarse_k, self._metadata.height)
        _, neighbors = self._faiss_index.search(normalized_query, search_k)
        return [int(index) for index in neighbors[0].tolist() if index >= 0]

    def _scored_image_candidates(
        self,
        image: Image.Image,
        coarse_k: int,
    ) -> list[tuple[float, int, dict[str, Any]]]:
        query_tokens, coarse_vector = self._query_encoder.encode_image(image)
        candidate_clip_ids = self._coarse_search(coarse_vector, coarse_k)
        if not candidate_clip_ids:
            return []

        candidate_rows = [self._metadata_rows[clip_id] for clip_id in candidate_clip_ids]
        candidate_clips = [
            self._shard_reader.get_clip(int(row["shard_id"]), int(row["shard_offset"]))
            for row in candidate_rows
        ]
        reranked = self._maxsim_scorer.score_candidates(query_tokens, candidate_clips, self._compressor)
        return [
            (score, candidate_clip_ids[candidate_index], candidate_rows[candidate_index])
            for candidate_index, score in reranked
        ]

    @staticmethod
    def _group_by_episode(
        ranked_rows: list[tuple[float, int, dict[str, Any]]]
    ) -> dict[str, list[tuple[float, int, dict[str, Any]]]]:
        grouped: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
        for score, clip_id, row in ranked_rows:
            grouped.setdefault(str(row["episode_id"]), []).append((score, clip_id, row))
        return grouped

    def _ensure_trajectory_index(self) -> None:
        if self._trajectory_index is not None:
            return
        if self.config.trajectory_embedding_dir is None:
            raise RuntimeError("trajectory_embedding_dir is required for trajectory search")

        trajectory_paths = sorted(
            path for path in self.config.trajectory_embedding_dir.glob("*.npy") if path.is_file()
        )
        if not trajectory_paths:
            raise FileNotFoundError(f"No trajectory embeddings found in {self.config.trajectory_embedding_dir}")

        all_frames: list[np.ndarray] = []
        self._trajectory_frame_episode_ids = []
        for trajectory_path in trajectory_paths:
            sequence = self._normalize_rows(np.load(trajectory_path))
            all_frames.append(sequence)
            self._trajectory_frame_episode_ids.extend([trajectory_path.stem] * sequence.shape[0])

        frame_matrix = np.ascontiguousarray(np.concatenate(all_frames, axis=0).astype(np.float32, copy=False))
        self._trajectory_index = faiss.IndexFlatIP(frame_matrix.shape[1])
        self._trajectory_index.add(frame_matrix)

    def _load_trajectory_sequence(self, episode_id: str) -> np.ndarray:
        if self.config.trajectory_embedding_dir is None:
            raise RuntimeError("trajectory_embedding_dir is required for trajectory search")

        trajectory_path = self.config.trajectory_embedding_dir / f"{episode_id}.npy"
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Missing trajectory embedding file for episode {episode_id}")
        return np.ascontiguousarray(np.load(trajectory_path).astype(np.float32, copy=False))

    def _build_search_result(
        self,
        clip_id: int,
        row: dict[str, Any],
        score: float,
    ) -> SearchResult:
        return SearchResult(
            clip_id=clip_id,
            episode_id=str(row["episode_id"]),
            dataset_name=str(row["dataset_name"]),
            robot_type=str(row["robot_type"]),
            score=score,
            timestamp_start=float(row["timestamp_start"]),
            timestamp_end=float(row["timestamp_end"]),
            language_instruction=None
            if row.get("language_instruction") is None
            else str(row["language_instruction"]),
        )

    @staticmethod
    def _build_episode_lookup(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
        lookup: dict[str, tuple[str, str]] = {}
        for row in rows:
            episode_id = str(row["episode_id"])
            if episode_id in lookup:
                continue
            lookup[episode_id] = (str(row["dataset_name"]), str(row["robot_type"]))
        return lookup

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return np.ascontiguousarray(matrix / safe_norms, dtype=np.float32)
