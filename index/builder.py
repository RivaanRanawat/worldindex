from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import structlog

from compression.shards import iter_shard_paths, read_coarse_vectors_from_shard


class IndexBuilder:
    def __init__(
        self,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._logger = structlog.get_logger(__name__).bind(component="index_builder")

    def build_faiss_index(self, compressed_dir: Path, output_path: Path) -> np.ndarray:
        shard_paths = iter_shard_paths(compressed_dir)
        if not shard_paths:
            raise FileNotFoundError(f"No compressed shards found in {compressed_dir}")

        coarse_vectors = np.ascontiguousarray(
            np.concatenate(
                [read_coarse_vectors_from_shard(shard_path) for shard_path in shard_paths],
                axis=0,
            ).astype(np.float32, copy=False)
        )
        normalized_vectors = self._normalize_rows(coarse_vectors)

        index = faiss.IndexHNSWFlat(normalized_vectors.shape[1], self.hnsw_m)
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        index.add(normalized_vectors)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
        self._logger.info(
            "faiss_index_built",
            index_path=str(output_path),
            vector_count=normalized_vectors.shape[0],
            dim=normalized_vectors.shape[1],
        )
        return coarse_vectors

    def validate_index(self, index_path: Path, coarse_vectors: np.ndarray) -> float:
        vectors = self._normalize_rows(np.asarray(coarse_vectors, dtype=np.float32))
        if vectors.ndim != 2:
            raise ValueError("coarse_vectors must be a 2D float32 array")

        query_count = min(1_000, vectors.shape[0])
        if query_count == 0:
            raise ValueError("coarse_vectors must contain at least one row")

        k = min(10, vectors.shape[0])
        hnsw_index = faiss.read_index(str(index_path))
        if hnsw_index.d != vectors.shape[1]:
            raise ValueError(
                f"index dimension {hnsw_index.d} does not match coarse_vectors width {vectors.shape[1]}"
            )
        if hasattr(hnsw_index, "hnsw"):
            hnsw_index.hnsw.efSearch = self.ef_search

        brute_force_index = faiss.IndexFlatIP(vectors.shape[1])
        brute_force_index.add(vectors)

        rng = np.random.default_rng(0)
        query_indices = rng.choice(vectors.shape[0], size=query_count, replace=False)
        queries = np.ascontiguousarray(vectors[query_indices])

        _, hnsw_neighbors = hnsw_index.search(queries, k)
        _, brute_force_neighbors = brute_force_index.search(queries, k)

        recall = float(
            np.mean(
                [
                    len(set(hnsw_row.tolist()) & set(brute_row.tolist())) / k
                    for hnsw_row, brute_row in zip(hnsw_neighbors, brute_force_neighbors, strict=True)
                ]
            )
        )
        self._logger.info(
            "faiss_index_validated",
            index_path=str(index_path),
            recall_at_10=recall,
            query_count=query_count,
        )
        assert recall > 0.95, f"recall@10 dropped below target: {recall:.4f}"
        return recall

    def _normalize_rows(self, vectors: np.ndarray) -> np.ndarray:
        row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        return np.ascontiguousarray(vectors / safe_norms, dtype=np.float32)
