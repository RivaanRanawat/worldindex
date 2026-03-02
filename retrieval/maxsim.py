import numpy as np
import structlog

from compression import CompressedClip, TokenCompressor


class MaxSimScorer:
    def __init__(
        self,
        candidate_batch_size: int = 25,
        use_gpu_if_available: bool = True,
        torch_device: str = "cuda",
    ) -> None:
        if candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        self.candidate_batch_size = candidate_batch_size
        self.use_gpu_if_available = use_gpu_if_available
        self.torch_device = torch_device
        self._logger = structlog.get_logger(__name__).bind(component="maxsim_scorer")

    def score_candidates(
        self,
        query_tokens: np.ndarray,
        candidate_clips: list[CompressedClip],
        compressor: TokenCompressor,
    ) -> list[tuple[int, float]]:
        if not candidate_clips:
            return []

        normalized_query = self._normalize_rows(query_tokens)
        if normalized_query.ndim != 2:
            raise ValueError("query_tokens must be a 2D array")

        scored_candidates: list[tuple[int, float]] = []
        for batch_start in range(0, len(candidate_clips), self.candidate_batch_size):
            batch = candidate_clips[batch_start : batch_start + self.candidate_batch_size]
            decompressed = [
                self._normalize_rows(compressor.decompress_clip(candidate_clip))
                for candidate_clip in batch
            ]
            concatenated = np.ascontiguousarray(
                np.concatenate(decompressed, axis=0).astype(np.float32, copy=False)
            )
            similarities = self._matmul(normalized_query, concatenated)

            offset = 0
            for local_index, candidate_tokens in enumerate(decompressed):
                next_offset = offset + candidate_tokens.shape[0]
                max_per_query = np.max(similarities[:, offset:next_offset], axis=1)
                score = float(np.sum(max_per_query, dtype=np.float32))
                scored_candidates.append((batch_start + local_index, score))
                offset = next_offset

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates

    def _matmul(self, normalized_query: np.ndarray, normalized_candidates: np.ndarray) -> np.ndarray:
        if self.use_gpu_if_available:
            gpu_result = self._torch_matmul(normalized_query, normalized_candidates)
            if gpu_result is not None:
                return gpu_result

        return np.ascontiguousarray(normalized_query @ normalized_candidates.T, dtype=np.float32)

    def _torch_matmul(
        self,
        normalized_query: np.ndarray,
        normalized_candidates: np.ndarray,
    ) -> np.ndarray | None:
        try:
            import torch
        except ImportError:
            return None

        if not torch.cuda.is_available():
            return None

        query_tensor = torch.from_numpy(normalized_query).to(self.torch_device)
        candidate_tensor = torch.from_numpy(normalized_candidates.T.copy()).to(self.torch_device)
        similarities = query_tensor @ candidate_tensor
        return np.ascontiguousarray(
            similarities.to(device="cpu", dtype=torch.float32).numpy(),
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return np.ascontiguousarray(matrix / safe_norms, dtype=np.float32)
