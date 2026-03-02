import heapq
import math

import numpy as np


class DTWMatcher:
    def dtw_distance(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        window: int | None = None,
    ) -> float:
        distance, _ = self._dtw(seq_a, seq_b, window, return_path=False)
        return distance

    def alignment_path(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        window: int | None = None,
    ) -> list[tuple[int, int]]:
        _, path = self._dtw(seq_a, seq_b, window, return_path=True)
        return path

    def lb_keogh(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
        window: int,
    ) -> float:
        normalized_query = self._normalize_sequence(query)
        normalized_candidate = self._normalize_sequence(candidate)
        if window < 0:
            raise ValueError("window must be non-negative")

        query_length = normalized_query.shape[0]
        candidate_length = normalized_candidate.shape[0]
        centers = self._aligned_centers(query_length, candidate_length)

        lower_bound = 0.0
        for query_index, center in enumerate(centers):
            start = max(0, center - window)
            end = min(candidate_length, center + window + 1)
            band = normalized_candidate[start:end]
            lower = np.min(band, axis=0)
            upper = np.max(band, axis=0)
            point = normalized_query[query_index]

            above = np.where(point > upper, point - upper, 0.0)
            below = np.where(point < lower, lower - point, 0.0)
            delta = above + below

            # For unit-normalized vectors, squared Euclidean distance is 2 * cosine distance.
            lower_bound += float(np.dot(delta, delta) * 0.5)

        return lower_bound

    def rank_by_trajectory(
        self,
        query_seq: np.ndarray,
        candidate_seqs: dict[str, np.ndarray],
        top_k: int,
        window_fraction: float = 0.2,
    ) -> list[tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not candidate_seqs:
            return []
        if not 0.0 <= window_fraction <= 1.0:
            raise ValueError("window_fraction must be between 0 and 1")

        lower_bounds: list[tuple[str, float]] = []
        for episode_id, candidate_seq in candidate_seqs.items():
            window = self.window_for_sequences(query_seq, candidate_seq, window_fraction)
            lower_bounds.append((episode_id, self.lb_keogh(query_seq, candidate_seq, window)))
        lower_bounds.sort(key=lambda item: item[1])

        best_distances: dict[str, float] = {}
        heap: list[tuple[float, str]] = []

        for episode_id, lower_bound in lower_bounds:
            threshold = math.inf if len(heap) < top_k else -heap[0][0]
            if lower_bound >= threshold:
                continue

            candidate_seq = candidate_seqs[episode_id]
            window = self.window_for_sequences(query_seq, candidate_seq, window_fraction)
            distance = self.dtw_distance(query_seq, candidate_seq, window)
            best_distances[episode_id] = distance

            if len(heap) < top_k:
                heapq.heappush(heap, (-distance, episode_id))
                continue

            if distance < -heap[0][0]:
                heapq.heapreplace(heap, (-distance, episode_id))

        ranked = [(episode_id, best_distances[episode_id]) for _, episode_id in heap]
        ranked.sort(key=lambda item: item[1])
        return ranked

    def window_for_sequences(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        window_fraction: float,
    ) -> int:
        if not 0.0 <= window_fraction <= 1.0:
            raise ValueError("window_fraction must be between 0 and 1")
        length_a = self._normalize_sequence(seq_a).shape[0]
        length_b = self._normalize_sequence(seq_b).shape[0]
        raw_window = int(math.ceil(max(length_a, length_b) * window_fraction))
        return max(raw_window, abs(length_a - length_b))

    def _dtw(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
        window: int | None,
        return_path: bool,
    ) -> tuple[float, list[tuple[int, int]]]:
        normalized_a = self._normalize_sequence(seq_a)
        normalized_b = self._normalize_sequence(seq_b)
        len_a = normalized_a.shape[0]
        len_b = normalized_b.shape[0]
        band = self._resolve_window(window, len_a, len_b)

        costs = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float32)
        costs[0, 0] = 0.0
        backpointers = np.full((len_a + 1, len_b + 1), -1, dtype=np.int8) if return_path else None

        for row in range(1, len_a + 1):
            col_start = max(1, row - band)
            col_end = min(len_b, row + band)
            for col in range(col_start, col_end + 1):
                cost = 1.0 - float(np.dot(normalized_a[row - 1], normalized_b[col - 1]))
                choices = (
                    costs[row - 1, col],
                    costs[row, col - 1],
                    costs[row - 1, col - 1],
                )
                best_index = int(np.argmin(choices))
                costs[row, col] = cost + float(choices[best_index])
                if backpointers is not None:
                    backpointers[row, col] = best_index

        final_distance = float(costs[len_a, len_b])
        if not np.isfinite(final_distance):
            raise ValueError("window constraint is too tight to produce a valid DTW path")
        if backpointers is None:
            return final_distance, []
        return final_distance, self._trace_path(backpointers, len_a, len_b)

    def _trace_path(
        self,
        backpointers: np.ndarray,
        len_a: int,
        len_b: int,
    ) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        row = len_a
        col = len_b
        while row > 0 and col > 0:
            path.append((row - 1, col - 1))
            pointer = int(backpointers[row, col])
            if pointer == 0:
                row -= 1
            elif pointer == 1:
                col -= 1
            elif pointer == 2:
                row -= 1
                col -= 1
            else:
                raise RuntimeError("encountered an invalid DTW backpointer")
        path.reverse()
        return path

    @staticmethod
    def _aligned_centers(query_length: int, candidate_length: int) -> np.ndarray:
        if candidate_length <= 0:
            raise ValueError("candidate must contain at least one vector")
        if query_length <= 0:
            raise ValueError("query must contain at least one vector")
        if query_length == 1:
            return np.asarray([0], dtype=np.int32)
        return np.rint(np.linspace(0, candidate_length - 1, num=query_length)).astype(np.int32)

    @staticmethod
    def _resolve_window(window: int | None, len_a: int, len_b: int) -> int:
        if window is None:
            return max(len_a, len_b)
        if window < 0:
            raise ValueError("window must be non-negative")
        return max(window, abs(len_a - len_b))

    @staticmethod
    def _normalize_sequence(sequence: np.ndarray) -> np.ndarray:
        matrix = np.asarray(sequence, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("sequence must be a 2D array")
        if matrix.shape[0] == 0:
            raise ValueError("sequence must contain at least one vector")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return np.ascontiguousarray(matrix / safe_norms, dtype=np.float32)
