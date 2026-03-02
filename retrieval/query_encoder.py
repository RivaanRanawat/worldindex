from __future__ import annotations

from typing import Any

import numpy as np
import structlog
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel
from retrieval.models import QueryEncoderConfig

_PCA_COMPONENTS_FILENAME = "pca_components.npy"
_PCA_MEAN_FILENAME = "pca_mean.npy"


class QueryEncoder:
    def __init__(
        self,
        config: QueryEncoderConfig,
        model: Any | None = None,
        video_processor: Any | None = None,
    ) -> None:
        self.config = config
        self._logger = structlog.get_logger(__name__).bind(component="query_encoder")
        self._model = model if model is not None else self._load_model()
        self._video_processor = (
            video_processor if video_processor is not None else AutoVideoProcessor.from_pretrained(self.config.model_id)
        )
        self._pca_components = np.ascontiguousarray(
            np.load(self.config.pca_artifact_dir / _PCA_COMPONENTS_FILENAME).astype(np.float32, copy=False)
        )
        self._pca_mean = np.ascontiguousarray(
            np.load(self.config.pca_artifact_dir / _PCA_MEAN_FILENAME).astype(np.float32, copy=False)
        )

    def encode_image(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        raw_tokens = self._encode_clip([image] * self.config.clip_length)
        projected_tokens = self._project(raw_tokens)
        spatial_token_count = self.config.spatial_token_count
        midpoint = self.config.midpoint_index
        start = midpoint * spatial_token_count
        end = start + spatial_token_count
        spatial_tokens = np.ascontiguousarray(projected_tokens[start:end], dtype=np.float32)
        coarse_vector = np.ascontiguousarray(projected_tokens.mean(axis=0, dtype=np.float32))
        return spatial_tokens, coarse_vector

    def encode_frame_sequence(self, frames: list[Image.Image]) -> np.ndarray:
        if not frames:
            raise ValueError("frames must contain at least one image")

        coarse_vectors = [self.encode_image(frame)[1] for frame in frames]
        return np.ascontiguousarray(np.stack(coarse_vectors, axis=0).astype(np.float32, copy=False))

    def _encode_clip(self, frames: list[Image.Image]) -> np.ndarray:
        import torch

        processed = dict(self._video_processor(list(frames), return_tensors="pt"))
        pixel_values = processed.get("pixel_values")
        if pixel_values is None:
            pixel_values = processed.get("pixel_values_videos")
        if pixel_values is None:
            raise KeyError("Video processor output must include pixel_values or pixel_values_videos")

        batch = pixel_values
        if not hasattr(batch, "dim"):
            batch = torch.as_tensor(batch)
        if batch.dim() == 4:
            batch = batch.unsqueeze(0)
        if batch.dim() != 5:
            raise ValueError(f"Expected 5D video batch, received shape {tuple(batch.shape)}")

        with torch.no_grad():
            raw_output = self._model.get_vision_features(batch.to(self.config.device))

        encoded = self._extract_tensor(raw_output)
        encoded_numpy = self._to_numpy(encoded)
        if encoded_numpy.ndim == 3:
            if encoded_numpy.shape[0] != 1:
                raise ValueError("encode_image expects a single encoded clip")
            encoded_numpy = encoded_numpy[0]
        if encoded_numpy.shape[0] != self.config.token_count:
            raise ValueError(
                f"Expected {self.config.token_count} tokens, received {encoded_numpy.shape[0]}"
            )
        return np.ascontiguousarray(encoded_numpy.astype(np.float32, copy=False))

    def _project(self, tokens: np.ndarray) -> np.ndarray:
        raw_tokens = np.asarray(tokens, dtype=np.float32)
        if raw_tokens.ndim != 2:
            raise ValueError("tokens must be a 2D array")
        if raw_tokens.shape[1] != self._pca_mean.shape[0]:
            raise ValueError(
                f"Expected token width {self._pca_mean.shape[0]}, received {raw_tokens.shape[1]}"
            )

        centered = raw_tokens - self._pca_mean
        projected = centered @ self._pca_components.T
        return np.ascontiguousarray(projected.astype(np.float32, copy=False))

    def _load_model(self) -> Any:
        import torch

        dtype = torch.float16 if not self.config.device.startswith("cpu") else torch.float32
        model = AutoModel.from_pretrained(self.config.model_id, torch_dtype=dtype)
        model.to(self.config.device)
        model.eval()
        return model

    @staticmethod
    def _extract_tensor(output: Any) -> Any:
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, dict):
            if "last_hidden_state" in output:
                return output["last_hidden_state"]
            if "vision_features" in output:
                return output["vision_features"]
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return np.ascontiguousarray(value.astype(np.float32, copy=False))
        if hasattr(value, "detach"):
            return np.ascontiguousarray(value.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.ascontiguousarray(np.asarray(value, dtype=np.float32))
