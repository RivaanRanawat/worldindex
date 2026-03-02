from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from retrieval import QueryEncoder, QueryEncoderConfig


class _FakeProcessor:
    def __call__(self, frames: list[Image.Image], return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        pixel_value = float(np.asarray(frames[0], dtype=np.float32).mean())
        return {
            "pixel_values": torch.full(
                (len(frames), 3, 2, 2),
                pixel_value,
                dtype=torch.float32,
            )
        }


class _FakeModel:
    def eval(self) -> _FakeModel:
        return self

    def to(self, device: str) -> _FakeModel:
        return self

    def get_vision_features(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = batch.shape[0]
        base = batch.sum(dim=(1, 2, 3, 4), keepdim=False).reshape(batch_size, 1, 1)
        tokens = torch.arange(batch_size * 8 * 4, dtype=torch.float32).reshape(batch_size, 8, 4)
        return {"last_hidden_state": tokens + base}


def test_query_encoder_projects_midpoint_spatial_tokens_and_frame_sequences(tmp_path: Path) -> None:
    pca_dir = tmp_path / "artifacts"
    pca_dir.mkdir()
    np.save(
        pca_dir / "pca_components.npy",
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        allow_pickle=False,
    )
    np.save(pca_dir / "pca_mean.npy", np.zeros(4, dtype=np.float32), allow_pickle=False)

    encoder = QueryEncoder(
        QueryEncoderConfig(
            pca_artifact_dir=pca_dir,
            clip_length=4,
            temporal_positions=2,
            spatial_grid_size=2,
        ),
        model=_FakeModel(),
        video_processor=_FakeProcessor(),
    )

    dark_image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    bright_image = Image.fromarray(np.full((4, 4, 3), 2, dtype=np.uint8))

    spatial_tokens, coarse_vector = encoder.encode_image(dark_image)
    expected_tokens = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)

    np.testing.assert_allclose(spatial_tokens, expected_tokens[4:, :3])
    np.testing.assert_allclose(coarse_vector, expected_tokens[:, :3].mean(axis=0))

    frame_sequence = encoder.encode_frame_sequence([dark_image, bright_image])
    assert frame_sequence.shape == (2, 3)
    assert frame_sequence[1, 0] > frame_sequence[0, 0]
