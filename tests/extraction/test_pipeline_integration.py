import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from extraction import ExtractionConfig, run_extraction
from ingestion.config import DatasetConfig

if os.environ.get("WORLDINDEX_RUN_REAL_EXTRACTION") != "1":
    pytest.skip(
        "Set WORLDINDEX_RUN_REAL_EXTRACTION=1 and install the video dependency group to run live extraction tests.",
        allow_module_level=True,
    )


def _load_droid_config() -> DatasetConfig:
    config_path = Path("config/datasets/droid.yaml")
    return DatasetConfig.from_yaml(config_path)


@pytest.mark.integration
def test_real_droid_clips_flow_through_pipeline(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config = _load_droid_config()

    output_dir = tmp_path / "output"
    checkpoint_db = tmp_path / "checkpoint.sqlite3"
    extraction_config = ExtractionConfig(
        dataset_configs=[config],
        model_id="fake",
        device="cpu",
        batch_size=1,
        queue_depth=2,
        flush_size=5,
        clip_former_factory="tests.extraction.spawn_fakes:build_limited_real_clip_former",
        model_loader="tests.extraction.spawn_fakes:load_full_shape_fake_encoder_model",
        output_dir=output_dir,
        checkpoint_db=checkpoint_db,
    )

    final_checkpoint = run_extraction(extraction_config)

    assert final_checkpoint == 19
    token_paths = sorted(output_dir.glob("tokens_*.npy"))
    assert token_paths
    for path in token_paths:
        tokens = np.load(path)
        assert tokens.shape[1:] == (8192, 1280)
        assert tokens.dtype == np.float32

    metadata_paths = sorted(output_dir.glob("metadata_*.parquet"))
    metadata_frame = pl.concat([pl.read_parquet(path) for path in metadata_paths], how="vertical")
    assert metadata_frame.height == 20
    assert metadata_frame.columns == [
        "episode_id",
        "dataset_name",
        "robot_type",
        "clip_start_frame",
        "clip_end_frame",
        "timestamp_start",
        "timestamp_end",
        "language_instruction",
        "num_original_frames",
        "clip_index",
    ]
