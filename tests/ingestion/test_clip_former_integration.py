import os
from pathlib import Path
import pytest

from ingestion import ClipFormer, DatasetConfig

if os.environ.get("WORLDINDEX_RUN_REAL_INGESTION") != "1":
    pytest.skip(
        "Set WORLDINDEX_RUN_REAL_INGESTION=1 and install the optional video dependency group to run live ingestion tests.",
        allow_module_level=True,
    )


def _load_integration_config() -> DatasetConfig:
    dataset_name = os.environ.get("WORLDINDEX_REAL_INGESTION_DATASET", "droid")
    config_path = Path("config/datasets") / f"{dataset_name}.yaml"
    if not config_path.is_file():
        raise ValueError(
            f"Unsupported integration dataset {dataset_name!r}; expected a config in {config_path.parent}"
        )
    return DatasetConfig.from_yaml(config_path)


@pytest.mark.integration
def test_real_streaming_clip_shapes() -> None:
    config = _load_integration_config()
    clip_former = ClipFormer(config)

    unique_episodes = 0
    seen_episode_ids: set[str] = set()
    for processed, metadata in clip_former.iter_clips():
        if metadata.episode_id in seen_episode_ids:
            continue

        seen_episode_ids.add(metadata.episode_id)
        unique_episodes += 1
        assert tuple(processed["pixel_values"].shape) == (64, 3, 256, 256)

        if unique_episodes == 5:
            break

    assert unique_episodes == 5
