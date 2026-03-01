import os
import pytest
from ingestion import ClipFormer, DatasetConfig

if os.environ.get("WORLDINDEX_RUN_REAL_INGESTION") != "1":
    pytest.skip(
        "Set WORLDINDEX_RUN_REAL_INGESTION=1 and install the optional video dependency group to run live ingestion tests.",
        allow_module_level=True,
    )


@pytest.mark.integration
def test_real_droid_streaming_clip_shapes() -> None:
    config = DatasetConfig(
        repo_id="lerobot/droid_1.0.1",
        image_key="observation.images.exterior_1_left",
        source_fps=15,
        robot_type="franka",
        language_key="language_instruction",
    )
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
