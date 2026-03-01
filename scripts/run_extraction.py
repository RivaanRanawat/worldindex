import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction import ExtractionConfig, run_extraction
from ingestion.config import DatasetConfig


def main() -> None:
    is_gpu = len(sys.argv) > 1 and sys.argv[1].lower() == 'gpu'
    print(f"is_gpu: {is_gpu}")
    if is_gpu:
        config = ExtractionConfig(
            dataset_configs=[DatasetConfig.from_yaml(ROOT / "config/datasets/droid.yaml")],
            model_id="facebook/vjepa2-vith-fpc64-256",
            device="cuda",
            batch_size=4,
            queue_depth=8,
            flush_size=25,
            output_dir=ROOT / "artifacts/extraction/gpu/droid",
            checkpoint_db=ROOT / "state/gpu/extraction_droid.sqlite3",
        )
    else:
        config = ExtractionConfig(
            dataset_configs=[DatasetConfig.from_yaml(ROOT / "config/datasets/droid.yaml")],
            model_id="facebook/vjepa2-vith-fpc64-256",
            device="cpu",
            batch_size=1,
            queue_depth=8,
            flush_size=25,
            output_dir=ROOT / "artifacts/extraction/droid",
            checkpoint_db=ROOT / "state/extraction_droid.sqlite3",
        )

    last_clip_index = run_extraction(config)
    print({"last_clip_index": last_clip_index, "output_dir": str(config.output_dir)})


if __name__ == "__main__":
    main()
