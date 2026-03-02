import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the WorldIndex offline pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline.yaml"),
        help="Path to the master pipeline config YAML.",
    )

    subparsers.add_parser("run", parents=[config_parent], help="Run extract, compress, and build-index.")
    subparsers.add_parser("extract", parents=[config_parent], help="Run or resume extraction only.")
    subparsers.add_parser("compress", parents=[config_parent], help="Run or resume compression only.")
    subparsers.add_parser("build-index", parents=[config_parent], help="Build or rebuild the FAISS index only.")
    subparsers.add_parser("serve", parents=[config_parent], help="Start the FastAPI serving process.")
    subparsers.add_parser("status", parents=[config_parent], help="Show current pipeline state.")
    subparsers.add_parser("validate", parents=[config_parent], help="Validate built artifacts.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrator = _load_orchestrator(args.config)

    if args.command == "run":
        print(f"Running pipeline from {args.config}")
        _print_status(orchestrator.run_full_pipeline())
        return 0

    if args.command == "extract":
        print(f"Running extract from {args.config}")
        _print_status(orchestrator.run_stage("extract"))
        return 0

    if args.command == "compress":
        print(f"Running compress from {args.config}")
        _print_status(orchestrator.run_stage("compress"))
        return 0

    if args.command == "build-index":
        print(f"Running build-index from {args.config}")
        _print_status(orchestrator.run_stage("build_index"))
        return 0

    if args.command == "serve":
        print(f"Serving artifacts from {args.config}")
        orchestrator.serve()
        return 0

    if args.command == "status":
        _print_status(orchestrator.get_status())
        return 0

    if args.command == "validate":
        report = orchestrator.validate()
        _print_validation(report.model_dump(mode="json"))
        return 0

    raise ValueError(f"Unsupported command: {args.command}")


def _load_orchestrator(config_path: Path) -> PipelineOrchestrator:
    from pipeline.config import PipelineConfig
    from pipeline.orchestrator import PipelineOrchestrator

    config = PipelineConfig.from_yaml(config_path)
    return PipelineOrchestrator(config)


def _print_status(status: dict[str, Any]) -> None:
    print("Pipeline Status")
    for task in status["tasks"]:
        status_line = (
            f"  - {task['task_name']}: {task['status']}"
            f" | retries={task['retry_count']}"
            f" | checkpoint={task['checkpoint']}"
        )
        if task["error_message"] is not None:
            status_line += f" | error={task['error_message']}"
        print(status_line)

    eta = status["estimated_seconds_remaining"]
    eta_display = "unknown" if eta is None else f"{eta:.3f}s"
    print(f"Total clips processed: {status['total_clips_processed']}")
    print(f"Estimated time remaining: {eta_display}")


def _print_validation(report: dict[str, Any]) -> None:
    print("Validation")
    print(f"Recall@10: {report['recall_at_10']:.4f}")
    print(
        f"Mean cosine similarity: {report['mean_cosine_similarity']:.4f}"
        f" across {report['sampled_clip_count']} clips"
    )
    print("Sample queries:")
    for sample in report["sample_queries"]:
        print(
            f"  - clip {sample['query_clip_id']}:"
            f" top_clip_ids={sample['top_clip_ids']}"
            f" episodes={sample['top_episode_ids']}"
        )
    print("Storage summary:")
    for artifact in report["artifact_sizes"]:
        print(f"  - {artifact['path']}: {_format_bytes(int(artifact['bytes']))}")


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


if __name__ == "__main__":
    raise SystemExit(main())
