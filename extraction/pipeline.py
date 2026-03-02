import importlib
import sqlite3
import time
from multiprocessing import get_all_start_methods, get_context, resource_tracker
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Full
from typing import Any, Literal, NamedTuple

import numpy as np
import polars as pl
import structlog
from pydantic import BaseModel, ConfigDict, Field

from ingestion.config import DatasetConfig

_CHECKPOINT_ROW_KEY = "single_node_extraction"
_PROGRESS_LOG_INTERVAL_SECONDS = 5.0
_JOIN_POLL_SECONDS = 0.5


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_configs: list[DatasetConfig] = Field(min_length=1)
    model_id: str
    device: str = "cpu"
    batch_size: int = Field(default=1, gt=0)
    queue_depth: int = Field(default=8, gt=0)
    flush_size: int = Field(default=100, gt=0)
    start_method: Literal["spawn", "fork", "forkserver"] | None = None
    clip_former_factory: str | None = None
    model_loader: str | None = None
    output_dir: Path
    checkpoint_db: Path


class SharedClipDescriptor(NamedTuple):
    shm_name: str
    shape: tuple[int, ...]
    dtype: str
    metadata: dict[str, Any]


def producer_fn(
    dataset_configs: list[DatasetConfig],
    q1: Queue,
    start_clip_index: int = 0,
    stop_event: Event | None = None,
    clip_former_factory: str | None = None,
) -> None:
    logger = structlog.get_logger(__name__).bind(worker="producer")
    clip_index = 0

    for dataset_config in dataset_configs:
        if stop_event is not None and stop_event.is_set():
            break

        clip_former = _build_clip_former(dataset_config, clip_former_factory)
        for processed, metadata in clip_former.iter_clips():
            if stop_event is not None and stop_event.is_set():
                break

            if clip_index < start_clip_index:
                clip_index += 1
                continue

            pixel_values = _to_numpy_array(processed["pixel_values"])
            metadata_dict = metadata.model_dump(mode="python")
            metadata_dict["clip_index"] = clip_index

            shared_memory = SharedMemory(create=True, size=pixel_values.nbytes)
            enqueued = False
            try:
                shared_array = np.ndarray(
                    pixel_values.shape,
                    dtype=pixel_values.dtype,
                    buffer=shared_memory.buf,
                )
                shared_array[...] = pixel_values
                descriptor = SharedClipDescriptor(
                    shm_name=shared_memory.name,
                    shape=tuple(pixel_values.shape),
                    dtype=str(pixel_values.dtype),
                    metadata=metadata_dict,
                )
                q1.put(descriptor)
                resource_tracker.unregister(shared_memory._name, "shared_memory")
                enqueued = True
            finally:
                shared_memory.close()
                if not enqueued:
                    shared_memory.unlink()

            clip_index += 1

    logger.info("producer_complete", next_clip_index=clip_index)
    q1.put(None)


def encoder_fn(
    q1: Queue,
    q2: Queue,
    model_id: str,
    device: str,
    batch_size: int,
    stop_event: Event | None = None,
    model_loader: str | None = None,
) -> None:
    logger = structlog.get_logger(__name__).bind(worker="encoder", device=device)
    model = _load_encoder_model(
        model_id=model_id,
        device=device,
        model_loader=model_loader,
    )
    pending_tensors: list[Any] = []
    pending_metadata: list[dict[str, Any]] = []

    while True:
        if stop_event is not None and stop_event.is_set():
            _offer_sentinel(q1)

        item = q1.get()
        if item is None:
            break

        descriptor = SharedClipDescriptor(*item)
        shared_memory = SharedMemory(name=descriptor.shm_name)
        try:
            shared_array = np.ndarray(
                descriptor.shape,
                dtype=np.dtype(descriptor.dtype),
                buffer=shared_memory.buf,
            )
            pending_tensors.append(_to_tensor(shared_array))
            pending_metadata.append(descriptor.metadata)
        finally:
            shared_memory.close()
            shared_memory.unlink()

        if len(pending_tensors) >= batch_size:
            _encode_batch(
                model=model,
                clips=pending_tensors,
                metadata=pending_metadata,
                q2=q2,
                device=device,
            )
            pending_tensors = []
            pending_metadata = []

    if pending_tensors:
        _encode_batch(
            model=model,
            clips=pending_tensors,
            metadata=pending_metadata,
            q2=q2,
            device=device,
        )

    logger.info("encoder_complete")
    q2.put(None)


def writer_fn(
    q2: Queue,
    output_dir: Path,
    checkpoint_db_path: Path,
    flush_size: int = 100,
) -> None:
    logger = structlog.get_logger(__name__).bind(worker="writer")
    output_dir.mkdir(parents=True, exist_ok=True)
    _initialize_checkpoint_db(checkpoint_db_path)

    token_buffer: list[np.ndarray] = []
    metadata_buffer: list[dict[str, Any]] = []

    while True:
        item = q2.get()
        if item is None:
            break

        tokens, metadata = item
        token_buffer.append(np.asarray(tokens, dtype=np.float32))
        metadata_buffer.append(dict(metadata))

        if len(token_buffer) >= flush_size:
            _flush_buffers(
                token_buffer=token_buffer,
                metadata_buffer=metadata_buffer,
                output_dir=output_dir,
                checkpoint_db_path=checkpoint_db_path,
            )
            logger.info(
                "writer_flushed",
                last_clip_index=metadata_buffer[-1]["clip_index"],
                clips_written=len(token_buffer),
            )
            token_buffer = []
            metadata_buffer = []

    if token_buffer:
        _flush_buffers(
            token_buffer=token_buffer,
            metadata_buffer=metadata_buffer,
            output_dir=output_dir,
            checkpoint_db_path=checkpoint_db_path,
        )
        logger.info(
            "writer_flushed_partial",
            last_clip_index=metadata_buffer[-1]["clip_index"],
            clips_written=len(token_buffer),
        )

    logger.info("writer_complete")


def run_extraction(
    config: ExtractionConfig,
    checkpoint: int | None = None,
) -> int:
    logger = structlog.get_logger(__name__).bind(component="single_node_extraction")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _initialize_checkpoint_db(config.checkpoint_db)

    persisted_checkpoint = _read_checkpoint(config.checkpoint_db)
    if checkpoint is not None and checkpoint > persisted_checkpoint:
        _write_checkpoint(config.checkpoint_db, checkpoint)
        persisted_checkpoint = checkpoint

    resume_from = persisted_checkpoint + 1
    context = _select_context(config.start_method)
    q1 = context.Queue(maxsize=config.queue_depth)
    q2 = context.Queue(maxsize=config.queue_depth)
    stop_event = context.Event()

    producer = context.Process(
        name="worldindex-producer",
        target=producer_fn,
        args=(config.dataset_configs, q1, resume_from),
        kwargs={
            "stop_event": stop_event,
            "clip_former_factory": config.clip_former_factory,
        },
    )
    encoder = context.Process(
        name="worldindex-encoder",
        target=encoder_fn,
        args=(q1, q2, config.model_id, config.device, config.batch_size),
        kwargs={
            "stop_event": stop_event,
            "model_loader": config.model_loader,
        },
    )
    writer = context.Process(
        name="worldindex-writer",
        target=writer_fn,
        args=(q2, config.output_dir, config.checkpoint_db),
        kwargs={"flush_size": config.flush_size},
    )
    processes = [producer, encoder, writer]

    for process in processes:
        process.start()

    interrupted = False
    failure_message: str | None = None
    last_progress_log = time.monotonic()

    try:
        while True:
            for process in processes:
                process.join(timeout=_JOIN_POLL_SECONDS)

            failed_processes = [
                process
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed_processes:
                failed = failed_processes[0]
                failure_message = (
                    f"{failed.name} exited with code {failed.exitcode}"
                )
                _signal_stop(stop_event=stop_event, q1=q1, q2=q2, force_writer=True)
                break

            now = time.monotonic()
            if now - last_progress_log >= _PROGRESS_LOG_INTERVAL_SECONDS:
                checkpoint = _read_checkpoint(config.checkpoint_db)
                logger.info(
                    "extraction_progress",
                    resume_from=resume_from,
                    processed_clips=max(checkpoint + 1, 0),
                    last_clip_index=checkpoint,
                )
                last_progress_log = now

            if all(not process.is_alive() for process in processes):
                break
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "keyboard_interrupt",
            last_checkpoint=_read_checkpoint(config.checkpoint_db),
        )
        _signal_stop(stop_event=stop_event, q1=q1, q2=q2, force_writer=False)
    finally:
        _join_or_terminate(processes, q1=q1, q2=q2)

    final_checkpoint = _read_checkpoint(config.checkpoint_db)
    if failure_message is not None:
        raise RuntimeError(failure_message)
    if interrupted:
        logger.info("extraction_stopped", last_clip_index=final_checkpoint)
    else:
        logger.info("extraction_complete", last_clip_index=final_checkpoint)
    return final_checkpoint


def _build_clip_former(
    config: DatasetConfig,
    clip_former_factory: str | None = None,
) -> Any:
    if clip_former_factory is not None:
        factory = _load_callable(clip_former_factory)
        return factory(config)

    from ingestion import ClipFormer

    return ClipFormer(config)


def _load_encoder_model(
    model_id: str,
    device: str,
    model_loader: str | None = None,
) -> Any:
    if model_loader is not None:
        loader = _load_callable(model_loader)
        return loader(model_id, device)

    import torch
    from transformers import AutoModel

    dtype = torch.float16 if not device.startswith("cpu") else torch.float32
    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    model.to(device)
    model.eval()
    return model


def _encode_batch(
    model: Any,
    clips: list[Any],
    metadata: list[dict[str, Any]],
    q2: Queue,
    device: str,
) -> None:
    import torch

    batch = torch.stack(clips, dim=0).to(device)
    with torch.no_grad():
        raw_output = model.get_vision_features(batch)

    encoded = _extract_tensor(raw_output)
    encoded_numpy = encoded.to(device="cpu", dtype=torch.float32).numpy()

    for clip_tokens, clip_metadata in zip(encoded_numpy, metadata, strict=True):
        q2.put((np.ascontiguousarray(clip_tokens), clip_metadata))


def _extract_tensor(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        if "last_hidden_state" in output:
            return output["last_hidden_state"]
        return output["vision_features"]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


def _to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    return np.ascontiguousarray(value.detach().cpu().numpy())


def _to_tensor(value: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(np.array(value, copy=True))


def _flush_buffers(
    token_buffer: list[np.ndarray],
    metadata_buffer: list[dict[str, Any]],
    output_dir: Path,
    checkpoint_db_path: Path,
) -> None:
    first_clip_index = int(metadata_buffer[0]["clip_index"])
    last_clip_index = int(metadata_buffer[-1]["clip_index"])
    token_path = output_dir / f"tokens_{first_clip_index:08d}_{last_clip_index:08d}.npy"
    metadata_path = output_dir / f"metadata_{first_clip_index:08d}_{last_clip_index:08d}.parquet"

    stacked_tokens = np.stack(token_buffer, axis=0).astype(np.float32, copy=False)
    metadata_frame = pl.DataFrame(metadata_buffer)

    temp_token_path = token_path.with_suffix(".npy.tmp")
    temp_metadata_path = metadata_path.with_suffix(".parquet.tmp")

    with temp_token_path.open("wb") as token_file:
        np.save(token_file, stacked_tokens, allow_pickle=False)
    metadata_frame.write_parquet(temp_metadata_path)

    temp_token_path.replace(token_path)
    temp_metadata_path.replace(metadata_path)
    _write_checkpoint(checkpoint_db_path, last_clip_index)


def _initialize_checkpoint_db(checkpoint_db_path: Path) -> None:
    checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(checkpoint_db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS extraction_checkpoint (
                checkpoint_key TEXT PRIMARY KEY,
                clip_index INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO extraction_checkpoint (checkpoint_key, clip_index, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (_CHECKPOINT_ROW_KEY, -1),
        )
        connection.commit()


def _read_checkpoint(checkpoint_db_path: Path) -> int:
    with sqlite3.connect(checkpoint_db_path) as connection:
        row = connection.execute(
            """
            SELECT clip_index
            FROM extraction_checkpoint
            WHERE checkpoint_key = ?
            """,
            (_CHECKPOINT_ROW_KEY,),
        ).fetchone()
    return int(row[0])


def _write_checkpoint(checkpoint_db_path: Path, clip_index: int) -> None:
    with sqlite3.connect(checkpoint_db_path) as connection:
        connection.execute(
            """
            INSERT INTO extraction_checkpoint (checkpoint_key, clip_index, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(checkpoint_key) DO UPDATE SET
                clip_index = excluded.clip_index,
                updated_at = CURRENT_TIMESTAMP
            """,
            (_CHECKPOINT_ROW_KEY, clip_index),
        )
        connection.commit()


def _signal_stop(
    stop_event: Event,
    q1: Queue,
    q2: Queue,
    force_writer: bool,
) -> None:
    stop_event.set()
    _offer_sentinel(q1)
    if force_writer:
        _offer_sentinel(q2)


def _offer_sentinel(queue: Queue) -> None:
    try:
        queue.put_nowait(None)
    except Full:
        pass


def _join_or_terminate(
    processes: list[Any],
    q1: Queue,
    q2: Queue,
) -> None:
    deadline = time.monotonic() + (_PROGRESS_LOG_INTERVAL_SECONDS * 2)
    while time.monotonic() < deadline:
        alive_processes = [process for process in processes if process.is_alive()]
        if not alive_processes:
            break

        for process in alive_processes:
            process.join(timeout=_JOIN_POLL_SECONDS)

    if any(process.is_alive() for process in processes):
        _offer_sentinel(q1)
        _offer_sentinel(q2)

    for process in processes:
        if process.is_alive():
            process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)

    q1.close()
    q2.close()
    q1.join_thread()
    q2.join_thread()


def _select_context(start_method: str | None = None) -> BaseContext:
    if start_method is not None:
        if start_method not in get_all_start_methods():
            raise ValueError(f"Unsupported multiprocessing start method: {start_method}")
        return get_context(start_method)

    if "spawn" in get_all_start_methods():
        return get_context("spawn")
    return get_context(get_all_start_methods()[0])

def _load_callable(import_path: str) -> Any:
    module_path, attribute = import_path.split(":", maxsplit=1)
    module = importlib.import_module(module_path)
    return getattr(module, attribute)
