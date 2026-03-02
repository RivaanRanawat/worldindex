import re
from pathlib import Path
from typing import Any

import numpy as np

from compression import CompressedClip, SHARD_SUFFIX, read_shard_header


_SHARD_NAME_PATTERN = re.compile(r"^shard_(\d+)\.widx$")
_SHARD_HEADER_SIZE = 32


class CompressedShardReader:
    def __init__(self, shard_dir: Path) -> None:
        self.shard_dir = shard_dir
        self._shard_paths = self._discover_shards()
        self._headers: dict[int, Any] = {}
        self._open_records: dict[int, np.memmap] = {}

    def get_clip(self, shard_id: int, shard_offset: int) -> CompressedClip:
        records = self._records(shard_id)
        header = self._headers[shard_id]
        if shard_offset < 0 or shard_offset >= header.clip_count:
            raise IndexError(f"shard_offset {shard_offset} is out of range for shard {shard_id}")

        record = records[shard_offset]
        return CompressedClip(
            centroid_ids=np.asarray(record["centroid_ids"]),
            quantized_residuals=np.asarray(record["quantized_residuals"]),
            coarse_vector=np.asarray(record["coarse_vector"]),
        )

    @property
    def open_shard_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._open_records))

    def _records(self, shard_id: int) -> np.memmap:
        if shard_id not in self._open_records:
            shard_path = self._shard_paths.get(shard_id)
            if shard_path is None:
                raise FileNotFoundError(f"Missing shard file for shard_id={shard_id}")

            header = read_shard_header(shard_path)
            self._headers[shard_id] = header
            self._open_records[shard_id] = np.memmap(
                shard_path,
                mode="r",
                dtype=header.record_dtype(),
                offset=_SHARD_HEADER_SIZE,
                shape=(header.clip_count,),
            )
        return self._open_records[shard_id]

    def _discover_shards(self) -> dict[int, Path]:
        shard_paths: dict[int, Path] = {}
        for path in sorted(self.shard_dir.glob(f"*{SHARD_SUFFIX}")):
            if not path.is_file():
                continue
            match = _SHARD_NAME_PATTERN.match(path.name)
            if match is None:
                continue
            shard_paths[int(match.group(1))] = path
        return shard_paths
