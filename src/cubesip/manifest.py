"""CSV manifest of everything cubesip has downloaded.

One row per variable per NetCDF file; all rows written by the same run share ``request_id``.
Appends are guarded by a lock file next to the CSV so several users/machines can share it.

Implemented in milestone M1 (read/append) and M2 (cache lookup).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cubesip.request import Request

COLUMNS = [
    "request_id",  # uuid4 hex, shared by the rows of one file
    "timestamp_utc",  # ISO 8601, when the file was written
    "user",  # getpass.getuser()
    "hostname",  # socket.gethostname()
    "driver",
    "dataset",
    "variable",  # one row per variable
    "geohash",
    "time_start",  # inclusive day, empty for static datasets
    "time_end",  # inclusive day, empty for static datasets
    "options",  # canonical JSON (sorted keys) of request.options
    "crs",  # native CRS of the file, e.g. EPSG:32632
    "file_size_bytes",  # size of the whole file (repeated on each row)
    "uri",  # file:// URI of the NetCDF
    "cubesip_version",
]


@dataclass(frozen=True)
class CacheHit:
    """A manifest file that fully covers a request."""

    uri: str
    path: Path
    geohash: str  # geohash of the cached file (equal to or an ancestor of the request's)
    request_id: str


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def read(self) -> pd.DataFrame:
        """All rows (empty frame with COLUMNS if the file does not exist yet)."""
        raise NotImplementedError("M1")

    def append(self, rows: list[dict[str, str]]) -> None:
        """Append rows under the file lock, writing the header if the file is new."""
        raise NotImplementedError("M1")

    def find_covering(self, request: Request) -> CacheHit | None:
        """Best cached file covering the request (see IMPLEMENTATION_PLAN.md, cache rules)."""
        raise NotImplementedError("M2")
