"""CSV manifest of everything cubesip has downloaded.

One row per variable per NetCDF file; all rows written by the same run share ``request_id``.
Appends are guarded by a lock file next to the CSV so several users/machines can share it.
"""

from __future__ import annotations

import csv
import getpass
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import pandas as pd
from filelock import FileLock

from cubesip import geohash as gh
from cubesip.request import Request

COLUMNS = [
    "request_id",  # uuid4 hex, shared by the rows of one file
    "timestamp_utc",  # ISO 8601, when the file was written
    "user",  # getpass.getuser()
    "hostname",  # socket.gethostname()
    "driver",
    "dataset",  # normalised by the driver (see Driver.dataset_key)
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

LOCK_TIMEOUT_S = 60


@dataclass(frozen=True)
class CacheHit:
    """A manifest file that fully covers a request."""

    uri: str
    path: Path
    geohash: str  # geohash of the cached file (equal to or an ancestor of the request's)
    request_id: str
    timestamp_utc: str


def path_to_uri(path: Path) -> str:
    return Path(path).resolve().as_uri()


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"only file:// URIs are supported, got {uri!r}")
    path = unquote(parsed.path)
    if parsed.netloc and parsed.netloc != "localhost":  # UNC share: file://server/share/x
        return Path(f"//{parsed.netloc}{url2pathname(path)}")
    return Path(url2pathname(path))


def make_rows(
    request: Request,
    *,
    dataset_key: str,
    path: Path,
    crs: str,
    request_id: str | None = None,
) -> list[dict[str, str]]:
    """Manifest rows (one per variable) for a file that was just written."""
    from cubesip import __version__

    common = {
        "request_id": request_id or uuid.uuid4().hex,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "user": _safe(getpass.getuser),
        "hostname": _safe(socket.gethostname),
        "driver": request.driver,
        "dataset": dataset_key,
        "geohash": request.geohash,
        "time_start": request.time.start.isoformat() if request.time else "",
        "time_end": request.time.end.isoformat() if request.time else "",
        "options": request.options_key,
        "crs": crs,
        "file_size_bytes": str(Path(path).stat().st_size),
        "uri": path_to_uri(path),
        "cubesip_version": __version__,
    }
    return [{**common, "variable": variable} for variable in request.variables]


def _safe(func) -> str:
    try:
        return str(func())
    except Exception:  # getuser() can fail in some service accounts
        return ""


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def read(self) -> pd.DataFrame:
        """All rows as strings (empty frame with COLUMNS if the file does not exist yet)."""
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return pd.DataFrame({column: pd.Series(dtype=str) for column in COLUMNS})
        frame = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        for column in COLUMNS:  # tolerate manifests written by older versions
            if column not in frame:
                frame[column] = ""
        return frame

    def append(self, rows: list[dict[str, str]]) -> None:
        """Append rows under the file lock, writing the header if the file is new."""
        unknown = {key for row in rows for key in row} - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown manifest columns: {sorted(unknown)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock_path, timeout=LOCK_TIMEOUT_S):
            new_file = not self.path.is_file() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="")
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)

    def find_covering(self, request: Request, dataset_key: str) -> CacheHit | None:
        """Best cached file covering the request.

        A file covers the request when its rows have the same driver, dataset and options,
        its geohash is the request's or an ancestor, its period contains the requested days
        (or both are static), it holds every requested variable and it still exists.
        The longest geohash wins, then the most recent file.
        """
        frame = self.read()
        if frame.empty:
            return None
        frame = frame[
            (frame["driver"] == request.driver)
            & (frame["dataset"] == dataset_key)
            & (frame["options"] == request.options_key)
            & frame["geohash"].isin(gh.ancestors(request.geohash))
        ]
        if request.time is None:
            frame = frame[(frame["time_start"] == "") & (frame["time_end"] == "")]
        else:
            # ISO dates compare correctly as strings.
            frame = frame[
                (frame["time_start"] != "")
                & (frame["time_start"] <= request.time.start.isoformat())
                & (frame["time_end"] >= request.time.end.isoformat())
            ]
        if frame.empty:
            return None

        candidates = []
        wanted = set(request.variables)
        for uri, rows in frame.groupby("uri", sort=False):
            if not wanted <= set(rows["variable"]):
                continue
            try:
                path = uri_to_path(str(uri))
            except ValueError:
                continue
            if not path.is_file():
                continue
            latest = rows.sort_values("timestamp_utc").iloc[-1]
            candidates.append(
                CacheHit(
                    uri=str(uri),
                    path=path,
                    geohash=str(latest["geohash"]),
                    request_id=str(latest["request_id"]),
                    timestamp_utc=str(latest["timestamp_utc"]),
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda hit: (len(hit.geohash), hit.timestamp_utc))
