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
    "backend",  # service the data came from (Driver.backend_url); empty for local files
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
    backend: str = ""  # empty: written before cubesip recorded back-ends


def path_to_uri(path: Path) -> str:
    """Turn a file path into a file:// address."""
    return Path(path).resolve().as_uri()


def uri_to_path(uri: str) -> Path:
    """Turn a file:// address back into a file path, network shares included. Other kinds of address
    are refused.
    """
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
    backend: str = "",
    request_id: str | None = None,
) -> list[dict[str, str]]:
    """Build the manifest rows for a file that was just written: one row per variable, each saying
    who downloaded what, when, from where, and where the file is.
    """
    from cubesip import __version__

    common = {
        "request_id": request_id or uuid.uuid4().hex,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "user": _safe(getpass.getuser),
        "hostname": _safe(socket.gethostname),
        "driver": request.driver,
        "dataset": dataset_key,
        "backend": backend,
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
    """Call a function and return its result as text, or an empty string if it fails (asking for the
    user name can fail on some service accounts).
    """
    try:
        return str(func())
    except Exception:  # getuser() can fail in some service accounts
        return ""


class Manifest:
    def __init__(self, path: Path) -> None:
        """Point at the manifest CSV file and at the lock file that sits next to it."""
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def read(self) -> pd.DataFrame:
        """Read every row of the manifest as text. If the file does not exist yet, return an empty
        table with the expected columns. Columns missing from older manifests are added, empty.
        """
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return pd.DataFrame({column: pd.Series(dtype=str) for column in COLUMNS})
        frame = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        for column in COLUMNS:  # tolerate manifests written by older versions
            if column not in frame:
                frame[column] = ""
        return frame

    def append(self, rows: list[dict[str, str]]) -> None:
        """Add rows at the end of the manifest. The lock file makes sure two users or machines never
        write at the same time. A new file gets its header line first, and a manifest from an older
        version is upgraded to the current columns before rows are added.
        """
        unknown = {key for row in rows for key in row} - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown manifest columns: {sorted(unknown)}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self.lock_path, timeout=LOCK_TIMEOUT_S):
            new_file = not self.path.is_file() or self.path.stat().st_size == 0
            if not new_file and self._header() != COLUMNS:
                self._upgrade()
            with self.path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="")
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)

    def _header(self) -> list[str]:
        """Read the column names from the first line of the manifest."""
        with self.path.open(newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle), [])

    def _upgrade(self) -> None:
        """Rewrite a manifest made by an older version of cubesip so that it has the current
        columns. Call it only while holding the lock. Without it, new rows would put values under
        the wrong column headers.
        """
        frame = self.read()  # adds missing columns as empty strings
        extra = [column for column in frame.columns if column not in COLUMNS]
        tmp = self.path.with_name(self.path.name + ".upgrade")
        frame[COLUMNS + extra].to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(self.path)

    def find_covering(
        self, request: Request, dataset_key: str, backend: str = ""
    ) -> CacheHit | None:
        """Find the best file already downloaded that holds everything the request asks for, or None
        if there is none.

        A file covers the request when its rows have the same driver, dataset, back-end and
        options, its geohash is the request's or an ancestor, its period contains the
        requested days (or both are static), it holds every requested variable and it still
        exists. Rows without a back-end (written before cubesip recorded it) match any
        back-end. The longest geohash wins, then a recorded back-end, then the most recent.
        """
        frame = self.read()
        if frame.empty:
            return None
        frame = frame[
            (frame["driver"] == request.driver)
            & (frame["dataset"] == dataset_key)
            & ((frame["backend"] == backend) | (frame["backend"] == ""))
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
                    backend=str(latest["backend"]),
                )
            )
        if not candidates:
            return None
        return max(
            candidates, key=lambda hit: (len(hit.geohash), bool(hit.backend), hit.timestamp_utc)
        )
