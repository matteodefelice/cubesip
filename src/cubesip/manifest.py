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
from typing import Any
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


def _uri_exists(uri: str) -> bool:
    """True when the uri still points at a file on disk; False for a deleted or moved file and for
    addresses that are not file:// paths.
    """
    try:
        return uri_to_path(uri).is_file()
    except ValueError:
        return False


_FILE_VIEW_DTYPES = {
    "uri": str,
    "driver": str,
    "dataset": str,
    "backend": str,
    "geohash": str,
    "time_start": str,
    "time_end": str,
    "variables": str,
    "rows": "int64",
    "file_size_bytes": "int64",
    "timestamp_utc": str,
    "exists": bool,
}


def _empty_frame(dtypes: dict[str, str | type]) -> pd.DataFrame:
    """An empty table with the given columns and kinds: text, whole numbers or True/False."""
    return pd.DataFrame(columns=list(dtypes)).astype(dtypes)


def _file_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the manifest to one row per file (uri): the newest metadata of that file, its
    variables joined together, how many rows it spans, and whether it is still on disk.
    """
    if frame.empty:
        return _empty_frame(_FILE_VIEW_DTYPES)
    sizes = pd.to_numeric(frame["file_size_bytes"], errors="coerce").fillna(0).astype("int64")
    ordered = frame.assign(file_size_bytes=sizes, exists=frame["uri"].map(_uri_exists)).sort_values(
        "timestamp_utc", kind="stable"
    )
    return (
        ordered.groupby("uri", sort=False)
        .agg(
            driver=("driver", "last"),
            dataset=("dataset", "last"),
            backend=("backend", "last"),
            geohash=("geohash", "last"),
            time_start=("time_start", "min"),
            time_end=("time_end", "max"),
            variables=("variable", lambda values: ", ".join(sorted(set(values)))),
            rows=("variable", "size"),
            file_size_bytes=("file_size_bytes", "last"),
            timestamp_utc=("timestamp_utc", "last"),
            exists=("exists", "first"),
        )
        .reset_index()
    )


_DATASET_VIEW_DTYPES = {
    "driver": str,
    "dataset": str,
    "backend": str,
    "files": "int64",
    "missing_files": "int64",
    "rows": "int64",
    "variables": str,
    "geohashes": "int64",
    "size_bytes": "int64",
    "time_start": str,
    "time_end": str,
}


def _by_dataset(frame: pd.DataFrame, view: pd.DataFrame) -> pd.DataFrame:
    """One row per driver / dataset / back-end: how many files and rows, which variables and
    cells, how much space, and the period covered.
    """
    if view.empty:
        return _empty_frame(_DATASET_VIEW_DTYPES)
    rows = []
    for (driver, dataset, backend), files in view.groupby(
        ["driver", "dataset", "backend"], sort=False
    ):
        file_rows = frame[frame["uri"].isin(files["uri"])]
        rows.append(
            {
                "driver": driver,
                "dataset": dataset,
                "backend": backend,
                "files": len(files),
                "missing_files": int((~files["exists"]).sum()),
                "rows": len(file_rows),
                "variables": ", ".join(sorted(set(file_rows["variable"]))),
                "geohashes": file_rows["geohash"].nunique(),
                "size_bytes": int(files["file_size_bytes"].sum()),
                "time_start": min((t for t in files["time_start"] if t), default=""),
                "time_end": max((t for t in files["time_end"] if t), default=""),
            }
        )
    return pd.DataFrame(rows, columns=list(_DATASET_VIEW_DTYPES))


def _missing(view: pd.DataFrame) -> pd.DataFrame:
    """The rows of the per-file view whose file is gone, without the ``exists`` column."""
    return view[~view["exists"]].drop(columns="exists").reset_index(drop=True)


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

    def _rewrite(self, frame: pd.DataFrame) -> None:
        """Replace the manifest file with the given rows: the columns cubesip knows first, then
        any columns another version may have added. The rows go to a temporary file that is then
        renamed over the manifest, so a crash halfway cannot leave a half-written file behind.
        Call it only while holding the lock.
        """
        extra = [column for column in frame.columns if column not in COLUMNS]
        tmp = self.path.with_name(self.path.name + ".tmp")
        frame[COLUMNS + extra].to_csv(tmp, index=False, encoding="utf-8")
        tmp.replace(self.path)

    def _upgrade(self) -> None:
        """Rewrite a manifest made by an older version of cubesip so that it has the current
        columns. Call it only while holding the lock. Without it, new rows would put values under
        the wrong column headers.
        """
        self._rewrite(self.read())  # read() adds missing columns as empty strings

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

    def summarize(self) -> dict[str, Any]:
        """Summarise what the manifest records, as plain data for programmatic use: how many
        rows, files and requests, the total size (and the size of files that are gone), which
        drivers, datasets, variables, cells, users and hosts appear, the period covered, and a
        ``by_dataset`` DataFrame with one row per driver / dataset / back-end. Works on an empty
        or missing manifest (all counts zero, empty lists and an empty ``by_dataset``).
        """
        frame = self.read()
        view = _file_view(frame)
        missing = view[~view["exists"]]
        return {
            "manifest": str(self.path),
            "rows": len(frame),
            "files": len(view),
            "missing_files": len(missing),
            "requests": frame["request_id"].nunique(),
            "datasets": frame.groupby(["driver", "dataset"]).ngroups,
            "variables": frame["variable"].nunique(),
            "geohashes": frame["geohash"].nunique(),
            "drivers": sorted({value for value in frame["driver"] if value}),
            "users": sorted({value for value in frame["user"] if value}),
            "hosts": sorted({value for value in frame["hostname"] if value}),
            "total_size_bytes": int(view["file_size_bytes"].sum()),
            "missing_size_bytes": int(missing["file_size_bytes"].sum()),
            "first_entry": str(view["timestamp_utc"].min()) if len(view) else "",
            "last_entry": str(view["timestamp_utc"].max()) if len(view) else "",
            "time_start": min((value for value in view["time_start"] if value), default=""),
            "time_end": max((value for value in view["time_end"] if value), default=""),
            "by_dataset": _by_dataset(frame, view),
        }

    def find_missing(self) -> pd.DataFrame:
        """Report the files the manifest records that can no longer be opened, one row per file
        (newest metadata, variables joined together, row count, recorded size), in manifest
        order. A file counts as missing when its path no longer exists or its address is not a
        file:// URI; an empty or missing manifest gives an empty report.
        """
        return _missing(_file_view(self.read()))

    def prune(self) -> pd.DataFrame:
        """Remove from the manifest the rows of files that are no longer on disk (the ones
        :meth:`find_missing` reports) and return that report of what was removed. The file is
        rewritten under the lock, atomically: written to a temporary file first, then renamed.
        Every remaining row is kept as it was, extra columns written by another version of
        cubesip included. Nothing is written when there is nothing to remove.
        """
        if not self.path.is_file():
            return _missing(_file_view(self.read()))
        with FileLock(self.lock_path, timeout=LOCK_TIMEOUT_S):
            frame = self.read()
            report = _missing(_file_view(frame))
            if report.empty:
                return report
            keep = frame[~frame["uri"].isin(report["uri"])]
            self._rewrite(keep)
            return report
