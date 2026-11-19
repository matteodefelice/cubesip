"""Output files: naming, NetCDF writing and reading."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from cubesip.request import Request, canonical_json

_KEEP_ENCODING = ("grid_mapping", "coordinates")


def dataset_slug(dataset_key: str) -> str:
    """Short file-name friendly label for a dataset (last path segment, no extension)."""
    segment = [part for part in re.split(r"[/\\]", dataset_key) if part][-1:] or ["dataset"]
    stem = re.sub(r"\.(zarr|nc|nc4|netcdf|tif|tiff)$", "", segment[0], flags=re.IGNORECASE)
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", stem).strip("-")
    return slug[:40] or "dataset"


def output_path(request: Request, dataset_key: str, backend: str = "") -> Path:
    """``{target_dir}/{driver}_{slug}_{geohash}_{start}-{end}_{hash8}.nc``.

    ``hash8`` covers dataset, variables, options and back-end, so the same request sent to
    two back-ends gives two files instead of one overwriting the other.
    """
    if request.time is None:
        period = "static"
    else:
        period = f"{request.time.start:%Y%m%d}-{request.time.end:%Y%m%d}"
    identity = {
        "dataset": dataset_key,
        "variables": sorted(request.variables),
        "options": request.options,
    }
    if backend:  # left out when empty so local file names stay as they were
        identity["backend"] = backend
    digest = hashlib.sha1(canonical_json(identity).encode()).hexdigest()[:8]
    name = f"{request.driver}_{dataset_slug(dataset_key)}_{request.geohash}_{period}_{digest}.nc"
    return Path(request.target_dir).expanduser().resolve() / name


def provenance_attrs(request: Request, dataset_key: str, backend: str = "") -> dict[str, str]:
    from cubesip import __version__

    attrs = {
        "cubesip_version": __version__,
        "cubesip_driver": request.driver,
        "cubesip_dataset": dataset_key,
        "cubesip_geohash": request.geohash,
        "cubesip_options": request.options_key,
    }
    if backend:
        attrs["cubesip_backend"] = backend
    if request.time is not None:
        attrs["cubesip_time_start"] = request.time.start.isoformat()
        attrs["cubesip_time_end"] = request.time.end.isoformat()
    return attrs


def write_netcdf(ds: xr.Dataset, path: Path, attrs: dict[str, str] | None = None) -> Path:
    """Write a (lazy) dataset to NetCDF4 with compression.

    Source encodings (zarr compressors, chunking, scale factors) are dropped so any input
    store can be written; attributes netCDF cannot hold are converted to strings.
    """
    out = ds.copy(deep=False)
    for variable in out.variables.values():
        variable.encoding = {k: v for k, v in variable.encoding.items() if k in _KEEP_ENCODING}
        variable.attrs = _sanitize_attrs(variable.attrs)
    out.attrs = _sanitize_attrs({**ds.attrs, **(attrs or {})})
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    history = f"{stamp} written by cubesip"
    previous = out.attrs.get("history")
    out.attrs["history"] = f"{previous}\n{history}" if previous else history

    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, var in out.data_vars.items()
        if var.ndim > 0 and var.dtype.kind in "biuf"
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def open_netcdf(path: Path) -> xr.Dataset:
    """Open a cubesip NetCDF lazily, with grid-mapping variables decoded as coordinates."""
    return xr.open_dataset(path, engine="netcdf4", chunks={}, decode_coords="all")


def _sanitize_attrs(attrs: dict) -> dict:
    clean = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, bool | np.bool_):
            clean[key] = int(value)
        elif isinstance(value, str | int | float | np.number | np.ndarray):
            clean[key] = value
        elif (
            isinstance(value, list | tuple)
            and value
            and all(
                isinstance(v, int | float | np.number) and not isinstance(v, bool) for v in value
            )
        ):
            clean[key] = list(value)
        else:
            clean[key] = json.dumps(value, default=str)
    return clean
