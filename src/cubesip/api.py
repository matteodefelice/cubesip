"""Public entry points: ``check`` (inspect a request) and ``run`` (execute it).

Typical use::

    import cubesip
    req = cubesip.Request.from_json("examples/request_s2_ndvi.json")
    report = cubesip.check(req)   # dataset found? cached? how big?
    result = cubesip.run(req)     # NetCDF path + lazy xarray.Dataset

Implemented in milestone M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import xarray as xr

from cubesip.checker import CheckReport
from cubesip.config import Config
from cubesip.request import Request


@dataclass
class Result:
    path: Path  # new NetCDF, or the cached file when from_cache is True
    dataset: xr.Dataset  # lazy, already subset to the request
    from_cache: bool
    manifest_rows: list[dict[str, str]] = field(default_factory=list)  # empty on cache hit


def check(request: Request | dict, config: Config | str | Path | None = None) -> CheckReport:
    """Validate the request and run the checker without downloading anything."""
    raise NotImplementedError("M2")


def run(
    request: Request | dict,
    config: Config | str | Path | None = None,
    *,
    force: bool = False,
) -> Result:
    """Execute a request.

    Cache hit (unless ``force``): return a lazy view of the cached file, write nothing.
    Otherwise: driver.fetch -> NetCDF in target_dir -> append one manifest row per variable.
    """
    raise NotImplementedError("M2")
