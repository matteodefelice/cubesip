"""Public entry points: ``check`` (inspect a request) and ``run`` (execute it).

Typical use::

    import cubesip
    req = cubesip.Request.from_json("examples/request_s2_ndvi.json")
    print(cubesip.check(req))     # cached? dataset found? how big?
    result = cubesip.run(req)     # NetCDF path + lazy xarray.Dataset
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import xarray as xr

from cubesip import checker, spatial
from cubesip.config import Config, load_config
from cubesip.drivers import get_driver
from cubesip.manifest import Manifest, make_rows
from cubesip.output import open_netcdf, output_path
from cubesip.request import Request

log = logging.getLogger(__name__)

RequestLike = Request | dict[str, Any] | str | Path
ConfigLike = Config | str | Path | None


@dataclass
class Result:
    path: Path  # new NetCDF, or the cached file when from_cache is True
    dataset: xr.Dataset  # lazy, already subset to the request
    from_cache: bool
    manifest_rows: list[dict[str, str]] = field(default_factory=list)  # empty on cache hit


def check(request: RequestLike, config: ConfigLike = None) -> checker.CheckReport:
    """Look at a request without downloading anything: is it already cached, does the dataset exist,
    and roughly how big would the download be. Returns a report you can print.
    """
    return checker.check(_as_request(request), _as_config(config))


def run(request: RequestLike, config: ConfigLike = None, *, force: bool = False) -> Result:
    """Get the data for a request. If a file already on disk covers it (and ``force`` is False),
    open that file and return the part that was asked for, without writing anything. Otherwise
    download the data with the right driver, save them as a NetCDF file in the target folder, and
    record the file in the manifest so it can be reused next time.
    """
    req = _as_request(request)
    cfg = _as_config(config)
    driver = get_driver(req.driver, cfg)
    driver.validate(req)
    key = driver.dataset_key(req.dataset)
    backend = driver.backend_url
    manifest = Manifest(cfg.manifest)

    if not force:
        hit = manifest.find_covering(req, key, backend)
        if hit is not None:
            log.info("serving %s from cache %s", req.geohash, hit.path)
            view = spatial.subset(open_netcdf(hit.path), req.bbox, req.time, req.variables)
            return Result(path=hit.path, dataset=view, from_cache=True)

    info = driver.describe(req)
    if info.missing_variables:
        raise KeyError(
            f"variables not available in {key}: {info.missing_variables}; "
            f"available: {info.available_variables}"
        )

    out_path = output_path(req, key, backend)
    partial = out_path.with_name(out_path.stem + ".partial.nc")
    log.info("fetching %s %s for %s -> %s", req.driver, key, req.geohash, out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        driver.fetch(req, partial)
        try:
            os.replace(partial, out_path)
        except PermissionError as exc:
            raise PermissionError(
                f"cannot replace {out_path}: it is open (e.g. an earlier Result.dataset); "
                "close it with result.dataset.close() and retry"
            ) from exc
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    dataset = open_netcdf(out_path)
    try:
        crs = spatial.native_crs(dataset)
    except ValueError:
        crs = ""
    rows = make_rows(req, dataset_key=key, path=out_path, crs=crs, backend=backend)
    manifest.append(rows)
    return Result(path=out_path, dataset=dataset, from_cache=False, manifest_rows=rows)


def _as_request(request: RequestLike) -> Request:
    """Turn what the user passed (a Request, a dict, a JSON string or the path to a JSON file) into
    a Request.
    """
    return request if isinstance(request, Request) else Request.from_json(request)


def _as_config(config: ConfigLike) -> Config:
    """Turn what the user passed (a Config, a path, or nothing) into a loaded Config. With nothing,
    the config file is looked up in the usual places.
    """
    return config if isinstance(config, Config) else load_config(config)
