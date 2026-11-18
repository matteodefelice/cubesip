"""Local driver: Zarr stores, NetCDF files and GeoTIFF/COG on the local filesystem.

- ``*.zarr`` directory -> ``xr.open_zarr``; ``*.nc`` -> ``xr.open_dataset``;
  ``*.tif``/``*.tiff`` -> ``rioxarray.open_rasterio`` (band dimension -> variables)
- static datasets (no time dimension) require ``time: null`` in the request
- subsetting via :func:`cubesip.spatial.subset`; size estimate is exact

Implemented in milestone M3.
"""

from __future__ import annotations

from pathlib import Path

from cubesip.drivers.base import DatasetInfo, Driver, SizeEstimate
from cubesip.request import Request


class LocalDriver(Driver):
    name = "local"

    def describe(self, request: Request) -> DatasetInfo:
        raise NotImplementedError("M3")

    def estimate_size(self, request: Request) -> SizeEstimate:
        raise NotImplementedError("M3")

    def fetch(self, request: Request, out_path: Path) -> Path:
        raise NotImplementedError("M3")
