"""Earth Data Hub driver: stream Zarr stores over HTTPS with xarray + zarr.

- dataset: path relative to config.earthdatahub.url (``era5/era5-single-levels-atmosphere-v0``,
  ``.zarr`` optional) or a full https URL
- auth: user ``edh`` + personal token; preferably passed as basic auth through fsspec
  ``storage_options`` so the token never appears in URLs, reprs or tracebacks
- open with ``xr.open_dataset(..., engine="zarr", chunks={})`` and subset lazily with
  :func:`cubesip.spatial.subset` before anything is downloaded
- size estimate is exact: ``subset.nbytes``

Implemented in milestone M4.
"""

from __future__ import annotations

from pathlib import Path

from cubesip.drivers.base import DatasetInfo, Driver, SizeEstimate
from cubesip.request import Request


class EarthDataHubDriver(Driver):
    name = "edh"

    def describe(self, request: Request) -> DatasetInfo:
        raise NotImplementedError("M4")

    def estimate_size(self, request: Request) -> SizeEstimate:
        raise NotImplementedError("M4")

    def fetch(self, request: Request, out_path: Path) -> Path:
        raise NotImplementedError("M4")
