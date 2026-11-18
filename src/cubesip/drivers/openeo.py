"""openEO driver: back-end collections (load_collection) and arbitrary STAC (load_stac).

- dataset starting with http(s):// -> ``connection.load_stac(url, ...)``
- otherwise                        -> ``connection.load_collection(collection_id, ...)``
- spatial_extent: geohash box as {"west", "south", "east", "north", "crs": 4326}
- temporal_extent: [start, end + 1 day] (openEO end bound is exclusive)
- options: ``max_cloud_cover`` (collections only) and ``properties`` passed through
- synchronous ``cube.download(out_path, format="netCDF")`` only (batch jobs: future work)

Implemented in milestone M5.
"""

from __future__ import annotations

from pathlib import Path

from cubesip.drivers.base import DatasetInfo, Driver, SizeEstimate
from cubesip.request import Request


class OpenEODriver(Driver):
    name = "openeo"

    def connect(self):  # -> openeo.Connection
        """Connect to config.openeo.url and authenticate with the configured method."""
        raise NotImplementedError("M5")

    def describe(self, request: Request) -> DatasetInfo:
        raise NotImplementedError("M5")

    def estimate_size(self, request: Request) -> SizeEstimate:
        raise NotImplementedError("M5")

    def fetch(self, request: Request, out_path: Path) -> Path:
        raise NotImplementedError("M5")
