"""URL driver: one raster file or Zarr store on a web server, read over HTTP(S) without
downloading it whole.

It is meant for gridded layers published as plain files, such as many of the global hazard
maps listed in the Climate Risk STAC (https://climate-risk-data.github.io/climate-risk-stac/):
JRC flood depth per return period, STORM cyclone wind speeds and similar. It works like the
``local`` driver, but the file is on a server.

- dataset: an ``http(s)://`` URL of

  - a GeoTIFF / COG, read with GDAL ``/vsicurl/``. Each band becomes a variable, named from
    the band descriptions when present, else ``band_1``, ``band_2``... (as in ``local``);
  - a GeoTIFF inside a zip archive, written ``<archive URL>!<path inside the archive>``, read
    with GDAL ``/vsizip/`` without downloading the archive;
  - a NetCDF file, read with netCDF-C's byte-range mode (``#mode=bytes``);
  - a Zarr store (``.zarr``).

  The format comes from the URL's extension. A URL without one (common on data repositories,
  e.g. ``data.4tu.nl/file/<id>/<id>``) is recognised from its first bytes.
- variables: band names (GeoTIFF) or variable names (NetCDF, Zarr).
- time: ``null`` for layers without a time dimension, as most hazard maps are.
- options: none. auth: none, public files only.

Everything except Zarr needs a server that answers HTTP range requests (most do; GDAL and
netCDF-C fail otherwise). A compressed zip member is decompressed from its start, so the
first read of a large member can take a few seconds; a ``.tif`` outside an archive does not
have that cost.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

import xarray as xr

from cubesip.drivers.base import DatasetNotFoundError, XarraySourceDriver
from cubesip.request import Request

Format = Literal["geotiff", "netcdf", "zarr"]

MEMBER_SEPARATOR = "!"
_SUFFIXES: dict[str, Format] = {
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".netcdf": "netcdf",
    ".zarr": "zarr",
}
_MAGIC: tuple[tuple[bytes, Format | Literal["zip"]], ...] = (
    (b"II*\x00", "geotiff"),
    (b"MM\x00*", "geotiff"),
    (b"II+\x00", "geotiff"),  # BigTIFF
    (b"MM\x00+", "geotiff"),
    (b"\x89HDF", "netcdf"),  # NetCDF-4
    (b"CDF", "netcdf"),  # NetCDF classic
    (b"PK", "zip"),
)


def split_member(dataset_key: str) -> tuple[str, str | None]:
    """Split ``<archive URL>!<member>`` into the archive URL and the member (None when there is
    no member).
    """
    url, sep, member = dataset_key.partition(MEMBER_SEPARATOR)
    return url, (member if sep else None)


def _suffix_format(url: str) -> Format | None:
    """The format given by the URL's extension (ignoring any query string), or None."""
    path = urlparse(url).path.rstrip("/").lower()
    for suffix, fmt in _SUFFIXES.items():
        if path.endswith(suffix):
            return fmt
    return None


def sniff_format(url: str, timeout: float = 60) -> Format:
    """Recognise a file without an extension from its first bytes (one small range request).
    Raises DatasetNotFoundError for a missing file, PermissionError when access is refused and
    ValueError for a zip archive (a member must be named) or an unknown format.
    """
    import requests

    try:
        with requests.get(
            url, headers={"Range": "bytes=0-7"}, stream=True, timeout=timeout
        ) as response:
            if response.status_code == 404:
                raise DatasetNotFoundError(f"url dataset not found (HTTP 404): {url}")
            if response.status_code in (401, 403):
                raise PermissionError(
                    f"access refused (HTTP {response.status_code}) to {url}: "
                    "the url driver reads public files only"
                )
            response.raise_for_status()
            head = response.raw.read(8)
    except requests.RequestException as exc:
        raise DatasetNotFoundError(f"cannot reach {url}: {exc}") from exc
    for magic, fmt in _MAGIC:
        if head.startswith(magic):
            if fmt == "zip":
                raise ValueError(
                    f"{url} is a zip archive: name the GeoTIFF inside it as "
                    f"'<archive URL>{MEMBER_SEPARATOR}<path inside the archive>'"
                )
            return fmt
    raise ValueError(
        f"cannot tell the format of {url} (no known extension, first bytes {head!r}); "
        "the url driver reads GeoTIFF, NetCDF and Zarr"
    )


def gdal_path(url: str, member: str | None = None) -> str:
    """The GDAL virtual path of a remote file, or of a file inside a remote zip archive. The
    archive is put in braces so that URLs without a ``.zip`` extension work too.
    """
    if member is None:
        return f"/vsicurl/{url}"
    return f"/vsizip/{{/vsicurl/{url}}}/{member}"


class UrlDriver(XarraySourceDriver):
    name = "url"

    def dataset_key(self, dataset: str) -> str:
        """Give the URL in a standard form: no spaces, no trailing '/', and the member (if any)
        without leading or trailing '/'.
        """
        url, member = split_member(dataset.strip())
        url = url.strip().rstrip("/")
        if member is None:
            return url
        return f"{url}{MEMBER_SEPARATOR}{member.strip().strip('/')}"

    def validate(self, request: Request) -> None:
        """Refuse a dataset that is not an http(s) URL, or an archive member that is not a GeoTIFF,
        then do the usual checks.
        """
        url, member = split_member(self.dataset_key(request.dataset))
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError(
                f"the url driver takes an http(s) URL, got {request.dataset!r}; "
                "use the 'local' driver for a file on disk"
            )
        if member is not None and _suffix_format(member) != "geotiff":
            raise ValueError(
                f"only GeoTIFF files (.tif, .tiff) can be read inside a zip archive, got {member!r}"
            )
        super().validate(request)

    def _open(self, dataset_key: str) -> xr.Dataset:
        """Open the remote file or store without downloading its data. A missing file becomes a
        DatasetNotFoundError.
        """
        url, member = split_member(dataset_key)
        if member is not None:
            return _open_geotiff(gdal_path(url, member), dataset_key)
        fmt = _suffix_format(url) or sniff_format(url)
        if fmt == "geotiff":
            return _open_geotiff(gdal_path(url), dataset_key)
        if fmt == "netcdf":
            try:
                return xr.open_dataset(
                    f"{url}#mode=bytes", engine="netcdf4", chunks={}, decode_coords="all"
                )
            except OSError as exc:
                raise DatasetNotFoundError(f"cannot open NetCDF at {url}: {exc}") from exc
        try:
            return xr.open_dataset(url, engine="zarr", chunks={}, decode_coords="all")
        except (FileNotFoundError, KeyError) as exc:
            raise DatasetNotFoundError(f"Zarr store not found: {url}") from exc


def _open_geotiff(path: str, dataset_key: str) -> xr.Dataset:
    """Open a remote GeoTIFF as the local driver does, turning GDAL's "cannot open" into a
    DatasetNotFoundError.
    """
    from rasterio.errors import RasterioIOError

    from cubesip.drivers.local import open_geotiff

    try:
        return open_geotiff(path)
    except RasterioIOError as exc:
        raise DatasetNotFoundError(f"cannot open GeoTIFF at {dataset_key}: {exc}") from exc
