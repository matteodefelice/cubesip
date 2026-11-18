"""Local driver: Zarr stores, NetCDF files and GeoTIFF/COG on the local filesystem.

- directory (``*.zarr``)  -> ``xr.open_zarr``
- ``*.tif`` / ``*.tiff``  -> ``rioxarray.open_rasterio``; each band becomes a variable, named
  from the band descriptions when present, else ``band_1``, ``band_2``...
- anything else           -> ``xr.open_dataset`` (NetCDF, HDF5...)

Static datasets (no time dimension) need ``time: null`` in the request.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from cubesip.drivers.base import DatasetNotFoundError, XarraySourceDriver


class LocalDriver(XarraySourceDriver):
    name = "local"

    def dataset_key(self, dataset: str) -> str:
        return Path(dataset.strip()).expanduser().resolve().as_posix()

    def _open(self, dataset_key: str) -> xr.Dataset:
        path = Path(dataset_key)
        if not path.exists():
            raise DatasetNotFoundError(f"local dataset not found: {path}")
        if path.is_dir():
            return xr.open_zarr(path, decode_coords="all")
        if path.suffix.lower() in (".tif", ".tiff"):
            return open_geotiff(path)
        return xr.open_dataset(path, chunks={}, decode_coords="all")


def open_geotiff(path: Path) -> xr.Dataset:
    """Open a (multi-band) GeoTIFF as a Dataset with one variable per band."""
    import rioxarray

    da = rioxarray.open_rasterio(path, chunks=True, mask_and_scale=True)
    if not isinstance(da, xr.DataArray):  # multiple subdatasets
        raise ValueError(f"{path}: GeoTIFFs with subdatasets are not supported")
    descriptions = da.attrs.pop("long_name", None)
    if isinstance(descriptions, str):
        descriptions = (descriptions,)
    if (
        descriptions
        and len(descriptions) == da.sizes["band"]
        and all(descriptions)
        and len(set(descriptions)) == len(descriptions)
    ):
        labels = [str(d) for d in descriptions]
    else:
        labels = [f"band_{int(b)}" for b in da["band"].values]
    return da.assign_coords(band=labels).to_dataset(dim="band")
