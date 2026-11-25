"""Driver registry. Drivers are imported lazily so that e.g. openeo is only loaded when used.

driver           dataset                     data come from
---------------  --------------------------  ------------------------------------------------
openeo           openEO collection id        the back-end at openeo.url (default CDSE)
cdse_federation  openEO collection id        the CDSE openEO Federation (CDSE + Terrascope)
stac             STAC collection URL         load_stac on the back-end named by stac.backend
edh              Earth Data Hub dataset      Earth Data Hub Zarr, read with xarray
tessera          TESSERA store version       TESSERA embeddings Zarr, read with geotessera
beeldmateriaal   ortho_hr                    Dutch 8 cm aerial orthophoto GeoTIFF tiles (RGB + CIR)
local            Zarr/NetCDF/GeoTIFF path    the local file system, read with xarray
"""

from __future__ import annotations

from importlib import import_module

from cubesip.config import Config
from cubesip.drivers.base import (
    DatasetInfo,
    DatasetNotFoundError,
    Driver,
    EmptyResultError,
    SizeEstimate,
    XarraySourceDriver,
)

_REGISTRY = {
    "openeo": ("cubesip.drivers.openeo", "OpenEODriver"),
    "cdse_federation": ("cubesip.drivers.openeo", "CDSEFederationDriver"),
    "stac": ("cubesip.drivers.stac", "StacDriver"),
    "edh": ("cubesip.drivers.edh", "EarthDataHubDriver"),
    "tessera": ("cubesip.drivers.tessera", "TesseraDriver"),
    "beeldmateriaal": ("cubesip.drivers.beeldmateriaal", "BeeldmateriaalDriver"),
    "local": ("cubesip.drivers.local", "LocalDriver"),
}


def get_driver(name: str, config: Config) -> Driver:
    """Create the driver with the given name (e.g. 'openeo' or 'local'), loading its code only now.
    Raises ValueError for an unknown name.
    """
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown driver {name!r}; available: {sorted(_REGISTRY)}") from None
    return getattr(import_module(module_name), class_name)(config)


__all__ = [
    "DatasetInfo",
    "DatasetNotFoundError",
    "Driver",
    "EmptyResultError",
    "SizeEstimate",
    "XarraySourceDriver",
    "get_driver",
]
