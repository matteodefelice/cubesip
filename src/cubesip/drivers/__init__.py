"""Driver registry. Drivers are imported lazily so that e.g. openeo is only loaded when used."""

from __future__ import annotations

from importlib import import_module

from cubesip.config import Config
from cubesip.drivers.base import DatasetInfo, DatasetNotFoundError, Driver, SizeEstimate

_REGISTRY = {
    "openeo": ("cubesip.drivers.openeo", "OpenEODriver"),
    "edh": ("cubesip.drivers.edh", "EarthDataHubDriver"),
    "local": ("cubesip.drivers.local", "LocalDriver"),
}


def get_driver(name: str, config: Config) -> Driver:
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown driver {name!r}; available: {sorted(_REGISTRY)}") from None
    return getattr(import_module(module_name), class_name)(config)


__all__ = ["DatasetInfo", "DatasetNotFoundError", "Driver", "SizeEstimate", "get_driver"]
