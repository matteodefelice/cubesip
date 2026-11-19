"""Driver interface shared by every driver (openeo, cdse_federation, stac, edh, local)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import xarray as xr

from cubesip import spatial
from cubesip.config import Config
from cubesip.output import provenance_attrs, write_netcdf
from cubesip.request import Request


class DatasetNotFoundError(LookupError):
    """The dataset does not exist on the source (or is not accessible with these credentials)."""


class EmptyResultError(LookupError):
    """The source answered without error but returned no data (nothing is written)."""


@dataclass
class DatasetInfo:
    """What the source says about a dataset, as far as the request is concerned."""

    dataset: str
    available_variables: list[str] | None  # None when the source cannot list them
    missing_variables: list[str] = field(default_factory=list)
    crs: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class SizeEstimate:
    nbytes: int | None
    method: Literal["exact", "estimate", "unknown"]
    detail: str = ""


class Driver(ABC):
    name: ClassVar[str]
    supported_options: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, config: Config) -> None:
        """Keep the config, so the driver can read its settings and credentials."""
        self.config = config

    def dataset_key(self, dataset: str) -> str:
        """Give the dataset name in a standard form, used in the manifest and to find cached files.
        By default this only trims spaces.
        """
        return dataset.strip()

    @property
    def backend_url(self) -> str:
        """Address of the service the data come from (empty for local files). It is saved in the
        manifest, so a file downloaded from one service is never reused for a request to another.
        """
        return ""

    def validate(self, request: Request) -> None:
        """Refuse a request this driver cannot handle, for example one with options it does not
        know.
        """
        unknown = sorted(set(request.options) - self.supported_options)
        if unknown:
            allowed = sorted(self.supported_options) or "none"
            raise ValueError(
                f"options not supported by the {self.name} driver: {unknown} (supported: {allowed})"
            )

    @property
    def max_request_bytes(self) -> int | None:
        """Download size above which the check warns (None: no limit)."""
        return None

    @abstractmethod
    def describe(self, request: Request) -> DatasetInfo:
        """Check that the dataset exists and which of the requested variables it has. Raises
        DatasetNotFoundError if the dataset does not exist.
        """

    @abstractmethod
    def estimate_size(self, request: Request) -> SizeEstimate:
        """Estimate how many bytes the requested data take once unpacked (exact when the source
        allows it).
        """

    @abstractmethod
    def fetch(self, request: Request, out_path: Path) -> Path:
        """Download the data and save them as a NetCDF file at ``out_path``, keeping the source's
        own CRS.
        """


class XarraySourceDriver(Driver):
    """Base for sources that xarray can open lazily (local files, Earth Data Hub Zarr).

    Subclasses only implement :meth:`_open`; selection, sizing and writing are shared.
    """

    def __init__(self, config: Config) -> None:
        """Set up the driver with an empty memory of datasets already opened."""
        super().__init__(config)
        self._opened: dict[str, xr.Dataset] = {}

    @abstractmethod
    def _open(self, dataset_key: str) -> xr.Dataset:
        """Open the source without loading its data; raise DatasetNotFoundError if it does not
        exist.
        """

    def open_source(self, request: Request) -> xr.Dataset:
        """Open the request's dataset, reusing it if this driver has already opened it."""
        key = self.dataset_key(request.dataset)
        if key not in self._opened:
            self._opened[key] = self._open(key)
        return self._opened[key]

    def describe(self, request: Request) -> DatasetInfo:
        """Open the dataset and report which variables it has, which requested ones are missing, its
        CRS, and notes such as a time dimension that the request does not limit.
        """
        ds = self.open_source(request)
        available = spatial.data_variables(ds)
        info = DatasetInfo(
            dataset=self.dataset_key(request.dataset),
            available_variables=available,
            missing_variables=[v for v in request.variables if v not in available],
        )
        try:
            info.crs = spatial.native_crs(ds)
        except ValueError as exc:
            info.notes.append(f"CRS unknown: {exc}")
        try:
            time_dim = spatial.find_dims(ds).time
        except ValueError as exc:
            info.notes.append(str(exc))
        else:
            if request.time is None and time_dim is not None:
                info.notes.append(
                    f"dataset has a time dimension ({ds.sizes[time_dim]} steps) and the request "
                    "has no time range: every time step will be retrieved"
                )
        return info

    def select(self, request: Request) -> xr.Dataset:
        """Cut the dataset down to the request's variables, area and days (data not loaded yet)."""
        return spatial.subset(
            self.open_source(request), request.bbox, request.time, request.variables
        )

    def estimate_size(self, request: Request) -> SizeEstimate:
        """Work out the exact size of the selected data and its length along each dimension."""
        selected = self.select(request)
        shape = ", ".join(f"{dim}={size}" for dim, size in selected.sizes.items())
        return SizeEstimate(nbytes=int(selected.nbytes), method="exact", detail=shape)

    def fetch(self, request: Request, out_path: Path) -> Path:
        """Write the selected data to a NetCDF file, with metadata saying where they came from."""
        attrs = provenance_attrs(request, self.dataset_key(request.dataset), self.backend_url)
        return write_netcdf(self.select(request), out_path, attrs)
