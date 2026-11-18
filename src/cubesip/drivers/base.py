"""Driver interface shared by openeo / edh / local."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from cubesip.config import Config
from cubesip.request import Request


class DatasetNotFoundError(LookupError):
    """The dataset does not exist on the source (or is not accessible with these credentials)."""


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

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def describe(self, request: Request) -> DatasetInfo:
        """Check the dataset exists and which requested variables are available.

        Raises DatasetNotFoundError if the dataset itself does not exist.
        """

    @abstractmethod
    def estimate_size(self, request: Request) -> SizeEstimate:
        """Uncompressed size of the requested cube (exact when the source allows it)."""

    @abstractmethod
    def fetch(self, request: Request, out_path: Path) -> Path:
        """Retrieve the data and write a NetCDF at ``out_path`` (native CRS kept)."""
