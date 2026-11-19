"""The user-facing request: what to fetch, where, when, and where to put it.

A request is provided as JSON (file, string, or dict) and validated into a :class:`Request`.
The JSON schema in ``schemas/request.schema.json`` is generated from this model
(see :func:`request_json_schema`).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cubesip import geohash as gh

DriverName = Literal["openeo", "cdse_federation", "stac", "edh", "local"]


class TimeRange(BaseModel):
    """Closed date interval: both ``start`` and ``end`` days are included."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date = Field(description="First day (inclusive), ISO format YYYY-MM-DD.")
    end: date = Field(description="Last day (inclusive), ISO format YYYY-MM-DD.")

    @model_validator(mode="after")
    def _check_order(self) -> TimeRange:
        if self.end < self.start:
            raise ValueError(f"time.end ({self.end}) is before time.start ({self.start})")
        return self

    @property
    def end_exclusive(self) -> date:
        """The day after ``end``, for APIs whose upper bound is exclusive (openEO, slicing)."""
        return self.end + timedelta(days=1)

    def covers(self, other: TimeRange) -> bool:
        return self.start <= other.start and self.end >= other.end


class Request(BaseModel):
    """A data request for one dataset, one geohash cell and an optional time range."""

    model_config = ConfigDict(extra="forbid")

    driver: DriverName = Field(
        description="Data source: 'openeo' (collections of the openEO back-end in the "
        "config, default CDSE), 'cdse_federation' (collections of the CDSE openEO Federation: "
        "CDSE + Terrascope), 'stac' (any STAC collection, loaded with openEO load_stac), "
        "'edh' (Earth Data Hub Zarr), 'local' (Zarr/NetCDF/GeoTIFF on disk)."
    )
    dataset: str = Field(
        min_length=1,
        description="openeo / cdse_federation: collection id (e.g. SENTINEL2_L2A); "
        "stac: STAC collection URL; "
        "edh: dataset path (e.g. era5/era5-single-levels-atmosphere-v0) or full URL; "
        "local: path to the store/file.",
    )
    variables: list[str] = Field(
        min_length=1, description="Bands / variables to retrieve, e.g. ['B04', 'B08'] or ['t2m']."
    )
    geohash: str = Field(description="Area of interest as a geohash cell, e.g. 'qem7'.")
    time: TimeRange | None = Field(
        default=None, description="Days to retrieve (inclusive); null for static datasets."
    )
    target_dir: Path = Field(description="Local directory where the NetCDF output is written.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Driver-specific extras, e.g. {'max_cloud_cover': 60} for openeo, "
        "cdse_federation and stac.",
    )

    @field_validator("geohash")
    @classmethod
    def _valid_geohash(cls, value: str) -> str:
        return gh.validate(value)

    @field_validator("variables")
    @classmethod
    def _unique_variables(cls, value: list[str]) -> list[str]:
        dupes = sorted({v for v in value if value.count(v) > 1})
        if dupes:
            raise ValueError(f"duplicate variables: {dupes}")
        return value

    @property
    def bbox(self) -> gh.BBox:
        """Geohash cell bounds in EPSG:4326."""
        return gh.bbox(self.geohash)

    @property
    def options_key(self) -> str:
        """Canonical JSON of ``options`` (sorted keys), used for cache matching."""
        return canonical_json(self.options)

    @classmethod
    def from_json(cls, source: str | Path | dict[str, Any]) -> Request:
        """Build a request from a dict, a JSON string, or a path to a JSON file."""
        if isinstance(source, dict):
            return cls.model_validate(source)
        if isinstance(source, str) and source.lstrip().startswith("{"):
            return cls.model_validate_json(source)
        return cls.model_validate_json(Path(source).read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def request_json_schema() -> dict[str, Any]:
    """JSON schema of the request document (kept in sync with schemas/request.schema.json)."""
    schema = Request.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "cubesip request"
    return schema


def write_request_json_schema(path: str | Path) -> None:
    """Regenerate the committed schema file (see scripts/export_schema.py)."""
    text = json.dumps(request_json_schema(), indent=2) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")
