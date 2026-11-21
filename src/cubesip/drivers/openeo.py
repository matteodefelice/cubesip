"""openEO drivers: collections of an openEO back-end, and the machinery they share.

- ``openeo``: the collections of ONE openEO back-end, the one set by ``openeo.url`` in the
  config. Any back-end listed on https://hub.openeo.org works; the default is the Copernicus
  Data Space Ecosystem (https://openeo.dataspace.copernicus.eu).
- ``cdse_federation``: the collections of the CDSE openEO Federation
  (https://openeofed.dataspace.copernicus.eu), which routes each request to CDSE or
  Terrascope and so also offers e.g. ``AGERA5`` and ``TERRASCOPE_S2_NDVI_V2``. Same CDSE
  account, its own ``cdse_federation`` config section.

For both, ``dataset`` is a collection id (``SENTINEL2_L2A``) read with ``load_collection``; a
STAC collection URL is refused and belongs to the ``stac`` driver (``drivers/stac.py``), which
reuses the :class:`OpenEOBackendDriver` base with ``load_stac``.

- spatial_extent: geohash box as {"west", "south", "east", "north", "crs": 4326}
- temporal_extent: [start, end + 1 day] (openEO end bound is exclusive)
- options:
    ``max_cloud_cover``: number; native argument of load_collection, an ``eo:cloud_cover``
      property filter for load_stac
    ``properties``: {"<property>": {"<op>": value}} with op in eq/neq/lt/lte/gt/gte
- synchronous ``cube.download(out_path, format="netCDF")`` only (batch jobs: future work)

Collection metadata (bands, resolution) is public and read without logging in; the size
estimate counts acquisition dates with a STAC item search (the STAC collection an openEO
collection links to as ``derived_from``, or the collection itself for load_stac).

Docs: https://open-eo.github.io/openeo-python-client/
"""

from __future__ import annotations

import logging
import math
import operator
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from cubesip.config import OpenEOConfig
from cubesip.drivers.base import (
    DatasetInfo,
    DatasetNotFoundError,
    Driver,
    EmptyResultError,
    SizeEstimate,
)
from cubesip.request import Request
from cubesip.spatial import to_native_bounds

log = logging.getLogger(__name__)

BYTES_PER_VALUE = 4  # openEO NetCDF output is typically float32
MAX_STAC_ITEMS = 5000
_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "eq": operator.eq,
    "neq": operator.ne,
    "lt": operator.lt,
    "lte": operator.le,
    "gt": operator.gt,
    "gte": operator.ge,
}


def is_url(dataset: str) -> bool:
    """True if the dataset is a web address (starts with http:// or https://)."""
    return dataset.strip().startswith(("http://", "https://"))


class OpenEOBackendDriver(Driver):
    """Shared by the drivers that run on an openEO back-end: connection and login, metadata
    cache, size estimate and synchronous download.

    Subclasses say which config section holds the back-end (``config_section``), how to read
    the dataset's metadata and how to build the cube.
    """

    supported_options = frozenset({"max_cloud_cover", "properties"})
    config_section: ClassVar[str]

    def __init__(self, config) -> None:
        """Set up the driver. The connection, the login and the metadata come later, only when
        needed.
        """
        super().__init__(config)
        self._connection = None
        self._authenticated = False
        self._metadata: dict[str, dict] = {}

    @property
    def section(self) -> str:
        """Name of the config section describing the back-end the requests run on."""
        return self.config_section

    @property
    def backend(self) -> OpenEOConfig:
        """Settings of the back-end used by this driver."""
        return getattr(self.config, self.section)

    @property
    def backend_url(self) -> str:
        """The back-end address, without a trailing '/'."""
        return self.backend.url.rstrip("/")

    @property
    def max_request_bytes(self) -> int | None:
        """Largest download this back-end allows in one go (from the config); the check warns above
        it.
        """
        return self.backend.max_sync_bytes

    def validate(self, request: Request) -> None:
        """Do the usual checks, and also check that the 'properties' filters are well formed."""
        super().validate(request)
        property_filters(request.options)  # raises ValueError on malformed filters

    # -- connection -------------------------------------------------------------------

    def connection(self, authenticate: bool = False):
        """Connect to the back-end (once, then reuse the connection) and log in if needed."""
        import openeo

        if self._connection is None:
            self._connection = openeo.connect(self.backend.url)
        if authenticate and not self._authenticated:
            self._authenticate(self._connection)
            self._authenticated = True
        return self._connection

    def _authenticate(self, connection) -> None:
        """Log in to the back-end with the method chosen in the config: browser login (oidc), client
        id and secret, or user name and password.
        """
        auth = self.backend.auth
        if auth.method == "oidc":
            # stored refresh token, else device-code flow (prints a URL to open once);
            # client_id None: the back-end's default client
            connection.authenticate_oidc(provider_id=auth.provider_id, client_id=auth.client_id)
        elif auth.method == "client_credentials":
            if not auth.client_id or auth.client_secret is None:
                raise PermissionError(
                    f"{self.section}.auth: client_credentials needs client_id/secret"
                )
            connection.authenticate_oidc_client_credentials(
                client_id=auth.client_id,
                client_secret=auth.client_secret.get_secret_value(),
                provider_id=auth.provider_id,
            )
        else:
            if not auth.username or auth.password is None:
                raise PermissionError(f"{self.section}.auth: basic needs username and password")
            connection.authenticate_basic(auth.username, auth.password.get_secret_value())

    # -- per-driver hooks -------------------------------------------------------------

    @abstractmethod
    def _read_metadata(self, dataset_key: str) -> dict:
        """Read the dataset description; raise DatasetNotFoundError if it does not exist."""

    @abstractmethod
    def search_target(self, meta: dict, dataset_key: str) -> tuple[str, str] | None:
        """Where acquisition dates can be counted: a STAC API address and collection id (None if
        unknown).
        """

    @abstractmethod
    def load(self, request: Request):
        """Prepare, without running it, the openEO job that loads the requested data."""

    # -- metadata ---------------------------------------------------------------------

    def metadata(self, request: Request) -> dict:
        """Get the dataset description, reading it only the first time and reusing it afterwards."""
        key = self.dataset_key(request.dataset)
        if key not in self._metadata:
            self._metadata[key] = self._read_metadata(key)
        return self._metadata[key]

    def describe(self, request: Request) -> DatasetInfo:
        """Report which bands the dataset has, which requested ones are missing, and its CRS when it
        has a single one.
        """
        meta = self.metadata(request)
        bands = available_bands(meta)
        missing = [] if bands is None else [v for v in request.variables if v not in bands]
        info = DatasetInfo(
            dataset=self.dataset_key(request.dataset),
            available_variables=bands,
            missing_variables=missing,
            crs=_single_epsg(meta),
        )
        if bands is None:
            info.notes.append("band names not listed in the collection metadata; not checked")
        return info

    # -- size -------------------------------------------------------------------------

    def estimate_size(self, request: Request) -> SizeEstimate:
        """Estimate the download size: the number of pixels in the area (from the resolution) times
        the number of bands times the number of dates, at 4 bytes per value. When the dates cannot
        be counted, one per day is assumed, which gives an upper bound.
        """
        meta = self.metadata(request)
        res = resolution_m(meta, request.variables)
        if res is None:
            return SizeEstimate(None, "unknown", "resolution not in collection metadata")
        width, height = box_size_m(request)
        nx, ny = max(1, math.ceil(width / res)), max(1, math.ceil(height / res))

        if request.time is None:
            steps, how = 1, "static"
        else:
            steps = self._count_time_steps(request, meta)
            how = "STAC search"
            if steps is None:
                steps = (request.time.end - request.time.start).days + 1
                how = "unknown, assumed daily (upper bound)"
        nbytes = nx * ny * len(request.variables) * steps * BYTES_PER_VALUE
        detail = (
            f"{nx}x{ny} px at {res:g} m, {len(request.variables)} band(s), "
            f"{steps} time step(s) ({how}), float32"
        )
        return SizeEstimate(nbytes=nbytes, method="estimate", detail=detail)

    def _count_time_steps(self, request: Request, meta: dict) -> int | None:
        """Count the distinct acquisition dates in the area and period by searching the STAC
        catalogue, skipping images that are too cloudy. Returns None when this is not possible; it
        never makes the check fail.
        """
        target = self.search_target(meta, self.dataset_key(request.dataset))
        if target is None or request.time is None:
            return None
        root, collection_id = target
        try:
            from pystac_client import Client

            search = Client.open(root).search(
                collections=[collection_id],
                bbox=list(request.bbox),
                datetime=f"{request.time.start.isoformat()}/{request.time.end.isoformat()}",
                limit=200,
                max_items=MAX_STAC_ITEMS,
            )
            max_cc = request.options.get("max_cloud_cover")
            dates = set()
            for item in search.items_as_dicts():
                props = item.get("properties", {})
                cloud = props.get("eo:cloud_cover")
                if max_cc is not None and cloud is not None and cloud > max_cc:
                    continue
                stamp = props.get("datetime") or props.get("start_datetime") or ""
                dates.add(stamp[:10])
            return len(dates)
        except Exception as exc:  # estimate only: never fail the check because of it
            log.warning("could not count STAC items at %s: %s", root, exc)
            return None

    # -- fetch ------------------------------------------------------------------------

    def fetch(self, request: Request, out_path: Path) -> Path:
        """Run the job on the back-end and download the result as NetCDF. Raises EmptyResultError if
        the back-end sends back an empty file (e.g. every image was cloudy).
        """
        cube = self.load(request)
        log.info("openEO synchronous download from %s -> %s", self.backend.url, out_path)
        cube.download(out_path, format="netCDF")
        out_path = Path(out_path)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            # Seen on the CDSE federation (Terrascope collections): HTTP 200 with an empty
            # body when no pixel in the area and period is valid, e.g. one fully cloudy scene.
            raise EmptyResultError(
                f"{self.backend.url} returned an empty result for {request.dataset} "
                f"{request.variables} @ {request.geohash}: probably no valid pixels in this "
                "area and period (all cloudy or no acquisitions); try a longer period"
            )
        return out_path


class OpenEODriver(OpenEOBackendDriver):
    """Collections of the openEO back-end set by ``openeo.url`` (default: CDSE)."""

    name = "openeo"
    config_section = "openeo"

    def validate(self, request: Request) -> None:
        """Refuse a URL (that belongs to the stac driver), then do the usual checks."""
        if is_url(request.dataset):
            raise ValueError(
                f"the {self.name} driver takes a collection id (e.g. SENTINEL2_L2A), not a URL; "
                "use the 'stac' driver for a STAC collection URL"
            )
        super().validate(request)

    def _read_metadata(self, dataset_key: str) -> dict:
        """Ask the back-end for the collection description; a missing collection raises
        DatasetNotFoundError.
        """
        from openeo.rest import OpenEoApiError

        try:
            return self.connection().describe_collection(dataset_key)
        except OpenEoApiError as exc:
            if exc.http_status_code == 404 or exc.code == "CollectionNotFound":
                raise DatasetNotFoundError(
                    f"openEO collection {dataset_key!r} not found on {self.backend.url}"
                ) from None
            raise

    def search_target(self, meta: dict, dataset_key: str) -> tuple[str, str] | None:
        """To count acquisition dates, use the STAC collection that the openEO collection says it
        comes from.
        """
        return derived_from_target(meta)

    def load(self, request: Request):
        """Prepare, without running it, the openEO job that loads the collection for the request's
        area, dates and bands, with the cloud-cover and property filters.
        """
        connection = self.connection(authenticate=True)
        filters = property_filters(request.options)
        return connection.load_collection(
            self.dataset_key(request.dataset),
            spatial_extent=spatial_extent(request),
            temporal_extent=temporal_extent(request),
            bands=list(request.variables),
            max_cloud_cover=request.options.get("max_cloud_cover"),
            properties=filters or None,
        )


class CDSEFederationDriver(OpenEODriver):
    """Collections of the CDSE openEO Federation (CDSE + Terrascope)."""

    name = "cdse_federation"
    config_section = "cdse_federation"


# -- helpers (pure, unit tested) ----------------------------------------------------------


def spatial_extent(request: Request) -> dict:
    """The geohash box in the form openEO expects (west, south, east, north, in lon/lat)."""
    box = request.bbox
    return {"west": box.west, "south": box.south, "east": box.east, "north": box.north, "crs": 4326}


def temporal_extent(request: Request) -> list[str] | None:
    """The dates in the form openEO expects: [first day, day after the last day], because openEO
    leaves the end date out. None for data without time.
    """
    if request.time is None:
        return None
    return [request.time.start.isoformat(), request.time.end_exclusive.isoformat()]


def predicate(op: str, value: Any) -> Callable[[Any], Any]:
    """Make a small test function: e.g. predicate('lte', 50) checks that a value is at most 50."""
    func = _OPS[op]
    return lambda x: func(x, value)


def property_filters(options: dict) -> dict[str, Callable[[Any], Any]]:
    """Turn the 'properties' option into test functions openEO can use: e.g. ``{'eo:cloud_cover':
    {'lte': 50}}`` becomes a check that cloud cover is at most 50. Raises ValueError if the option
    is malformed.
    """
    raw = options.get("properties") or {}
    if not isinstance(raw, dict):
        raise ValueError("options.properties must be an object {property: {op: value}}")
    filters = {}
    for prop, condition in raw.items():
        if not isinstance(condition, dict) or len(condition) != 1:
            raise ValueError(
                f"options.properties[{prop!r}] must have exactly one operator, e.g. {{'lte': 50}}"
            )
        ((op, value),) = condition.items()
        if op not in _OPS:
            raise ValueError(f"unknown operator {op!r} for {prop!r}; use one of {sorted(_OPS)}")
        filters[prop] = predicate(op, value)
    return filters


def available_bands(meta: dict) -> list[str] | None:
    """List the band names found anywhere in the dataset description (cube dimensions, item assets,
    band summaries), without repeats. Returns None if none are listed.
    """
    names: list[str] = []

    def add(values) -> None:
        """Add names to the list, skipping empty and repeated ones."""
        for value in values or []:
            if value and value not in names:
                names.append(str(value))

    for dim in (meta.get("cube:dimensions") or {}).values():
        if dim.get("type") == "bands":
            add(dim.get("values"))
    add((meta.get("item_assets") or {}).keys())
    summaries = meta.get("summaries") or {}
    for key in ("eo:bands", "bands"):
        add(b.get("name") for b in summaries.get(key) or [] if isinstance(b, dict))
    return names or None


def resolution_m(meta: dict, bands: list[str]) -> float | None:
    """Find the pixel size in metres of the finest requested band, from the dataset description.
    Sizes given in degrees are roughly converted to metres. Returns None if no size is listed.
    """
    found: list[float] = []
    summaries = meta.get("summaries") or {}
    for key in ("eo:bands", "bands"):
        for band in summaries.get(key) or []:
            if isinstance(band, dict) and band.get("name") in bands and band.get("gsd"):
                found.append(float(band["gsd"]))
    assets = meta.get("item_assets") or {}
    for band in bands:
        asset = assets.get(band) or {}
        if asset.get("gsd"):
            found.append(float(asset["gsd"]))
        for raster in asset.get("raster:bands") or []:
            if raster.get("spatial_resolution"):
                found.append(float(raster["spatial_resolution"]))
    if not found:
        gsd = summaries.get("gsd")
        if isinstance(gsd, list) and gsd:
            found.append(min(float(g) for g in gsd))
        elif isinstance(gsd, int | float):
            found.append(float(gsd))
    if not found:
        return None
    res = min(found)
    return res * 111_320 if res < 1 else res  # degrees -> metres, roughly


def box_size_m(request: Request) -> tuple[float, float]:
    """Width and height in metres of the geohash box, measured in the local UTM zone."""
    box = request.bbox
    lon = (box.west + box.east) / 2
    lat = (box.south + box.north) / 2
    zone = min(int((lon + 180) // 6) + 1, 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    native = to_native_bounds(box, f"EPSG:{epsg}")
    return native.east - native.west, native.north - native.south


def fetch_stac_collection(url: str) -> dict:
    """Download a STAC collection description from a URL. Raises DatasetNotFoundError if it is
    missing or is not a collection.
    """
    import requests

    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        raise DatasetNotFoundError(f"STAC collection not found: {url}")
    response.raise_for_status()
    meta = response.json()
    if meta.get("type") != "Collection":
        raise DatasetNotFoundError(f"{url} is not a STAC collection (type={meta.get('type')})")
    return meta


def derived_from_target(meta: dict) -> tuple[str, str] | None:
    """For an openEO collection: find the STAC collection it says it comes from (its 'derived_from'
    link) and return that STAC API address and collection id.

    The openEO collection's own ``root`` link points at the openEO API, not a STAC API, so
    the root is taken from the ``derived_from`` URL instead.
    """
    links = meta.get("links") or []
    derived = [link.get("href") for link in links if link.get("rel") == "derived_from"]
    stac = [href for href in derived if href and "/collections/" in href]
    if not stac:
        return None
    root, collection_id = stac[0].rstrip("/").rsplit("/collections/", 1)
    return root.rstrip("/"), collection_id


def stac_search_target(meta: dict, collection_url: str) -> tuple[str, str] | None:
    """For a STAC collection description: its STAC API address (the 'root' link, or guessed from the
    URL) and its id. Returns None if either is missing.
    """
    roots = [link.get("href") for link in meta.get("links") or [] if link.get("rel") == "root"]
    root = roots[0] if roots and roots[0] else collection_url.split("/collections/")[0]
    collection_id = meta.get("id")
    if not collection_id or not root:
        return None
    return root.rstrip("/"), str(collection_id)


def _single_epsg(meta: dict) -> str | None:
    """The dataset's CRS as 'EPSG:xxxx' when the description lists exactly one, else None."""
    epsg = (meta.get("summaries") or {}).get("proj:epsg")
    if isinstance(epsg, list) and len(epsg) == 1:
        epsg = epsg[0]
    return f"EPSG:{epsg}" if isinstance(epsg, int) else None
