"""openEO driver: back-end collections (load_collection) and arbitrary STAC (load_stac).

- dataset starting with http(s):// -> ``connection.load_stac(url, ...)``
- otherwise                        -> ``connection.load_collection(collection_id, ...)``
- spatial_extent: geohash box as {"west", "south", "east", "north", "crs": 4326}
- temporal_extent: [start, end + 1 day] (openEO end bound is exclusive)
- options:
    ``max_cloud_cover``: number; native argument of load_collection, an ``eo:cloud_cover``
      property filter for load_stac
    ``properties``: {"<property>": {"<op>": value}} with op in eq/neq/lt/lte/gt/gte
- synchronous ``cube.download(out_path, format="netCDF")`` only (batch jobs: future work)

Collection metadata (bands, resolution) is public and read without logging in; the size
estimate counts acquisition dates with a STAC item search (the collection itself for
load_stac, the STAC collection an openEO collection links to as ``derived_from``).

Docs: https://open-eo.github.io/openeo-python-client/
"""

from __future__ import annotations

import logging
import math
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cubesip.drivers.base import DatasetInfo, DatasetNotFoundError, Driver, SizeEstimate
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
    return dataset.startswith(("http://", "https://"))


class OpenEODriver(Driver):
    name = "openeo"
    supported_options = frozenset({"max_cloud_cover", "properties"})

    def __init__(self, config) -> None:
        super().__init__(config)
        self._connection = None
        self._authenticated = False
        self._metadata: dict[str, dict] = {}

    @property
    def max_request_bytes(self) -> int | None:
        return self.config.openeo.max_sync_bytes

    def dataset_key(self, dataset: str) -> str:
        value = dataset.strip()
        return value.rstrip("/") if is_url(value) else value

    def validate_options(self, request: Request) -> None:
        super().validate_options(request)
        property_filters(request.options)  # raises ValueError on malformed filters

    # -- connection -------------------------------------------------------------------

    def connection(self, authenticate: bool = False):
        """Connect to ``openeo.url``; log in with the configured method when needed."""
        import openeo

        if self._connection is None:
            self._connection = openeo.connect(self.config.openeo.url)
        if authenticate and not self._authenticated:
            self._authenticate(self._connection)
            self._authenticated = True
        return self._connection

    def _authenticate(self, connection) -> None:
        auth = self.config.openeo.auth
        if auth.method == "oidc":
            # stored refresh token, else device-code flow (prints a URL to open once)
            connection.authenticate_oidc(provider_id=auth.provider_id)
        elif auth.method == "client_credentials":
            if not auth.client_id or auth.client_secret is None:
                raise PermissionError("openeo.auth: client_credentials needs client_id/secret")
            connection.authenticate_oidc_client_credentials(
                client_id=auth.client_id,
                client_secret=auth.client_secret.get_secret_value(),
                provider_id=auth.provider_id,
            )
        else:
            if not auth.username or auth.password is None:
                raise PermissionError("openeo.auth: basic needs username and password")
            connection.authenticate_basic(auth.username, auth.password.get_secret_value())

    # -- metadata ---------------------------------------------------------------------

    def metadata(self, request: Request) -> dict:
        """Collection description (openEO) or STAC collection JSON (load_stac)."""
        key = self.dataset_key(request.dataset)
        if key not in self._metadata:
            self._metadata[key] = self._fetch_stac(key) if is_url(key) else self._describe(key)
        return self._metadata[key]

    def _describe(self, collection_id: str) -> dict:
        from openeo.rest import OpenEoApiError

        try:
            return self.connection().describe_collection(collection_id)
        except OpenEoApiError as exc:
            if exc.http_status_code == 404 or exc.code == "CollectionNotFound":
                raise DatasetNotFoundError(
                    f"openEO collection {collection_id!r} not found on {self.config.openeo.url}"
                ) from None
            raise

    @staticmethod
    def _fetch_stac(url: str) -> dict:
        import requests

        response = requests.get(url, timeout=60)
        if response.status_code == 404:
            raise DatasetNotFoundError(f"STAC collection not found: {url}")
        response.raise_for_status()
        meta = response.json()
        if meta.get("type") != "Collection":
            raise DatasetNotFoundError(f"{url} is not a STAC collection (type={meta.get('type')})")
        return meta

    def describe(self, request: Request) -> DatasetInfo:
        meta = self.metadata(request)
        bands = available_bands(meta)
        info = DatasetInfo(
            dataset=self.dataset_key(request.dataset),
            available_variables=bands,
            missing_variables=[v for v in request.variables if bands and v not in bands],
            crs=_single_epsg(meta),
        )
        if bands is None:
            info.notes.append("band names not listed in the collection metadata; not checked")
        return info

    # -- size -------------------------------------------------------------------------

    def estimate_size(self, request: Request) -> SizeEstimate:
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
        target = stac_search_target(meta, self.dataset_key(request.dataset))
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

    def load(self, request: Request):
        """Build the (lazy) openEO DataCube for the request."""
        key = self.dataset_key(request.dataset)
        connection = self.connection(authenticate=True)
        common = {
            "spatial_extent": spatial_extent(request),
            "temporal_extent": temporal_extent(request),
            "bands": list(request.variables),
        }
        filters = property_filters(request.options)
        if is_url(key):
            max_cc = request.options.get("max_cloud_cover")
            if max_cc is not None:
                filters.setdefault("eo:cloud_cover", _predicate("lte", max_cc))
            return connection.load_stac(key, properties=filters or None, **common)
        return connection.load_collection(
            key,
            max_cloud_cover=request.options.get("max_cloud_cover"),
            properties=filters or None,
            **common,
        )

    def fetch(self, request: Request, out_path: Path) -> Path:
        cube = self.load(request)
        log.info("openEO synchronous download -> %s", out_path)
        cube.download(out_path, format="netCDF")
        return Path(out_path)


# -- helpers (pure, unit tested) ----------------------------------------------------------


def spatial_extent(request: Request) -> dict:
    box = request.bbox
    return {"west": box.west, "south": box.south, "east": box.east, "north": box.north, "crs": 4326}


def temporal_extent(request: Request) -> list[str] | None:
    if request.time is None:
        return None
    return [request.time.start.isoformat(), request.time.end_exclusive.isoformat()]


def _predicate(op: str, value: Any) -> Callable[[Any], Any]:
    func = _OPS[op]
    return lambda x: func(x, value)


def property_filters(options: dict) -> dict[str, Callable[[Any], Any]]:
    """``{"eo:cloud_cover": {"lte": 50}}`` -> ``{"eo:cloud_cover": lambda x: x <= 50}``."""
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
        filters[prop] = _predicate(op, value)
    return filters


def available_bands(meta: dict) -> list[str] | None:
    """Band names from cube:dimensions, item_assets and eo:bands/bands summaries."""
    names: list[str] = []

    def add(values) -> None:
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
    """Finest ground sampling distance (m) among the requested bands."""
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
    """Width/height (m) of the geohash box in the local UTM zone."""
    box = request.bbox
    lon = (box.west + box.east) / 2
    lat = (box.south + box.north) / 2
    zone = min(int((lon + 180) // 6) + 1, 60)
    epsg = (32600 if lat >= 0 else 32700) + zone
    native = to_native_bounds(box, f"EPSG:{epsg}")
    return native.east - native.west, native.north - native.south


def stac_search_target(meta: dict, dataset_key: str) -> tuple[str, str] | None:
    """(STAC API root, collection id) where acquisition dates can be counted."""
    links = meta.get("links") or []
    if is_url(dataset_key):
        collection_url, collection_id = dataset_key, meta.get("id")
    else:
        derived = [link.get("href") for link in links if link.get("rel") == "derived_from"]
        stac = [h for h in derived if h and "/collections/" in h]
        if not stac:
            return None
        collection_url = stac[0].rstrip("/")
        collection_id = collection_url.rsplit("/", 1)[-1]
        links = []  # the openEO links point at the openEO API, not the STAC API
    roots = [link.get("href") for link in links if link.get("rel") == "root"]
    root = roots[0] if roots and roots[0] else collection_url.split("/collections/")[0]
    if not collection_id or not root:
        return None
    return root.rstrip("/"), str(collection_id)


def _single_epsg(meta: dict) -> str | None:
    epsg = (meta.get("summaries") or {}).get("proj:epsg")
    if isinstance(epsg, list) and len(epsg) == 1:
        epsg = epsg[0]
    return f"EPSG:{epsg}" if isinstance(epsg, int) else None
