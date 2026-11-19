"""STAC driver: any STAC collection, read with openEO ``load_stac``.

``dataset`` is a STAC collection URL, e.g.
https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2. Nothing is
read by cubesip itself: the openEO back-end named by ``stac.backend`` in the config
(``openeo``, the default, or ``cdse_federation``) loads the items, cuts them to the geohash
box and returns the NetCDF. It logs in with that back-end's credentials.

- band names are the STAC asset keys (``red``, ``nir08``), listed from ``item_assets`` or
  the ``eo:bands`` / ``bands`` summaries
- options: ``max_cloud_cover`` (turned into an ``eo:cloud_cover <= value`` property filter)
  and ``properties``, as for the openEO drivers
- the size estimate counts acquisition dates with an item search on the collection's own
  STAC API

Planetary Computer's signed assets work through the CDSE back-end (verified live).
"""

from __future__ import annotations

from cubesip.drivers.openeo import (
    OpenEOBackendDriver,
    fetch_stac_collection,
    is_url,
    predicate,
    property_filters,
    spatial_extent,
    stac_search_target,
    temporal_extent,
)
from cubesip.request import Request


class StacDriver(OpenEOBackendDriver):
    name = "stac"

    @property
    def section(self) -> str:
        """Config section of the openEO back-end that runs the request (set by stac.backend)."""
        return self.config.stac.backend

    def dataset_key(self, dataset: str) -> str:
        """Give the STAC collection URL in a standard form (no spaces, no trailing '/')."""
        return dataset.strip().rstrip("/")

    def validate(self, request: Request) -> None:
        """Refuse a dataset that is not a URL (that belongs to the openeo drivers), then do the
        usual checks.
        """
        if not is_url(request.dataset):
            raise ValueError(
                f"the stac driver takes a STAC collection URL, got {request.dataset!r}; "
                "use the 'openeo' or 'cdse_federation' driver for an openEO collection id"
            )
        super().validate(request)

    def _read_metadata(self, dataset_key: str) -> dict:
        """Download the STAC collection description."""
        return fetch_stac_collection(dataset_key)

    def search_target(self, meta: dict, dataset_key: str) -> tuple[str, str] | None:
        """Where to search the collection's items to count acquisition dates: its STAC API and
        collection id.
        """
        return stac_search_target(meta, dataset_key)

    def load(self, request: Request):
        """Prepare, without running it, the openEO job that loads the STAC collection for the
        request's area, dates and bands, with a cloud-cover filter if one is asked for.
        """
        connection = self.connection(authenticate=True)
        filters = property_filters(request.options)
        max_cc = request.options.get("max_cloud_cover")
        if max_cc is not None:
            filters.setdefault("eo:cloud_cover", predicate("lte", max_cc))
        return connection.load_stac(
            self.dataset_key(request.dataset),
            spatial_extent=spatial_extent(request),
            temporal_extent=temporal_extent(request),
            bands=list(request.variables),
            properties=filters or None,
        )
