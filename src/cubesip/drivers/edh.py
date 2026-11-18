"""Earth Data Hub driver: stream Zarr stores over HTTPS with xarray + zarr.

- dataset: path relative to ``earthdatahub.url`` (``era5/era5-single-levels-atmosphere-v0``,
  ``.zarr`` optional) or a full https URL
- auth: user ``edh`` + personal token, sent as an HTTP basic-auth header through fsspec
  ``storage_options`` so the token never appears in URLs, reprs or error messages
- opened lazily (``engine="zarr", chunks={}``); only the selected chunks are downloaded
- selection, exact size and NetCDF writing come from :class:`XarraySourceDriver`

Docs: https://earthdatahub.destine.eu/getting-started
"""

from __future__ import annotations

import base64

import xarray as xr

from cubesip.drivers.base import DatasetNotFoundError, XarraySourceDriver

USERNAME = "edh"


class EarthDataHubDriver(XarraySourceDriver):
    name = "edh"

    @property
    def base_url(self) -> str:
        return self.config.earthdatahub.url.rstrip("/")

    def dataset_key(self, dataset: str) -> str:
        value = dataset.strip()
        if value.startswith(self.base_url + "/"):
            value = value[len(self.base_url) + 1 :]
        value = value.rstrip("/")
        if value.endswith(".zarr"):
            value = value[: -len(".zarr")]
        return value if value.startswith(("http://", "https://")) else value.strip("/")

    def store_url(self, dataset_key: str) -> str:
        if dataset_key.startswith(("http://", "https://")):
            return f"{dataset_key}.zarr"
        return f"{self.base_url}/{dataset_key}.zarr"

    def _storage_options(self) -> dict:
        token = self.config.earthdatahub.token
        if token is None or not token.get_secret_value():
            raise PermissionError(
                "Earth Data Hub token missing: set earthdatahub.token in cubesip.yaml"
            )
        # Session-level header: aiohttp drops Authorization on cross-origin redirects.
        credentials = base64.b64encode(f"{USERNAME}:{token.get_secret_value()}".encode()).decode()
        return {"client_kwargs": {"headers": {"Authorization": f"Basic {credentials}"}}}

    def _open(self, dataset_key: str) -> xr.Dataset:
        import aiohttp

        url = self.store_url(dataset_key)
        try:
            return xr.open_dataset(
                url,
                engine="zarr",
                chunks={},
                decode_coords="all",
                storage_options=self._storage_options(),
            )
        except aiohttp.ClientResponseError as exc:
            if exc.status in (401, 403):
                raise PermissionError(
                    f"Earth Data Hub refused access to {url} (HTTP {exc.status}): "
                    "check earthdatahub.token and that your account can read this dataset"
                ) from None
            if exc.status == 404:
                raise DatasetNotFoundError(f"Earth Data Hub dataset not found: {url}") from None
            raise
        except (FileNotFoundError, KeyError) as exc:
            raise DatasetNotFoundError(f"Earth Data Hub dataset not found: {url}") from exc
        except Exception as exc:
            if type(exc).__name__.endswith("NotFoundError"):  # zarr.errors.*NotFoundError
                raise DatasetNotFoundError(f"Earth Data Hub dataset not found: {url}") from exc
            raise
