# How drivers work, and how to write a new one

This page explains, in plain terms, how cubesip talks to data sources and what you need to
write to add a new one. For what each existing driver does and its options, see
[drivers.md](drivers.md).

## The idea in one paragraph

cubesip does the same job for every data source: take a request (a dataset, some variables,
a geohash cell, a period), check whether a file on disk already covers it, and if not,
download the data and save them as a NetCDF file listed in the manifest. Only one part of
this changes from source to source: *how to ask the source for data*. That part is the
**driver**. Everything else (caching, file names, the manifest, the pre-flight check,
`cubesip.split()`) is shared and lives outside the drivers.

So a driver is a small class that answers three questions about a request:

1. **Does the dataset exist, and does it have these variables?** (`describe`)
2. **How big will the download be?** (`estimate_size`)
3. **Download it and save it as NetCDF here.** (`fetch`)

## Where drivers sit

```
                cubesip.check(req)            cubesip.run(req)
                       |                             |
                       v                             v
            +--------------------------------------------------+
            |  shared code (api.py, checker.py, manifest.py,   |
            |  output.py, spatial.py)                          |
            |   - validate the request                         |
            |   - look for a cached file in the manifest       |
            |   - choose the output file name                  |
            |   - write a .partial.nc file, then rename it     |
            |   - record the new file in the manifest          |
            +--------------------------------------------------+
                       |   calls the driver's methods
                       v
            +--------------------------------------------------+
            |  driver (drivers/<name>.py)                      |
            |   validate · dataset_key · backend_url           |
            |   describe · estimate_size · fetch               |
            +--------------------------------------------------+
                       |
                       v
               the data source (openEO, Zarr over HTTP, GeoTIFF tiles, local files...)
```

The driver never touches the manifest, never decides file names and never checks the cache.
It only reads from its source and writes the one NetCDF file it is given.

## What happens during `cubesip.run`

In order, `run` ([api.py](../src/cubesip/api.py)) does this:

1. `get_driver(req.driver, config)` creates the driver named in the request.
2. `driver.validate(req)` refuses a request the driver cannot handle.
3. `driver.dataset_key(req.dataset)` and `driver.backend_url` give the dataset name and
   service address used to look things up in the manifest.
4. If the manifest has a file for the same driver, dataset, back-end and options, whose
   geohash is the request's or a parent of it, and whose period and variables cover the
   request, that file is opened and cut to the request. **The driver is not asked for
   anything else.**
5. Otherwise `driver.describe(req)` checks the dataset and variables.
6. `driver.fetch(req, path)` downloads and writes the NetCDF file.
7. The file is opened again and one row per variable is added to the manifest.

`cubesip.check` ([checker.py](../src/cubesip/checker.py)) runs steps 1 to 5 and then
`driver.estimate_size(req)`, without downloading. It turns errors into lines of the report
instead of raising them.

## The driver interface

All drivers inherit from `Driver` in [drivers/base.py](../src/cubesip/drivers/base.py).

| Member | Must you write it? | What it does |
|---|---|---|
| `name` | yes | The driver's name, as written in the request's `driver` field. |
| `supported_options` | if the driver has options | The keys allowed in `request.options`. Anything else is refused by the default `validate`. |
| `describe(request)` | yes | Check the dataset exists (raise `DatasetNotFoundError` if not) and return a `DatasetInfo`: available variables, missing ones, CRS, notes (notes become warnings in `check`). |
| `estimate_size(request)` | yes | Return a `SizeEstimate`: bytes, and whether this is `"exact"`, an `"estimate"` or `"unknown"`. |
| `fetch(request, out_path)` | yes | Download, build an `xarray.Dataset`, write it to `out_path` with `write_netcdf`, return the path. |
| `validate(request)` | optional | Extra checks on the request (e.g. "time must be whole years"). Call `super().validate(request)` first so unknown options are still refused. Raise `ValueError`. |
| `dataset_key(dataset)` | optional | Turn the user's `dataset` into one standard spelling (e.g. drop a trailing `/`, expand `v1.1` to a URL). Two spellings of the same dataset must give the same key, or the cache will not find the file. |
| `backend_url` | optional | Address of the service. Stored in the manifest so a file from one service is never reused for another. Empty by default. |
| `max_request_bytes` | optional | Size above which `check` warns. `None` (no warning) by default. |

### Shortcut: `XarraySourceDriver`

If xarray can open your source lazily (a Zarr store, NetCDF, anything `xr.open_dataset`
reads), inherit from `XarraySourceDriver` instead. You then write a single method,
`_open(dataset_key)`, that returns the opened dataset without loading it.
`describe`, `estimate_size` (exact, from the selected array sizes) and `fetch` are already
done for you, using `spatial.subset` to cut the dataset to the geohash, the period and the
variables. [local.py](../src/cubesip/drivers/local.py) (about 30 lines of driver code) and
[edh.py](../src/cubesip/drivers/edh.py) are built this way.

When the source is not a lazy xarray dataset (an API, a tiled archive, a library with its
own reader), inherit from `Driver` and write the three methods yourself, like
[tessera.py](../src/cubesip/drivers/tessera.py) and
[beeldmateriaal.py](../src/cubesip/drivers/beeldmateriaal.py).

## Rules every driver follows

These keep the cache correct and the output files alike across sources.

- **Keep the source's own grid and CRS.** Do not reproject. Write the CRS into the file
  (a `spatial_ref` coordinate with CF grid-mapping attributes, and `grid_mapping` on each
  variable), so `spatial.native_crs` can read it back. The cache cuts child geohashes out
  of parent files, which needs the CRS.
- **Return values as the source stores them.** No scale factors, no unit conversion, no
  derived indices. Users do that afterwards.
- **Cover the whole geohash cell and the whole period.** A cached file is reused for any
  request inside it, so it must really contain everything the request asked for.
- **Dimensions are named `time`, `y`/`x` (or `lat`/`lon`).** Leave `time` out only for
  static data (the request then has `time: null`).
- **Write with `write_netcdf(ds, out_path, provenance_attrs(...))`** from
  [output.py](../src/cubesip/output.py). It compresses the data, drops source encodings and
  stores where the file came from. Write to the `out_path` you are given: it is a temporary
  `.partial.nc` name that `run` renames once you return.
- **Fail with the right error, before writing anything:**
  - `DatasetNotFoundError`: the dataset does not exist.
  - `PermissionError`: credentials are missing or refused. Never put a token in an error
    message or a URL.
  - `ValueError`: the request makes no sense for this source (wrong years, bad option).
  - `EmptyResultError`: the source answered, but there are no data at all for this cell and
    period. Nothing is written or recorded.
- **Load heavy libraries inside methods, not at the top of the module**, when they are
  optional or slow to import. Drivers are themselves imported only when used.
- **Options change the file identity.** `request.options` is part of the cache key and the
  file name, so only put things there that change the downloaded data.

## Adding a new driver, step by step

Say you want a driver called `mysource`.

### 1. Write the class

Create `src/cubesip/drivers/mysource.py`. A module docstring that says what `dataset`,
`variables`, `time` and `options` mean for this source helps the next reader a lot (see the
top of `tessera.py`).

The smallest version, for a source xarray can open:

```python
"""mysource driver: <one line on what the data are and where they come from>.

- dataset: <what the user writes, e.g. a product id>
- variables: <names>
- auth: <none / config section mysource.token>
"""

from __future__ import annotations

import xarray as xr

from cubesip.drivers.base import DatasetNotFoundError, XarraySourceDriver


class MySourceDriver(XarraySourceDriver):
    name = "mysource"

    @property
    def backend_url(self) -> str:
        return "https://data.mysource.example"

    def _open(self, dataset_key: str) -> xr.Dataset:
        url = f"{self.backend_url}/{dataset_key}.zarr"
        try:
            return xr.open_dataset(url, engine="zarr", chunks={}, decode_coords="all")
        except FileNotFoundError as exc:
            raise DatasetNotFoundError(f"mysource dataset not found: {url}") from exc
```

The full version, for any other source:

```python
from pathlib import Path

from cubesip.drivers.base import DatasetInfo, Driver, EmptyResultError, SizeEstimate
from cubesip.output import provenance_attrs, write_netcdf
from cubesip.request import Request


class MySourceDriver(Driver):
    name = "mysource"
    supported_options = frozenset({"resolution"})

    def validate(self, request: Request) -> None:
        super().validate(request)  # refuses unknown options
        if request.time is None:
            raise ValueError("mysource needs a time range")

    def describe(self, request: Request) -> DatasetInfo:
        available = [...]  # ask the source which variables it has
        return DatasetInfo(
            dataset=self.dataset_key(request.dataset),
            available_variables=available,
            missing_variables=[v for v in request.variables if v not in available],
            crs="EPSG:32631",
        )

    def estimate_size(self, request: Request) -> SizeEstimate:
        nbytes = ...  # pixels x variables x time steps x bytes per value
        return SizeEstimate(nbytes=nbytes, method="estimate", detail="...")

    def fetch(self, request: Request, out_path: Path) -> Path:
        box = request.bbox  # west, south, east, north in degrees (EPSG:4326)
        ds = ...  # read the data for box and request.time into an xarray.Dataset
        if not any(ds[v].notnull().any() for v in ds.data_vars):  # nothing at all
            raise EmptyResultError(f"no mysource data at {request.geohash}")
        attrs = provenance_attrs(request, self.dataset_key(request.dataset), self.backend_url)
        return write_netcdf(ds, out_path, attrs)
```

Useful helpers: `request.bbox` (the geohash cell in degrees), `request.time.start` /
`.end` (inclusive days) and `.end_exclusive`, `spatial.to_native_bounds(bbox, crs)` (the
cell in the source's CRS) and `spatial.subset(ds, bbox, time, variables)`.

### 2. Register it

- [drivers/\_\_init\_\_.py](../src/cubesip/drivers/__init__.py): add
  `"mysource": ("cubesip.drivers.mysource", "MySourceDriver")` to `_REGISTRY`, and a line
  to the table in the module docstring.
- [request.py](../src/cubesip/request.py): add `"mysource"` to `DriverName`, and describe
  it in the `driver`, `dataset` (and `options`, if any) field descriptions.
- Regenerate the JSON schema:

  ```bash
  uv run python scripts/export_schema.py
  ```

### 3. Configuration and dependencies (only if needed)

- **Credentials or a service address:** add a section to `Config` in
  [config.py](../src/cubesip/config.py) (see `EarthDataHubConfig`), store secrets as
  `SecretStr`, add the section name to the empty-value cleanup in `load_config`, and add a
  commented example to [cubesip.example.yaml](../cubesip.example.yaml). Read it in the
  driver as `self.config.mysource`.
- **A new library:** if most users will not need it, make it an optional extra in
  [pyproject.toml](../pyproject.toml) (like `tessera`) and raise an `ImportError` that says
  how to install it (see `_require_geotessera` in `tessera.py`).

### 4. Test it

Add `tests/test_mysource.py`. Tests run offline: replace the remote source with a small
fake (see `FakeStore` in [test_tessera.py](../tests/test_tessera.py)), or with a
synthetic dataset from [conftest.py](../tests/conftest.py). Cover at least:

- `validate` refuses bad requests and unknown options;
- `describe` reports missing variables and raises `DatasetNotFoundError` for a missing
  dataset;
- `fetch` writes a file with the right variables, dimensions and CRS;
- a full `cubesip.run` writes a manifest row, and a second run (or a child geohash) is
  served from the cache.

A test that calls the real service goes in [test_live.py](../tests/test_live.py) with the
`live` marker; it is skipped by default and run with `uv run pytest -m live`.

### 5. Document it

- Add a row to the table in [drivers.md](drivers.md) and, if the driver has its own rules
  (options, time steps, sizes), a section below it.
- Add an example request in [examples/](../examples/) (`request_mysource.json`).
- Mention the source in the first paragraph of the [README](../README.md).

## Checklist

- [ ] `drivers/mysource.py` with `name`, `describe`, `estimate_size`, `fetch` (or `_open`)
- [ ] `validate` / `dataset_key` / `backend_url` / `supported_options` where they apply
- [ ] Output keeps the native CRS and raw values, covers the whole cell and period
- [ ] Registered in `_REGISTRY` and `DriverName`; schema regenerated
- [ ] Config section and optional extra, if needed
- [ ] Offline tests, and a `live` test if there is a remote service
- [ ] `drivers.md`, an example request, the README
