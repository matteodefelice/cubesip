# cubesip

Retrieve remote sensing and climate data for a site or region, addressed by a
[geohash](https://en.wikipedia.org/wiki/Geohash), from openEO, Earth Data Hub or local
Zarr/NetCDF/GeoTIFF, with a shared CSV manifest that avoids downloading the same data twice.

See [plan.MD](plan.MD) for the goals and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for
the design decisions and milestone status.

## Setup

```bash
uv sync
cp cubesip.example.yaml cubesip.yaml   # then add your credentials (git-ignored)
uv run pytest
```

If the project folder is synced (OneDrive, Dropbox...), keep the virtual environment outside
it, e.g. `export UV_PROJECT_ENVIRONMENT=~/.venvs/cubesip` before `uv sync`.

## Usage

```python
import cubesip

# Find the geohash of a site (precision 6 is ~1.2 x 0.6 km)
cubesip.geohash.encode(lat=45.30, lon=-3.70, precision=6)

req = cubesip.Request.from_json("examples/request_s2_ndvi.json")
print(cubesip.check(req))  # cached? dataset and bands found? estimated size?
result = cubesip.run(req)  # downloads, writes NetCDF, appends to the manifest

result.path  # NetCDF file (or the cached file it was cut from)
result.from_cache  # True when served from an existing file
ds = result.dataset  # lazy xarray.Dataset, subset to the request

ndvi = (ds["B08"] - ds["B04"]) / (ds["B08"] + ds["B04"])
```

`check` and `run` read the config from `./cubesip.yaml` (or `$CUBESIP_CONFIG`); pass a path
or a `cubesip.Config` as the second argument to use another one. `run(req, force=True)`
ignores the cache.

Band values are returned as the provider stores them; cubesip does not apply scale factors.
Convert to physical units before computing indices, using the `scale`/`offset` listed in the
collection's STAC `raster:bands` metadata. For example, Landsat Collection 2 Level-2 surface
reflectance is `DN * 2.75e-5 - 0.2`; with a non-zero offset, an NDVI computed on raw values
is wrong.

### Examples

[examples/quickstart.py](examples/quickstart.py) is the shortest start: one request for a
month of ERA5 2 m temperature, then the result printed and plotted with xarray. It is split
into `# %%` cells to run one at a time in VS Code, or as a whole:

```bash
uv run --group examples python examples/quickstart.py
```

Two scripts in [examples/](examples/) compute a yearly metric for one site. Both make one
cubesip request per year, write a PNG chart and a CSV table to `data/`, and read the cached
files instead of downloading when run again. Both default to the same cropland cell in the
Beauce plain (France); pass `--latlon LAT LON` or `--geohash` for your own site.

[examples/ndvi_yearly.py](examples/ndvi_yearly.py): yearly mean NDVI from Sentinel-2 (openEO).
Clouds, shadows and snow are masked with the scene classification band; the yearly value is
the mean of monthly means, so summer's many clear scenes don't outweigh the winter.

```bash
uv run --group examples python examples/ndvi_yearly.py
uv run --group examples python examples/ndvi_yearly.py --latlon 45.30 9.50 --years 2020 2024
```

[examples/precip_p90_yearly.py](examples/precip_p90_yearly.py): yearly 90th percentile of
daily precipitation from ERA5 (Earth Data Hub). Hourly `tp` is summed into daily totals
(each value is the hour ending at its timestamp) and the percentile is taken over wet days
(>= 1 mm; `--wet-day-threshold 0` uses every day).

```bash
uv run --group examples python examples/precip_p90_yearly.py
uv run --group examples python examples/precip_p90_yearly.py --years 1995 2024 --percentile 95
```

### Requests

Request documents follow [schemas/request.schema.json](schemas/request.schema.json); see
[examples/](examples/).

| Field | Meaning |
|---|---|
| `driver` | `openeo`, `edh` (Earth Data Hub) or `local` |
| `dataset` | openEO collection id or STAC collection URL; EDH dataset path (`era5/era5-single-levels-atmosphere-v0`); local path |
| `variables` | bands / variables to retrieve |
| `geohash` | area of interest (the geohash cell) |
| `time` | `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`, both days included; `null` for static data |
| `target_dir` | where the NetCDF is written |
| `options` | openEO only: `max_cloud_cover`, `properties` (`{"platform": {"eq": "landsat-9"}}`, operators `eq neq lt lte gt gte`) |

### Cache

Every downloaded file adds one row per variable to the CSV manifest set in `cubesip.yaml`.
A later request is served from an existing file, without downloading, when the file has the
same driver, dataset and options, covers the request's period and variables, and its geohash
is the same cell or a parent cell (`qem` covers `qem7`). Outputs keep the source's native
CRS (UTM for Sentinel-2, lat/lon for ERA5).

## Tests

```bash
uv run pytest              # offline suite
uv run pytest -m live      # real openEO / Earth Data Hub calls, needs cubesip.yaml
```
