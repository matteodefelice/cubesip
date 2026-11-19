# cubesip

Retrieve remote sensing and climate data for a site or region, addressed by a
[geohash](https://en.wikipedia.org/wiki/Geohash), from openEO back-ends, any STAC catalogue,
Earth Data Hub or local Zarr/NetCDF/GeoTIFF, with a shared CSV manifest that avoids
downloading the same data twice.

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

### Drivers

A request names one driver: it decides what `dataset` means and where the data come from.

| Driver | `dataset` | Data come from | Config section |
|---|---|---|---|
| `openeo` | openEO collection id, e.g. `SENTINEL2_L2A` | the openEO back-end at `openeo.url`: any back-end listed on the [openEO Hub](https://hub.openeo.org); default Copernicus Data Space Ecosystem (CDSE) | `openeo` |
| `cdse_federation` | openEO collection id, e.g. `TERRASCOPE_S2_NDVI_V2`, `AGERA5` | the [CDSE openEO Federation](https://documentation.dataspace.copernicus.eu/APIs/openEO/federation/openeo_federation.html) (`openeofed.dataspace.copernicus.eu`), which routes each request to CDSE or Terrascope, so it has the CDSE collections plus Terrascope's | `cdse_federation` |
| `stac` | STAC collection URL, e.g. Planetary Computer `.../collections/landsat-c2-l2` | any STAC catalogue, read by openEO `load_stac` on the back-end named in `stac.backend` (`openeo` or `cdse_federation`) | `stac` |
| `edh` | Earth Data Hub dataset path, e.g. `era5/era5-single-levels-atmosphere-v0` | Earth Data Hub Zarr, streamed with xarray | `earthdatahub` |
| `local` | path to a Zarr store, NetCDF or GeoTIFF | the local file system, read with xarray | none |

The three openEO-based drivers log in with their section's `auth` and download
synchronously. A collection id given to `stac`, or a URL given to `openeo` /
`cdse_federation`, is refused with a message naming the right driver. The federation uses
the same CDSE account but its own login client, so the first `cdse_federation` download asks
for a device-code login once; the refresh token is then stored. When no pixel in the area
and period is valid (e.g. a single fully cloudy scene), the federation returns an empty
file; cubesip raises `EmptyResultError` and records nothing.

### Requests

Request documents follow [schemas/request.schema.json](schemas/request.schema.json); see
[examples/](examples/).

| Field | Meaning |
|---|---|
| `driver` | `openeo`, `cdse_federation`, `stac`, `edh` or `local` (see [Drivers](#drivers)) |
| `dataset` | openEO collection id (`openeo`, `cdse_federation`); STAC collection URL (`stac`); EDH dataset path (`era5/era5-single-levels-atmosphere-v0`); local path |
| `variables` | bands / variables to retrieve |
| `geohash` | area of interest (the geohash cell) |
| `time` | `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`, both days included; `null` for static data |
| `target_dir` | where the NetCDF is written |
| `options` | `openeo`, `cdse_federation`, `stac` only: `max_cloud_cover`, `properties` (`{"platform": {"eq": "landsat-9"}}`, operators `eq neq lt lte gt gte`) |

### Cache

Every downloaded file adds one row per variable to the CSV manifest set in `cubesip.yaml`.
A later request is served from an existing file, without downloading, when the file has the
same driver, dataset, back-end and options, covers the request's period and variables, and
its geohash is the same cell or a parent cell (`qem` covers `qem7`). Outputs keep the
source's native CRS (UTM for Sentinel-2, lat/lon for ERA5).

The back-end is the service the data came from: the openEO back-end URL (`openeo.url`,
`cdse_federation.url`, or the one `stac.backend` names) or `earthdatahub.url`; empty for
`local`. Pointing `openeo.url` at another back-end therefore downloads again instead of
reusing files from the old one. Rows written before cubesip recorded the back-end have it
empty; they are still reused, and `check()` warns that their back-end is assumed. A
manifest from an older version gets the new `backend` column the next time a row is added.

## Tests

```bash
uv run pytest              # offline suite
uv run pytest -m live      # real openEO / STAC / Earth Data Hub calls, needs cubesip.yaml
```
