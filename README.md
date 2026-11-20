# cubesip

Retrieve remote sensing and climate data for a site or region, addressed by a
[geohash](https://en.wikipedia.org/wiki/Geohash), from openEO back-ends, any STAC catalogue,
Earth Data Hub, TESSERA embeddings or local Zarr/NetCDF/GeoTIFF, with a shared CSV manifest
that avoids downloading the same data twice.

See [plan.MD](plan.MD) for the goals and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for
the design decisions and milestone status.

## Setup

```bash
uv sync
cp cubesip.example.yaml cubesip.yaml   # then add your credentials (git-ignored)
uv run pytest
```

The `tessera` driver needs the optional `tessera` extra (geotessera, Python 3.12+):
`uv sync --extra tessera`, or `pip install 'cubesip[tessera]'`.

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

[examples/polygon_ndvi_lst_soil_moisture.ipynb](examples/polygon_ndvi_lst_soil_moisture.ipynb)
is a guided tour of the package. Starting from a polygon given as WKT, it fetches a year of
Sentinel-2 NDVI (10 m), Landsat 8/9 land-surface temperature (30 m, via Planetary
Computer STAC) and CLMS Sentinel-1 surface soil moisture (1 km). It then maps the annual
means inside the polygon and plots the polygon-averaged time series. Along the way it uses
`containing_geohash`, `check`, `run`, `regrid` and `mask_to_polygon`. Open it in Jupyter or
VS Code with the `examples` group installed (`uv sync --group examples`); the first run
downloads about 150 MB from the Copernicus Data Space Ecosystem.

### Drivers

A request names one driver, which decides what `dataset` means and where the data come
from: `openeo` and `cdse_federation` (openEO collections), `stac` (any STAC collection, via
openEO), `edh` (Earth Data Hub), `tessera` (TESSERA embeddings) and `local` (Zarr, NetCDF,
GeoTIFF). See [docs/drivers.md](docs/drivers.md) for what each one needs, how to filter
scenes, and which filter operators the back-ends accept.

### Requests

Request documents follow [schemas/request.schema.json](schemas/request.schema.json); see
[examples/](examples/).

| Field | Meaning |
|---|---|
| `driver` | `openeo`, `cdse_federation`, `stac`, `edh`, `tessera` or `local` (see [docs/drivers.md](docs/drivers.md)) |
| `dataset` | openEO collection id (`openeo`, `cdse_federation`); STAC collection URL (`stac`); EDH dataset path (`era5/era5-single-levels-atmosphere-v0`); TESSERA store version (`v1.1`); local path |
| `variables` | bands / variables to retrieve |
| `geohash` | area of interest (the geohash cell) |
| `time` | `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`, both days included; `null` for static data |
| `target_dir` | where the NetCDF is written |
| `options` | `openeo`, `cdse_federation`, `stac`: `max_cloud_cover`, `properties` (`{"platform": {"eq": "landsat-9"}}`; CDSE accepts only `eq`, `lte` and `gte`, see [docs/drivers.md](docs/drivers.md#which-operators-work)); `tessera`: `bands`, `depth` |

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

### Processing

`cubesip.processing` holds helpers that work on requests and results.

`containing_geohash` gives the smallest geohash cell that fully contains a geometry, given as
WKT in EPSG:4326 (longitude first). Use it to request data for a field or a region:

```python
from cubesip.processing import containing_geohash, regrid

containing_geohash("POLYGON ((-3.71 45.29, -3.69 45.29, -3.69 45.31, -3.71 45.31, -3.71 45.29))")
# 'gbj3'
containing_geohash("POINT (-3.7 45.3)", max_precision=6)  # a point gets a 12-character cell
```

Geohash cells nest inside 45 x 45 degree level-1 cells, so a geometry crossing the equator,
the prime meridian or another multiple of 45 degrees is in no cell and raises `ValueError`.

`regrid` resamples one or more results onto the grid of a reference result (its CRS and
cells), with rasterio/GDAL:

```python
s2 = cubesip.run(s2_request)  # Sentinel-2 for a geohash (UTM, 10 m)
era5 = cubesip.run(era5_request)  # ERA5 for the same geohash (lat/lon, 0.25 deg)
tessera = cubesip.run(tessera_request)  # TESSERA embeddings (UTM, 10 m)

t2m = regrid(era5, s2)  # ERA5 on the Sentinel-2 grid
t2m, emb = regrid([era5, tessera], s2)  # a list gives a list
```

- Inputs are `Result` objects, Datasets or DataArrays; a DataArray (e.g. an NDVI you
  computed) comes back as a DataArray.
- `method` is a rasterio resampling name: `nearest` (default, right for classes and masks
  such as `SCL`), `bilinear` or `cubic` to interpolate onto a finer grid, `average`, `min`,
  `max`, `med`, `mode` or `sum` to aggregate onto a coarser one.
- Only x/y change: time, band and other dimensions are kept as they are, not aligned with
  the reference's.
- Values come back as floats (integer bands are converted), NaN where the source has no
  data. The result is lazy when the source is.
- A source with a single cell along x or y, such as ERA5 for a level-6 geohash, has no cell
  size, so that cell is taken to cover the whole reference. That is right when the reference
  is for the same geohash or one inside it: cubesip only returns one cell when the request
  area lies inside it. The reference itself needs at least two cells along x and y.

`mask_to_polygon` blanks out the cells of a result outside a polygon, such as a field
boundary, given as WKT in EPSG:4326 (longitude first):

```python
from cubesip.processing import mask_to_polygon

field = "POLYGON ((-3.705 45.298, -3.695 45.298, -3.695 45.304, -3.705 45.304, -3.705 45.298))"
s2_field = mask_to_polygon(s2, field)  # NaN outside the field
s2_field = mask_to_polygon(s2, field, fill=0)  # 0 instead, keeping integer bands as integers
era5_field = mask_to_polygon(era5, field, all_touched=True)  # keep the cells the field touches
```

- The polygon (POLYGON or MULTIPOLYGON) is projected onto the result's grid, so it works for
  UTM and lat/lon results alike, including 0..360 longitudes.
- A cell is kept when its centre is inside the polygon. `all_touched=True` keeps every cell
  the polygon touches: use it for polygons not much bigger than a cell, such as a field in
  ERA5, whose single cell (no known size) is then kept when the polygon is inside the result.
- Every variable on the x/y grid is masked; the grid itself is not cropped. With the default
  NaN fill, integer bands become floats. The result is lazy when the source is, and a
  DataArray comes back as a DataArray.
- A warning is issued when no cell is kept, which usually means latitude and longitude are
  swapped in the WKT.

## Tests

```bash
uv run pytest              # offline suite
uv run pytest -m live      # real openEO / STAC / Earth Data Hub calls, needs cubesip.yaml
```
