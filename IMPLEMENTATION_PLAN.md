# cubesip: implementation plan

Companion to [plan.MD](plan.MD) (the goals). This file records the design decisions and the
milestones to build it.

**Status (2026-11-18):** M1-M5 implemented and covered by the offline test suite (85 tests).
Earth Data Hub is tested against a local HTTP server that enforces `edh:<token>` basic auth.

Live runs on CDSE (`uv run pytest -m live`), login via stored refresh token:
- SENTINEL2_L2A, `u0nd9w`, 2 weeks, B04/B08: downloaded in ~10 s, float32, native
  EPSG:32632, all pixels valid; child geohash `u0nd9w1` served from cache.
- Planetary Computer `landsat-c2-l2` via `load_stac`, `u0nd9w`, 2 months, red/nir08:
  downloaded in ~15 s, all pixels valid. Estimate 10 dates / 29x21 px vs actual 10 / 30x21.

- Earth Data Hub ERA5 `era5-single-levels-atmosphere-v0`, `u0nd9` (~4 km), 2 days, t2m:
  downloaded in ~10 s, 48 hourly steps (end day inclusive), the single 0.25° cell that
  intersects the cell, plausible values; the token is in neither the NetCDF nor the manifest.

All drivers are now verified end to end. M6 (docs) partly done in README.

**Driver split (2026-11-19):** the single openEO driver is now three drivers with one meaning
each: `openeo` (collections of the back-end in `openeo.url`), `cdse_federation` (collections
of the CDSE openEO Federation) and `stac` (STAC collection URLs via `load_stac`). Live
`check()` passes for all three (S2 on CDSE, `TERRASCOPE_S2_NDVI_V2` on the federation,
Planetary Computer Landsat via `stac`). Federation downloads verified directly:
`SENTINEL2_L2A` (routed to CDSE), `AGERA5` and `TERRASCOPE_S2_NDVI_V2` (Terrascope, July 2024,
5 clear dates). The federation logs in with its own OIDC client (one device-code login, then
a stored refresh token). With no valid pixel in the request (one fully cloudy scene) it
answers HTTP 200 with an empty body; the openEO drivers now raise `EmptyResultError` instead
of leaving a 0-byte file. `auth.client_id` is now passed to the oidc login (it was ignored).

**TESSERA driver (2026-11-19):** `tessera` reads TESSERA embeddings with geotessera's Zarr API
(`GeoTesseraZarr.read_region`, one call per year), optional extra `cubesip[tessera]`
(geotessera >= 0.11, Python 3.12+). Verified live on the public v1.1 store
(`https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate`): `u094v8`, 2024, 62x83 px,
128 channels, EPSG:32631, all pixels valid, ~7 s; a child geohash is served from the cache.

**Beeldmateriaal driver (2026-11-25):** `beeldmateriaal` reads the Beeldmateriaal Nederland
8 cm orthophoto tiles (the open-data GeoTIFFs behind PDOK Luchtfoto RGB / CIR): 1 x 1 km
tiles in EPSG:28992, 3-band uint8, internally tiled JPEG (q95), no overviews; RGB and CIR
(NIR, red, green; CIR red equals RGB red). Tiles are located via the download portal's
GeoJSON index per year and product (`BM_HRL{year}O_{RGB|CIR}_TIF.json`, published for 2025
and 2026), cached as CSV in `<data_dir>/_index/`. Windows are read with rasterio over
`/vsicurl/` onto a grid snapped to the resolution (default 0.08 m). PDOK's WMS was rejected:
JPEG only, max 2500 px. Verified live: `u178kdc` (Utrecht), 2025, 4 variables,
1185x1915 px, pixels identical to a direct read of the tile, ~6 s plus ~15 s for the two
index downloads the first time; a child geohash is served from the cache.

## 1. Decisions

### Agreed with the user (2026-11-18)

| Topic | Decision |
|---|---|
| Manifest format | **CSV**, one row per variable, appends guarded by a lock file |
| Cache reuse | **Lazy view only**: on a cache hit `run()` returns the cached file path + an xarray view subset to the request. No new file, no new manifest row |
| Output grid | **Native CRS** of each source (UTM for Sentinel-2, lat/lon for ERA5). The geohash box is transformed into that CRS for clipping; no resampling |

### Defaults chosen (change here if you disagree)

| Topic | Default |
|---|---|
| Area | One geohash per request (1-12 chars), lower-cased. The geohash cell box is the area |
| Spatial selection | Keep every cell that **intersects** the box (box grown by half a cell), so a small geohash on ERA5's 0.25° grid never returns empty |
| Time | `{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}`, **both days inclusive**; `null` for static data. Drivers convert to their own convention (openEO end is exclusive -> `end + 1 day`; hourly data -> `start <= t < end + 1 day`) |
| API | Two steps: `cubesip.check(request)` (no download) then `cubesip.run(request)`. `run(force=True)` bypasses the cache |
| Manifest write | Automatic after a successful download; the new rows are also returned in `Result.manifest_rows` |
| Request input | JSON file / JSON string / dict, validated by a pydantic model; JSON schema generated into `schemas/request.schema.json` |
| Drivers | `openeo`: collections of the back-end at `openeo.url` (any on https://hub.openeo.org; default CDSE `https://openeo.dataspace.copernicus.eu`). `cdse_federation`: collections of `https://openeofed.dataspace.copernicus.eu` (CDSE + Terrascope). `stac`: STAC collection URL, `load_stac` on the back-end named in `stac.backend` (`openeo` default, or `cdse_federation`). `edh`: Earth Data Hub Zarr. `tessera`: TESSERA embeddings via geotessera, `dataset` = store version resolved to the store URL, variable `embedding` (time, band, y, x), whole calendar years only, options `bands` / `depth`. `local`: files on disk. Each driver accepts one kind of `dataset` and refuses the other kind with a pointer to the right driver |
| openEO auth | Per back-end section (`openeo`, `cdse_federation`): `oidc` (stored refresh token, device-code login the first time), `client_credentials` or `basic` |
| openEO execution | Synchronous `download()` only; checker warns when the estimate exceeds that back-end's `max_sync_bytes` |
| EDH auth | Token in YAML, sent as a basic-auth `Authorization` header (`edh:<token>`) via fsspec `storage_options`, never embedded in a URL (aiohttp drops the header on cross-origin redirects) |
| Local driver | Zarr, NetCDF, GeoTIFF/COG |
| Output file name | `{target_dir}/{driver}_{dataset-slug}_{geohash}_{start}_{end}_{hash8}.nc` (`static` instead of dates; `hash8` from dataset, variables, options and back-end URL) |
| Tooling | uv, Python ≥3.11 (3.12 pinned), src layout, pytest, ruff. Live tests marked `live`, skipped by default |

## 2. Layout

```
cubesip.example.yaml     config template (copy to cubesip.yaml, git-ignored)
examples/                request JSON examples (S2 NDVI, Terrascope NDVI via the CDSE federation,
                         Landsat via STAC, ERA5, local static) and example scripts
schemas/                 request.schema.json (generated, test keeps it in sync)
scripts/export_schema.py regenerates the schema
src/cubesip/
  __init__.py            public API: Request, check, run, load_config, geohash
  request.py             Request / TimeRange model
  config.py              YAML config model + lookup
  geohash.py             bbox, encode, containing, ancestors, covers
  spatial.py             dims/CRS detection, bbox transform, subset
  manifest.py            CSV read/append/lock, cache lookup
  output.py              file naming, NetCDF writing/reading, provenance attrs
  checker.py             CheckReport + check()
  api.py                 check() / run() orchestration, Result
  processing.py          containing_geohash (WKT -> cell), regrid (results onto a reference grid),
                         mask_to_polygon (cells outside a WKT polygon -> NaN or a fill value)
  drivers/base.py        Driver ABC, XarraySourceDriver (shared by local + edh)
  drivers/local.py       LocalDriver
  drivers/edh.py         EarthDataHubDriver
  drivers/openeo.py      OpenEOBackendDriver (shared openEO machinery), OpenEODriver,
                         CDSEFederationDriver (load_collection)
  drivers/stac.py        StacDriver (load_stac on the back-end named by stac.backend)
  drivers/tessera.py     TesseraDriver (TESSERA embeddings via geotessera, optional extra)
  drivers/beeldmateriaal.py  BeeldmateriaalDriver (Dutch 8 cm orthophoto GeoTIFF tiles)
tests/                   offline suite; test_live.py runs only with -m live
```

## 3. Components

### Request (JSON)

```json
{
  "driver": "openeo",
  "dataset": "SENTINEL2_L2A",
  "variables": ["B04", "B08", "SCL"],
  "geohash": "qem7k2",
  "time": {"start": "2024-01-01", "end": "2024-12-31"},
  "target_dir": "./data",
  "options": {"max_cloud_cover": 60}
}
```

`options` are driver-specific: openeo / cdse_federation / stac `max_cloud_cover` (mapped to an
`eo:cloud_cover` property filter for `stac`) and `properties`; none yet for edh/local.

### Config (YAML)

See `cubesip.example.yaml`. Lookup: explicit path -> `$CUBESIP_CONFIG` -> `./cubesip.yaml` ->
`~/.config/cubesip/cubesip.yaml`. Secrets are `SecretStr` (hidden in reprs).

### Manifest (CSV)

Columns (`manifest.COLUMNS`): `request_id, timestamp_utc, user, hostname, driver, dataset,
backend, variable, geohash, time_start, time_end, options, crs, file_size_bytes, uri,
cubesip_version`. `backend` is `Driver.backend_url` (openEO back-end or EDH URL, empty for
local); an older manifest is rewritten with the new header before the next append.
`user` = `getpass.getuser()`, `hostname` = `socket.gethostname()`, `uri` = `file:///...`.

### Checker / cache rules

`check()` runs, in order:

1. **Cache lookup** in the manifest. A file covers the request when, for the rows sharing a `uri`:
   - same `driver`, `dataset` and canonical `options` JSON;
   - same `backend` (rows with an empty `backend`, written before it existed, match any);
   - row `geohash` is the request geohash or one of its ancestors (prefix rule);
   - time: both null, or `time_start <= start` and `time_end >= end`;
   - every requested variable is present;
   - the file still exists on disk.
   Pick the longest matching geohash, then the most recent. Partial coverage (e.g. half the
   time range) counts as a miss: the whole request is fetched. Mosaicking child cells is not
   supported.
2. **Dataset check** (only on a miss): dataset exists on the source and requested variables are
   available (`DatasetNotFoundError` / `missing_variables`).
3. **Size estimate**: exact for edh/local (`subset.nbytes`); for openEO an estimate (see M5).
   Warnings: size above `max_sync_bytes`, unknown size, missing variables.

`CheckReport.ok` tells whether `run()` can proceed.

### run()

- Cache hit (and not `force`): open cached NetCDF lazily, `spatial.subset()` to the request box
  (transformed to the file's CRS) / days / variables. Return `Result(path=cached, dataset=view,
  from_cache=True, manifest_rows=[])`.
- Miss: `driver.fetch(request, out_path)` -> NetCDF -> manifest append (one row per variable,
  shared `request_id`) -> `Result(path=new, dataset=lazy open, from_cache=False, rows)`.
- Write to a temp name and rename on success so a failed download never leaves a half file
  that the manifest could point to.

## 4. Milestones

Each milestone ends with green `uv run pytest` and `uv run ruff check .`.

**M1 Core (offline)**
- `geohash`: `bbox` (decode), `ancestors`, `covers`. Tests against known values
  (e.g. `ezs42` from the Wikipedia example).
- `spatial`: `find_dims` (CF `axis`/`standard_name`, then names incl. `valid_time`),
  `native_crs` (rioxarray `rio.crs`, CF `grid_mapping`, else EPSG:4326 for lat/lon),
  `to_native_bounds` (pyproj `Transformer.transform_bounds` with densification),
  `subset` (0..360 longitudes, descending latitude, half-cell growth, inclusive end day).
  Tests on synthetic datasets: lat/lon ascending + descending, 0..360, UTM, static (no time).
- `manifest`: `read`, `append` with `filelock`, header on first write; `summarize`
  (counts, sizes, period, per-dataset breakdown), `find_missing` and `prune` for files
  that are no longer on disk.

**M2 Checker + API (offline)**
- `Manifest.find_covering` with the rules above.
- `checker.check`, `api.check`, `api.run` (temp-file + rename, manifest append, lazy result).
- Tests with a fake in-memory driver: miss -> fetch -> rows; repeat -> cache hit on same
  geohash; child geohash / shorter time -> hit on parent file; different options -> miss;
  deleted file -> miss; `force=True` -> fetch.

**M3 Local driver**
- Zarr / NetCDF / GeoTIFF open, `describe`, exact size, `fetch` = subset + `to_netcdf`.
- End-to-end test fully offline: synthetic Zarr -> `run` -> NetCDF + manifest -> second `run`
  served from cache.

**M4 Earth Data Hub driver** (done; verified live)
- Open `{url}/{dataset}.zarr` with `engine="zarr", chunks={}`, token as an `Authorization`
  header via `storage_options` (verified with zarr 3 + fsspec against a local auth server,
  zarr v2 and v3 stores).
- `describe` = variables in the store; size exact; `fetch` = subset + `to_netcdf` (dask).
- Live test (`-m live`): ERA5 `t2m` for `u0nd9`, two days.
- Docs: https://earthdatahub.destine.eu/getting-started and the ERA5 dataset page
  (dims `valid_time`, `latitude`, `longitude`; hourly; 0.25°).

**M5 openEO driver** (done; verified live for collections and `load_stac`)
- `connect()`: `openeo.connect(url)` + auth method from YAML; login only when fetching.
- Collections: `describe_collection(id)` for existence, band names (`cube:dimensions` /
  `summaries.eo:bands`), resolution and native CRS.
- STAC URL: GET the collection JSON for existence and asset/band names; `load_stac`.
- `fetch`: `load_collection` / `load_stac` with `spatial_extent={west,south,east,north,crs:4326}`,
  `temporal_extent=[start, end+1d]`, `bands`, `max_cloud_cover` / `properties`;
  `cube.download(out_path, format="netCDF")`.
- Size estimate: pixels (box in native CRS / resolution) × bands × 4 bytes × time steps.
  Time steps = distinct acquisition dates from a STAC item search (pystac-client), filtered by
  `max_cloud_cover`: the collection itself for `load_stac`; for openEO collections the STAC
  collection linked as `derived_from` (CDSE: `sentinel-2-l2a`). Without one, an upper bound
  assuming daily steps.
- Live test: SENTINEL2_L2A two weeks for a level-6 geohash, then a child geohash from cache.

**M1-M3** done as described above.

**M6 Docs** (README usage, request fields, cache rules done)
- README walkthrough of the NDVI use case (request -> check -> run -> NDVI with xarray).

**M7 Processing** (done, offline tests)
- `processing.containing_geohash(wkt)`: smallest cell holding the geometry's bounding box
  (edges included), via `geohash.containing`; shapely reads the WKT (EPSG:4326, lon first).
- `processing.regrid(sources, reference, method=...)`: rasterio `reproject` (GDAL warp) onto
  the reference's CRS and cells, lazily per dask block over the non-spatial dims; float
  output, NaN where the source has no data. A single-cell source axis (ERA5 for a small
  geohash) has no cell size, so it is stretched over the reference.
- `processing.mask_to_polygon(source, wkt, fill=nan, all_touched=False)`: the polygon is
  densified in lon/lat, projected to the result's CRS and burnt north-up with rasterio
  `geometry_mask` (cell centre inside, or every touched cell); the mask is put back in the
  grid's row/column order and applied with `where`, lazily. Integer bands become floats
  only for a NaN fill.

## 5. Risks and things to verify during implementation

1. ~~**Planetary Computer through CDSE `load_stac`**~~ Verified: the back-end reads PC's signed
   assets, and `load_stac` band names are the STAC asset keys (`red`, `nir08`).
2. **openEO synchronous limits**: large requests time out. Batch jobs are out of scope, so the
   checker's size warning is the safeguard; tune `max_sync_bytes` after the first live runs.
3. ~~**openEO NetCDF CRS metadata**~~ Verified: CDSE output has a `crs` grid-mapping variable
   that `spatial.native_crs` reads (EPSG:32632), and cache reuse works on it.
7. **Scaled band values**: data arrive as stored integers (e.g. Landsat C2 L2
   `DN * 2.75e-5 - 0.2`). Documented in the README; applying `raster:bands` scale/offset
   automatically is a possible follow-up.
4. **Manifest on a synced folder**: the repo lives under OneDrive. A lock file protects
   concurrent writes on one machine/network share, but OneDrive sync between machines can still
   create conflict copies. Keep the shared manifest on a network share or a non-synced folder.
5. **Virtualenv under OneDrive**: `uv sync` creates `.venv` here and OneDrive will try to sync
   thousands of files. Set `UV_PROJECT_ENVIRONMENT` to a path outside OneDrive.
8. ~~**Back-end not in the cache key**~~ Fixed (2026-11-19): the manifest has a `backend`
   column (`Driver.backend_url`) and a cached file is reused only for the same back-end.
   Rows from before have it empty and still match any back-end (`check()` warns).
6. **Cache key on options**: options are compared as canonical JSON; semantically equal but
   differently written options (e.g. `60` vs `60.0`) miss the cache. Acceptable for now.
