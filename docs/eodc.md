# EODC openEO driver: findings and plan

Status (2026-11-22): **not decided**. The investigation below says an EODC driver looks
feasible and small, but it was never tested against the live service: `openeo.eodc.eu` (and
`openeo.cloud`, also hosted by EODC) timed out on every attempt, while `eodc.eu` and
`stac.eodc.eu` answered. Access is probably paid.

[`scripts/probe_eodc.py`](../scripts/probe_eodc.py) checks everything listed under "Still to
confirm" once the API is reachable.

## Why EODC

The goal was global soil moisture. EODC's catalogue turned out to be mostly Sentinel-1
backscatter, Austria-specific products, ERA5-Land, CLMS and GFM, with no Sentinel-2 L2A, and
its soil moisture product (`CGLS_SSM_1KM`) covers Europe only. Global soil moisture is
already available through drivers cubesip has (`cubesip.check` passed for a site in Kenya,
2026-11-21):

| Source | Driver | Dataset | Variables | Notes |
|---|---|---|---|---|
| CLMS Soil Water Index | `openeo` (CDSE, free) | `CLMS_SWI_GLOBAL_12_5KM_DAILY_V4` | `swi001` ... `swi100`, `qflag...` | ASCAT, 12.5 km, daily, 2007-now, index 0-100 |
| ERA5-Land | `edh` | `era5/reanalysis-era5-land-no-antartica-v0` | `swvl1` ... `swvl4` | reanalysis, 0.1 deg, hourly, m3/m3, four layers |

NASA SMAP would need a new driver (Harmony subsetting service, Earthdata login).

## How the evidence was gathered

1. **Archived capabilities** (Wayback Machine, 2025-01-20, API 1.1.0, back-end 2024.11.5).
2. **Back-end source code**: [eodcgmbh/openeo-argoworkflows](https://github.com/eodcgmbh/openeo-argoworkflows)
   (on openeo-fastapi), last pushed 2026-07.
3. **Public STAC** at `https://stac.eodc.eu/api/v1`, read with cubesip's own metadata helpers.

## Findings

| Topic | Finding | Impact on cubesip |
|---|---|---|
| Synchronous processing | `POST /result` listed (2025) | `cube.download()` works; no batch jobs needed |
| How sync runs | Queues a full Argo workflow and keeps the HTTP request open, checking every 15 s; on failure a generic 500 "Submit as batch job to view logs" | Long requests may hit proxy timeouts; poor error messages |
| Output format | `netCDF` is the only output format; `save_result` always writes one `.nc` | Fits; the multi-file tar response never happens |
| CRS in output | `save_result` strips `grid_mapping` and the scalar `spatial_ref`, writes the CRS only as global attribute `crs` | `spatial.native_crs` fails on projected cubes (e.g. Equi7Grid); needs a fallback to `ds.attrs["crs"]` |
| Filters | `properties` (and so `max_cloud_cover`) passed only to the DEDL `load_stac` loader; the odc fallback ignores it silently | Unknown which loader production uses; test with `--properties` |
| Login | OIDC (EGI Check-in or Keycloak), roles read from the token (e.g. `vo.openeo.cloud` `early_adopter`) | `oidc` auth method; access needs group membership |
| Billing | One plan, `eodc-cloud`, `paid: true`, in credits | Probably paid; ask EODC (or ESA Network of Resources sponsorship) |
| Collections | Proxied from the STAC API; no `derived_from` link | Date count must search `stac.eodc.eu` with the same id (verified: item search works) |
| Metadata | `SENTINEL1_SIG0_20M`: bands VV/VH/HH/HV and 20 m found by the existing helpers | Band check and size estimate work |

## Plan if it goes ahead

1. **Reach the API**: try another network; if it still fails, ask EODC support whether
   `openeo.eodc.eu` is in production and at which URL.
2. **Get access**: EGI Check-in account, `vo.openeo.cloud` membership, credits or
   sponsorship. Ask EODC what it costs.
3. **Probe**:
   ```bash
   uv run python scripts/probe_eodc.py
   uv run python scripts/probe_eodc.py --download --properties '{"sat:orbit_state": {"eq": "descending"}}'
   ```
   Then other collection kinds (`--collection era5-land-datacube`; a static one like
   `COP_DEM`, which the probe does not fully cover since it always sends a time range) and a
   large request (`--geohash u2ed --start 2024-01-01 --end 2024-12-31`) to find the
   synchronous limit for `max_sync_bytes`.
4. **End to end without a driver**: set `openeo.url: https://openeo.eodc.eu` in a copy of
   `cubesip.yaml` and run `cubesip.check` / `cubesip.run`. Expected: dates "unknown, assumed
   daily" (no `derived_from`) and a CRS failure on `run`. Anything else is a new problem.
5. **Driver** (small, like `CDSEFederationDriver`):
   - `EODCDriver(OpenEODriver)`, `name = "eodc"`, own `eodc` config section
     (`url: https://openeo.eodc.eu`, `auth.provider_id` from the probe), registry entry,
     `DriverName` literal, `_drop_empty` loop, docs table, example YAML;
   - `search_target` returning `("https://stac.eodc.eu/api/v1", collection_id)`;
   - `spatial.native_crs` fallback to a global `crs` attribute (benefits any source);
   - drop `max_cloud_cover` / `properties` from `supported_options` if step 3 shows they are
     ignored.

If the synchronous download fails or times out even on small requests, the driver needs
batch-job support instead (`create_job` / `start_and_wait` / download results), which is a
bigger change.
