# Drivers

A request names one driver. The driver decides what `dataset` means and where the data come
from.

| Driver | `dataset` | Data come from | Config section |
|---|---|---|---|
| `openeo` | openEO collection id, e.g. `SENTINEL2_L2A` | the openEO back-end at `openeo.url`: any back-end listed on the [openEO Hub](https://hub.openeo.org); default Copernicus Data Space Ecosystem (CDSE) | `openeo` |
| `cdse_federation` | openEO collection id, e.g. `TERRASCOPE_S2_NDVI_V2`, `AGERA5` | the [CDSE openEO Federation](https://documentation.dataspace.copernicus.eu/APIs/openEO/federation/openeo_federation.html) (`openeofed.dataspace.copernicus.eu`), which routes each request to CDSE or Terrascope, so it has the CDSE collections plus Terrascope's | `cdse_federation` |
| `stac` | STAC collection URL, e.g. Planetary Computer `.../collections/landsat-c2-l2` | any STAC catalogue, read by openEO `load_stac` on the back-end named in `stac.backend` (`openeo` or `cdse_federation`) | `stac` |
| `edh` | Earth Data Hub dataset path, e.g. `era5/era5-single-levels-atmosphere-v0` | Earth Data Hub Zarr, streamed with xarray | `earthdatahub` |
| `tessera` | TESSERA store version, e.g. `v1.1` (or a store URL) | [TESSERA](https://github.com/ucam-eo/tessera) embeddings, read with [geotessera](https://github.com/ucam-eo/geotessera)'s Zarr API from the public Source Cooperative stores | none |
| `local` | path to a Zarr store, NetCDF or GeoTIFF | the local file system, read with xarray | none |

## openEO-based drivers: `openeo`, `cdse_federation`, `stac`

The three openEO-based drivers log in with their section's `auth` and download
synchronously. A collection id given to `stac`, or a URL given to `openeo` /
`cdse_federation`, is refused with a message naming the right driver. The federation uses
the same CDSE account but its own login client, so the first `cdse_federation` download asks
for a device-code login once; the refresh token is then stored. When no pixel in the area
and period is valid (e.g. a single fully cloudy scene), the federation returns an empty
file; cubesip raises `EmptyResultError` and records nothing.

### Filtering scenes: `max_cloud_cover` and `properties`

Two request options select which scenes (STAC items) the back-end loads, based on their
metadata:

- `max_cloud_cover`: a number, the highest scene cloud cover (%) to keep. It filters whole
  scenes; clouds in the scenes that are kept still have to be masked pixel by pixel (with
  `SCL` for Sentinel-2, `qa_pixel` for Landsat).
- `properties`: one condition per property, written `{"<property>": {"<operator>": value}}`,
  e.g. `{"platform": {"eq": "landsat-9"}}` or `{"eo:cloud_cover": {"lte": 20}}`.

```python
{
    "driver": "stac",
    "dataset": "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2",
    "variables": ["lwir11", "qa_pixel"],
    "geohash": "u094v",
    "time": {"start": "2024-01-01", "end": "2024-12-31"},
    "target_dir": "./data",
    "options": {"max_cloud_cover": 80, "properties": {"platform": {"eq": "landsat-9"}}},
}
```

#### Which operators work

cubesip accepts six operators in `properties`: `eq`, `neq`, `lt`, `lte`, `gt` and `gte`. It
checks only that the operator is one of these, then hands the condition to the back-end,
**and the back-end decides what it supports.** On the Copernicus Data Space Ecosystem (the
default back-end) only three of the six work:

| Operator | Meaning | `openeo` driver on CDSE | `stac` driver on CDSE |
|---|---|---|---|
| `eq` | equal to | works | works |
| `lte` | at most | works | works |
| `gte` | at least | works | works |
| `neq` | not equal to | refused | refused |
| `lt` | less than | refused | refused |
| `gt` | greater than | refused | refused |

Tested on 2026-11-20 with Sentinel-2 L2A (`openeo`) and Planetary Computer Landsat C2 L2
(`stac`). Both drivers behave alike because CDSE runs `load_collection` through `load_stac`
internally. The `cdse_federation` driver and other openEO back-ends set with `openeo.url`
have not been tested and may accept a different set. The CDSE error message also lists
`array_contains`, which cubesip does not offer.

A refused operator fails **when `cubesip.run` sends the job, not before**: `cubesip.check`
reports the request as ready to download, because it does not ask the back-end about
operators. The error comes back from the back-end as:

```
openeo.rest.OpenEoApiError: [400] PropertyConditionInvalid: Property filtering only
supports ['eq', 'lte', 'gte', 'array_contains'], not 'neq'.
```

Nothing is written to disk or to the manifest when this happens.

How to get the same selection with the operators that work:

- **`lt` / `gt`** on a number: use `lte` / `gte` with a value just inside the bound, e.g.
  `{"eo:cloud_cover": {"lte": 19.99}}` instead of `{"lt": 20}`.
- **`neq`**: name the value you want with `eq` instead of the one you don't, e.g.
  `{"platform": {"eq": "landsat-9"}}` instead of `{"platform": {"neq": "landsat-8"}}`. If you
  want several values, make one request per value: each is cached separately.

Only one condition per property is allowed, so a range (`gte` and `lte` on the same
property) cannot be written as two conditions; filter on one bound and drop the rest after
loading.

## TESSERA embeddings

[TESSERA](https://github.com/ucam-eo/tessera) is a foundation model that compresses a year of
Sentinel-1 and Sentinel-2 observations into a 128-channel embedding for each 10 m pixel
(global, 2017-2025 for v1.1). The `tessera` driver reads them with geotessera; the stores are
public, so no credentials are needed.

The driver needs the optional `tessera` extra (geotessera, Python 3.12+):
`uv sync --extra tessera`, or `pip install 'cubesip[tessera]'`.

```python
result = cubesip.run(
    {
        "driver": "tessera",
        "dataset": "v1.1",  # store version; v1, v2 or a store URL also work
        "variables": ["embedding"],
        "geohash": "u094v8",
        "time": {"start": "2024-01-01", "end": "2024-12-31"},  # whole calendar years
        "target_dir": "./data",
        "options": {"bands": [0, 1, 2]},  # optional: keep these channels (zero-based)
    }
)
emb = result.dataset["embedding"]  # float32 (time, band, y, x), native UTM, 10 m
```

- `variables` is always `["embedding"]`; values are dequantised floats, NaN where there is no
  embedding (open water, outside coverage). A cell with no embedding at all raises
  `EmptyResultError`.
- `time` must be whole calendar years, one step per year stamped on 1 January: an embedding
  summarises the whole year, and a cached annual file could not serve a partial period.
- `options`: `bands` (channels to keep, in order) and `depth` (a published embedding prefix
  of v2 stores, e.g. 16).
- The grid is the UTM zone of the geohash centre (TESSERA uses the northern UTM CRS,
  EPSG:326NN, everywhere). The manifest's dataset is the resolved store URL, so `v1.1` today
  and a future default store are cached separately.
- The whole cube is read into memory before writing: about 2.6 MB per year for a level-6
  geohash, but about 2.7 GB per year for a level-4 cell at mid-latitudes; `check()` warns
  above 2 GB.
