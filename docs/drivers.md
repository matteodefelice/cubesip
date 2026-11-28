# Drivers

A request names one driver. The driver decides what `dataset` means and where the data come
from. To add a new source, see [writing-a-driver.md](writing-a-driver.md).

| Driver | `dataset` | Data come from | Config section |
|---|---|---|---|
| `openeo` | openEO collection id, e.g. `SENTINEL2_L2A` | the openEO back-end at `openeo.url`: any back-end listed on the [openEO Hub](https://hub.openeo.org); default Copernicus Data Space Ecosystem (CDSE) | `openeo` |
| `cdse_federation` | openEO collection id, e.g. `TERRASCOPE_S2_NDVI_V2`, `AGERA5` | the [CDSE openEO Federation](https://documentation.dataspace.copernicus.eu/APIs/openEO/federation/openeo_federation.html) (`openeofed.dataspace.copernicus.eu`), which routes each request to CDSE or Terrascope, so it has the CDSE collections plus Terrascope's | `cdse_federation` |
| `stac` | STAC collection URL, e.g. Planetary Computer `.../collections/landsat-c2-l2` | any STAC catalogue, read by openEO `load_stac` on the back-end named in `stac.backend` (`openeo` or `cdse_federation`) | `stac` |
| `edh` | Earth Data Hub dataset path, e.g. `era5/era5-single-levels-atmosphere-v0` | Earth Data Hub Zarr, streamed with xarray | `earthdatahub` |
| `tessera` | TESSERA store version, e.g. `v1.1` (or a store URL) | [TESSERA](https://github.com/ucam-eo/tessera) embeddings, read with [geotessera](https://github.com/ucam-eo/geotessera)'s Zarr API from the public Source Cooperative stores | none |
| `beeldmateriaal` | `ortho_hr` (8 cm, spring) or `ortho_lr` (25 cm, summer) | [Beeldmateriaal Nederland](https://www.beeldmateriaal.nl/dataroom) aerial orthophotos of the Netherlands (RGB and near-infrared), the open-data GeoTIFF tiles behind PDOK Luchtfoto RGB / CIR, read window by window over HTTP | none |
| `url` | `http(s)` URL of a GeoTIFF/COG, NetCDF or Zarr store, or `<zip URL>!<GeoTIFF inside the zip>` | one file on a web server, read with HTTP range requests (GDAL `/vsicurl/` and `/vsizip/`, netCDF-C byte-range mode, Zarr over HTTP); meant for global hazard and exposure maps such as those in the [Climate Risk STAC](https://climate-risk-data.github.io/climate-risk-stac/) | none |
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

## Beeldmateriaal orthophotos (PDOK Luchtfoto)

Beeldmateriaal Nederland flies the whole of the Netherlands twice a year and publishes the
orthophoto mosaics as open data (CC BY 4.0), as GeoTIFF tiles in RD New (EPSG:28992). Each
mosaic comes in two products: RGB and colour-infrared (CIR, bands near-infrared, red, green).
The request's `dataset` picks the mosaic, and with it the resolution and the season:

| `dataset` | Resolution | Flight | Tiles | RGB | Near-infrared (CIR) |
|---|---|---|---|---|---|
| `ortho_hr` | 8 cm (5 cm in some areas) | spring, leafless | 1 x 1 km | 2021 onwards | 2025 onwards |
| `ortho_lr` | 25 cm | summer, leaves on | 4 x 4 km | 2022 onwards | 2022 onwards |

PDOK's *Luchtfoto RGB* and *Luchtfoto CIR* WMS/WMTS services show the same imagery, but they
only serve JPEG renderings, at most 2500 x 2500 px per request. The `beeldmateriaal` driver
reads the GeoTIFF tiles themselves, and only the part of each tile that the geohash covers.
No account is needed.

```python
result = cubesip.run(
    {
        "driver": "beeldmateriaal",
        "dataset": "ortho_hr",  # 8 cm; "ortho_lr" for 25 cm
        "variables": ["red", "nir"],  # any of red, green, blue, nir
        "geohash": "u178kdc",  # Utrecht, ~150 x 150 m
        "time": {"start": "2025-01-01", "end": "2025-12-31"},  # whole calendar years
        "options": {"resolution": 0.08},  # optional, metres; default the mosaic's own
    }
)
ds = result.dataset  # uint8 (time, y, x) per variable, RD New, 8 cm
ndvi = (ds["nir"].astype("float32") - ds["red"]) / (ds["nir"].astype("float32") + ds["red"])
```

- `variables`: `red`, `green`, `blue` come from the RGB tiles, `nir` from the CIR tiles. Red
  and green are in both, so `["nir", "red"]` reads only the CIR tiles. Values are 8-bit
  digital numbers as published (JPEG-compressed at quality 95), not reflectances, so an
  NDVI computed from them is a relative index. It is useful within one image, less so
  between years.
- `time`: whole calendar years, one step per year stamped on 1 January. A year or product
  without tiles is refused with the years that exist (see the table); for near-infrared
  before 2025, use `ortho_lr`. Older 25 cm years (2016-2020) are only published as
  nationwide ECW mosaics of 100-250 GB in the dataroom, which the driver does not read. The tiles of the
  current year are delivered over several months; `check()` warns about the share of the
  area without tiles, and those pixels are 0.
- `options.resolution`: output pixel size in metres, by default the mosaic's own (0.08 or
  0.25). It must divide 1 km into whole pixels (0.05, 0.08, 0.1, 0.25, 0.5, 1, ...). At the
  tile's own resolution, pixels are copied untouched; otherwise they are averaged (5 cm
  tiles at the default 0.08). The whole tile window is still read at full resolution, since
  the tiles have no overviews.
- Grid: RD New, with pixel edges on multiples of the resolution, so cells and years line up
  exactly and a child geohash is cut from the cached file.
- Size: 8 cm is fine-grained. A level-7 geohash (~150 m) is about 2.3 MB per variable per
  year at 8 cm and 0.23 MB at 25 cm; a level-6 one (~1.2 x 0.6 km) about 110 MB at 8 cm and
  12 MB at 25 cm. The cube is held in memory before writing, and `check()` warns above
  2 GB. For larger areas use `ortho_lr`, a coarser `resolution` or `cubesip.split()`.
- Tiles are found through the GeoJSON indexes that the [download
  portal](https://basisdata.nl/hwh-portal/download/index.html) loads, one per mosaic, year
  and product. They list the same tile URLs as the portal's download-as-text option (e.g.
  `.../hwh-ortho/2024/Ortho/1/09/beelden_tif_tegels/2024_122000_517000_RGB_hrl.tif`). The
  8 cm indexes are 20-55 MB, the 25 cm ones about 2 MB. Each is downloaded once and kept as
  a small CSV in `<data_dir>/_index/`; those of the last two years are refreshed after a
  week. About 2% of the 8 cm tiles are listed twice (two neighbouring flight blocks); the
  most recently modified file is used. The URLs of the tiles read are stored in the NetCDF
  attribute `source_tiles`.

## Files on a web server: `url`

The `url` driver reads one file that a server publishes as is, without downloading it whole.
It is the `local` driver for files on the web: a GeoTIFF becomes one variable per band, a
NetCDF or Zarr store keeps its variable names, and only the pixels of the geohash cell are
transferred. No account is needed; only public files can be read.

It was written for the global hazard and exposure maps that the [Climate Risk
STAC](https://climate-risk-data.github.io/climate-risk-stac/) lists. That catalogue
describes datasets but does not serve them: it has no search API, and its items point to
whole-world files, zip archives or landing pages. So the `url` driver reads the file an item
points to, not the catalogue itself. The catalogue is still the place to find the file and
read the dataset's licence.

| `dataset` written as | Read with | Example |
|---|---|---|
| `https://.../map.tif` | GDAL `/vsicurl/` | a Cloud Optimized GeoTIFF |
| `https://.../maps.zip!folder/map.tif` | GDAL `/vsizip/` over `/vsicurl/` | JRC flood depth, STORM wind speeds |
| `https://.../data.nc` | netCDF-C byte-range mode (`#mode=bytes`) | SPEIbase |
| `https://.../store.zarr` | xarray + zarr over HTTP | any public Zarr |

Two global hazard maps, both checked live on 2026-11-28:

```python
# JRC river flood depth (m) for the 100-year return period, 30 arc seconds (~1 km)
jrc = cubesip.run(
    {
        "driver": "url",
        "dataset": "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps/"
        "floodMapGL_rp100y.zip!floodMapGL_rp100y.tif",
        "variables": ["band_1"],
        "geohash": "u173",  # around Amsterdam
        "time": None,
    }
)

# STORM 100-year tropical cyclone wind speed (m/s), western Pacific, present climate, 0.1 deg.
# The archive URL on 4TU.ResearchData has no extension; the "!" still names the member.
storm = cubesip.run(
    {
        "driver": "url",
        "dataset": "https://data.4tu.nl/file/0ea98bdd-5772-4da8-ae97-99735e891aff/"
        "0b98c6f5-c7af-45bb-bb6e-dac53e1b8d55!STORM_FIXED_RETURN_PERIODS_WP_100_YR_RP.tif",
        "variables": ["band_1"],
        "geohash": "wdw5",  # Manila
        "time": None,
    }
)
```

- `dataset`: the format comes from the URL's extension. A URL without one (data
  repositories often have none) is recognised from its first bytes. A zip archive needs the
  member named after `!`, and only GeoTIFF members can be read. To list the members of an
  archive: `zipfile.ZipFile(fsspec.open(url).open()).namelist()` reads only its directory.
- `variables`: `band_1`, `band_2`... for GeoTIFFs whose bands have no descriptions (most
  hazard maps), otherwise the descriptions. NetCDF and Zarr use their own variable names.
- `time`: `null` for layers without a time dimension, which most hazard maps are. Each return
  period or scenario is usually a separate file, so it is a separate `dataset` and its own
  manifest row.
- No-data pixels become NaN, as with `local` GeoTIFFs. A cell where the map has no value
  (e.g. no flood) is written as all NaN, not refused.
- The server must answer HTTP range requests (most do). A member of a compressed zip is
  decompressed from its start, so its first read takes a few seconds for a large map (about
  3 s for the 37 MB JRC member).
- Coarse maps and small cells: every pixel that touches the cell is kept, so a level-7
  geohash still gets the one ~1 km pixel that contains it, and child cells are cut from the
  cached file.
- What it does not cover: datasets that are only on the Copernicus Climate Data Store (most
  of the catalogue's derived climate indices) need a queued CDS API request and are not
  supported; vector, event and table data (CEMS rapid mapping, storm tracks) do not fit
  cubesip's gridded model; files behind a login (NASA Earthdata, for GEHE) cannot be read
  and fail as not found or refused.
