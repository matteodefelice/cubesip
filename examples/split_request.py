# %%
# cubesip split: a request too big for one openEO download (about 570 MB, over the 500 MB
# limit) is split into 32 smaller requests, one per child geohash cell. Each part is run and
# cached on its own, reduced to a summer median NDVI map, and the maps are joined back together.
#
# The file is split into "# %%" cells: run them one at a time in VS Code (or any editor
# that understands the cell markers), or run the whole file from the repository root:
#
#     uv run --group examples python examples/split_request.py
#
# Needs a Copernicus Data Space account (openeo section of cubesip.yaml). The first run makes
# 32 openEO downloads; later runs read them back from the cache.

import os
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt

import cubesip

# cubesip.yaml and data/ are in the repository root; editors often start the cells in
# the file's own folder, so move up from examples/.
if Path.cwd().name == "examples":
    os.chdir(Path.cwd().parent)

# %%
# Precision 4 is a cell of about 26 x 20 km here: at 10 m that is 2646 x 2002 pixels per date.
lat, lon = 48.30, 1.65  # cropland in the Beauce plain, France
area = cubesip.geohash.encode(lat, lon, precision=4)
print(area, cubesip.geohash.bbox(area))

# %%
# Two summer months of Sentinel-2 red, near infrared and scene classification. check() warns
# that the estimated size is over the openEO limit for one download.
request = cubesip.Request.from_json(
    {
        "driver": "openeo",
        "dataset": "SENTINEL2_L2A",
        "variables": ["B04", "B08", "SCL"],
        "geohash": area,
        "time": {"start": "2024-07-01", "end": "2024-08-31"},
        "target_dir": "data",
        "options": {"max_cloud_cover": 30},
    }
)
print(cubesip.check(request))

# %%
# split() gives 32 copies of the request, identical except for the geohash: one character
# longer, so each part is about 1/32 of the area (and of the size).
parts = cubesip.split(request)
print([p.geohash for p in parts])
print(cubesip.check(parts[0]))

# %%
# Run the parts one after the other. Each one is an ordinary request: it is written to its own
# NetCDF and to the manifest, so a rerun (or a run that stopped halfway) only downloads what
# is missing. Each part is reduced right away to a 2D map, to keep memory low.
CLEAR_SCL_CLASSES = [4, 5, 6]  # vegetation, not vegetated, water: no clouds, shadows or snow

maps = []
for i, part in enumerate(parts, 1):
    result = cubesip.run(part)
    print(f"{i:2d}/32 {part.geohash}", "(from cache)" if result.from_cache else "(downloaded)")
    ds = result.dataset
    red, nir = ds["B04"], ds["B08"]
    clear = ds["SCL"].isin(CLEAR_SCL_CLASSES) & ((nir + red) > 0)
    ndvi = ((nir - red) / (nir + red)).where(clear)
    maps.append(ndvi.median("t").compute())
    ds.close()

# %%
# Join the 32 maps into one. Neighbouring parts can share a row or column of pixels at their
# edges (with the same values), so combine_first is used rather than concat: it takes the
# union of the coordinates and fills each map's gaps from the next one.
ndvi_map = reduce(lambda a, b: a.combine_first(b), maps)
ndvi_map = ndvi_map.sortby("y", ascending=False)  # north up, as in the source files
print(ndvi_map.sizes)

# %%
# Plot the summer median NDVI of the whole cell.
ndvi_map.plot(cmap="YlGn", vmin=0, vmax=1, figsize=(9, 7), cbar_kwargs={"label": "NDVI"})
plt.title(f"Sentinel-2 median NDVI, geohash {area} (32 parts), July-August 2024")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
