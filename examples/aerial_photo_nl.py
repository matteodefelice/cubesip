# %%
# cubesip aerial photo example: retrieve the 8 cm spring orthophoto of a site in the
# Netherlands (RGB and near-infrared, Beeldmateriaal Nederland / PDOK Luchtfoto) and show
# it as true colour, false colour and a vegetation index.
#
# The file is split into "# %%" cells: run them one at a time in VS Code (or any editor
# that understands the cell markers), or run the whole file from the repository root:
#
#     uv run --group examples python examples/aerial_photo_nl.py
#
# No account is needed. The first run downloads two tile indexes (~40 MB each, kept in
# data/_index/) and about 10 MB of imagery.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cubesip

# cubesip.yaml and data/ are in the repository root; editors often start the cells in
# the file's own folder, so move up from examples/.
if Path.cwd().name == "examples":
    os.chdir(Path.cwd().parent)

# %%
# The site is addressed by a geohash. At 8 cm, precision 7 (about 150 x 150 m, ~2 million
# pixels) is a good size; precision 6 is already ~110 MB per band.
lat, lon = 52.0560, 4.8000  # Groene Hart polder near Bodegraven: meadows and ditches
site = cubesip.geohash.encode(lat, lon, precision=7)
print(site, cubesip.geohash.bbox(site))

# %%
# The request: the high-resolution orthophoto, all four bands, the 2025 spring flight
# (time is given as whole calendar years: there is one flight per year). "ortho_hr" is the
# 8 cm spring mosaic (RGB from 2021, near-infrared from 2025); "ortho_lr" is the 25 cm
# summer mosaic (RGB and near-infrared from 2022), e.g. for near-infrared in 2024.
request = {
    "driver": "beeldmateriaal",
    "dataset": "ortho_hr",
    "variables": ["red", "green", "blue", "nir"],
    "geohash": site,
    "time": {"start": "2025-01-01", "end": "2025-12-31"},
    "target_dir": "data",
}
print(cubesip.check(request))

# %%
# Retrieve. The first run reads the tile windows over HTTP and writes a NetCDF to data/;
# later runs read it back from the cache.
result = cubesip.run(request)
print(result.path, "(from cache)" if result.from_cache else "(downloaded)")

ds = result.dataset.isel(time=0).load()
print(ds)

# %%
# Values are 8-bit digital numbers (0-255), in RD New (EPSG:28992) at 8 cm. Stack three
# bands into an image: red-green-blue for true colour, and nir-red-green for the classic
# false colour, where vegetation shows up red.
true_colour = np.dstack([ds["red"], ds["green"], ds["blue"]])
false_colour = np.dstack([ds["nir"], ds["red"], ds["green"]])

# A simple NDVI from the digital numbers: a relative index (they are not reflectances),
# fine for telling vegetation from streets and roofs within one image.
nir = ds["nir"].astype("float32")
red = ds["red"].astype("float32")
ndvi = (nir - red) / (nir + red).where(nir + red > 0)
print("NDVI quantiles:", ndvi.quantile([0.05, 0.5, 0.95]).round(2).values)

# %%
# Plot the three views side by side, in RD metres.
extent = [
    float(ds.x.min()), float(ds.x.max()), float(ds.y.min()), float(ds.y.max())
]  # fmt: skip
fig, axes = plt.subplots(1, 3, figsize=(12, 6.5), sharex=True, sharey=True, layout="constrained")
axes[0].imshow(true_colour, extent=extent)
axes[0].set_title("True colour (RGB)")
axes[1].imshow(false_colour, extent=extent)
axes[1].set_title("False colour (NIR, red, green)")
im = axes[2].imshow(ndvi, extent=extent, cmap="RdYlGn", vmin=-0.3, vmax=0.5)
axes[2].set_title("NDVI (from digital numbers)")
fig.colorbar(im, ax=axes, shrink=0.6, label="NDVI")
for ax in axes:
    ax.set_xlabel("x (RD, m)")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(axis="x", labelrotation=30)
axes[0].set_ylabel("y (RD, m)")
fig.suptitle(f"Beeldmateriaal 8 cm orthophoto, spring 2025, geohash {site}")
fig.savefig(Path("data") / f"aerial_photo_{site}.png", dpi=120)
plt.show()
