# %%
# cubesip hazard maps example: read global climate hazard maps straight from the servers that
# publish them, with the `url` driver, and turn them into return-period curves for one site.
#
# - JRC global river flood depth, 30 arc seconds (~1 km), for return periods of 10 to 500
#   years: one zip per return period on the JRC open data server.
# - STORM tropical cyclone wind speeds, 0.1 degrees, for return periods of 10 to 10,000 years
#   in the present climate: one zip with 168 GeoTIFFs on 4TU.ResearchData.
#
# Both datasets are listed in the Climate Risk STAC (https://climate-risk-data.github.io/
# climate-risk-stac/), which says where each file is. The `url` driver then reads only the
# pixels of the geohash cell, from inside the remote zips, without downloading them whole.
#
# The file is split into "# %%" cells: run them one at a time in VS Code (or any editor that
# understands the cell markers), or run the whole file from the repository root:
#
#     uv run --group examples python examples/hazard_maps.py
#
# No account is needed. The first run takes about a minute and transfers a few hundred MB
# (each compressed zip member is read from its start); the NetCDFs written to data/ are a
# few kB, and later runs read them from the cache.

import os
import zipfile
from pathlib import Path

import fsspec
import matplotlib.pyplot as plt
import numpy as np

import cubesip

# cubesip.yaml and data/ are in the repository root; editors often start the cells in the
# file's own folder, so move up from examples/.
if Path.cwd().name == "examples":
    os.chdir(Path.cwd().parent)

JRC = "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps"
STORM_PRESENT = (
    "https://data.4tu.nl/file/0ea98bdd-5772-4da8-ae97-99735e891aff/"
    "0b98c6f5-c7af-45bb-bb6e-dac53e1b8d55"
)

# %%
# River floods. The site is the level-4 geohash cell (~25 x 20 km) between the Waal (Rhine)
# and the Maas (Meuse) around Nijmegen, in the Netherlands. At ~1 km pixels it holds about
# 40 x 20 values.
flood_site = cubesip.geohash.encode(51.80, 5.86, precision=4)
print(flood_site, cubesip.geohash.bbox(flood_site))

# One request per return period: each map is its own file, so its own dataset. The part after
# "!" names the GeoTIFF inside the zip.
flood_periods = [10, 20, 50, 100, 200, 500]
depth = {}
for rp in flood_periods:
    request = {
        "driver": "url",
        "dataset": f"{JRC}/floodMapGL_rp{rp}y.zip!floodMapGL_rp{rp}y.tif",
        "variables": ["band_1"],  # the maps have one unnamed band: water depth in metres
        "geohash": flood_site,
        "time": None,  # static maps
        "target_dir": "data",
    }
    result = cubesip.run(request)
    print(f"{rp:>4}-year: {result.path.name}", "(cache)" if result.from_cache else "")
    depth[rp] = result.dataset["band_1"].load()

# %%
# Pixels outside the flooded area are NaN (no-data in the map). Treat them as dry, then
# summarise each return period by the share of the cell that is flooded and the mean depth
# where it is.
flooded_share = {}
mean_depth = {}
for rp, d in depth.items():
    wet = d.fillna(0) > 0
    flooded_share[rp] = float(wet.mean()) * 100
    mean_depth[rp] = float(d.where(wet).mean())
    print(f"{rp:>4}-year: {flooded_share[rp]:5.1f}% flooded, mean depth {mean_depth[rp]:.2f} m")

# %%
# Tropical cyclones. STORM ships every return period and ocean basin as a separate GeoTIFF in
# one zip. Listing the archive reads only its directory at the end of the file.
with fsspec.open(STORM_PRESENT, "rb") as f:
    members = zipfile.ZipFile(f).namelist()
basin = "WP"  # western Pacific; the others are EP, NA, NI, SI and SP
storm_files = {
    int(name.split("_")[-3]): name
    for name in members
    if name.startswith(f"STORM_FIXED_RETURN_PERIODS_{basin}_")
}
storm_periods = sorted(storm_files)
print(len(members), "files in the archive;", basin, "return periods:", storm_periods)

# %%
# The site is Manila. A level-5 cell (~5 x 5 km) is smaller than one 0.1 degree pixel: cubesip
# keeps every pixel that touches the cell, so each map gives one or a few values.
storm_site = cubesip.geohash.encode(14.60, 121.00, precision=5)
wind = {}
for rp in storm_periods:
    result = cubesip.run(
        {
            "driver": "url",
            "dataset": f"{STORM_PRESENT}!{storm_files[rp]}",
            "variables": ["band_1"],  # 10-minute sustained wind speed, m/s
            "geohash": storm_site,
            "time": None,
            "target_dir": "data",
        }
    )
    wind[rp] = float(result.dataset["band_1"].max())
print({rp: round(v, 1) for rp, v in wind.items()})

# %%
# Plot: the 100-year flood map, the flood curve and the wind curve.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")

d100 = depth[100]
im = axes[0].imshow(
    d100.where(d100 > 0),
    extent=[float(d100.x.min()), float(d100.x.max()), float(d100.y.min()), float(d100.y.max())],
    cmap="Blues",
    vmin=0,
)
fig.colorbar(im, ax=axes[0], label="water depth (m)")
axes[0].set_title(f"JRC 100-year river flood depth, {flood_site}")
axes[0].set_xlabel("longitude")
axes[0].set_ylabel("latitude")

axes[1].plot(flood_periods, [flooded_share[rp] for rp in flood_periods], "o-", label="flooded")
axes[1].set_xscale("log")
axes[1].set_xlabel("return period (years)")
axes[1].set_ylabel("share of the cell flooded (%)")
twin = axes[1].twinx()
twin.plot(flood_periods, [mean_depth[rp] for rp in flood_periods], "s--", color="tab:orange")
twin.set_ylabel("mean depth where flooded (m)", color="tab:orange")
axes[1].set_title("River flood curve (JRC)")

axes[2].plot(storm_periods, [wind[rp] for rp in storm_periods], "o-", color="tab:red")
axes[2].set_xscale("log")
axes[2].set_xlabel("return period (years)")
axes[2].set_ylabel("wind speed (m/s)")
axes[2].set_title(f"Cyclone wind, Manila ({storm_site}), STORM {basin}")
for ax in axes[1:]:
    ax.grid(True, which="both", alpha=0.3)

out = Path("data") / f"hazard_maps_{flood_site}_{storm_site}.png"
fig.savefig(out, dpi=120)
print("saved", out)
plt.show()

# %%
# The flooded share rises with the return period; that is what an exposure layer (buildings,
# population, crops) would then be overlaid on. Note what the maps are: the JRC maps cover
# river floods only (no coastal or rain-on-the-spot flooding) and ignore flood defences, so
# for a diked delta like this one they show where water would go, not how often it does.
print(
    "max depth per return period (m):",
    {rp: round(float(np.nanmax(d)), 2) for rp, d in depth.items()},
)
