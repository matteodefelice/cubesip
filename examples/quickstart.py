# %%
# cubesip quickstart: retrieve one month of ERA5 2 m temperature for a site and look at it
# with xarray.
#
# The file is split into "# %%" cells: run them one at a time in VS Code (or any editor
# that understands the cell markers), or run the whole file from the repository root:
#
#     uv run --group examples python examples/quickstart.py
#
# Needs an Earth Data Hub token in cubesip.yaml.

import os
from pathlib import Path

import matplotlib.pyplot as plt

import cubesip

# cubesip.yaml and data/ are in the repository root; editors often start the cells in
# the file's own folder, so move up from examples/.
if Path.cwd().name == "examples":
    os.chdir(Path.cwd().parent)

# %%
# The site is addressed by a geohash. Precision 6 is a cell of about 1.2 x 0.6 km.
lat, lon = 48.30, 1.65  # cropland in the Beauce plain, France
site = cubesip.geohash.encode(lat, lon, precision=6)
print(site, cubesip.geohash.bbox(site))

# %%
# The request: driver and dataset, variables, where and when (both days included).
# check() says whether it is already cached and how big it is, without downloading.
request = {
    "driver": "edh",
    "dataset": "era5/era5-single-levels-atmosphere-v0",
    "variables": ["t2m"],
    "geohash": site,
    "time": {"start": "2024-07-01", "end": "2024-07-31"},
    "target_dir": "data",
}
print(cubesip.check(request))

# %%
# Retrieve. The first run downloads and writes a NetCDF to data/; later runs read it back
# from the cache.
result = cubesip.run(request)
print(result.path, "(from cache)" if result.from_cache else "(downloaded)")

ds = result.dataset
print(ds)

# %%
# ERA5 is on a 0.25 deg grid, so the geohash cell falls in a single grid point: squeeze
# drops the size-1 latitude/longitude dimensions and leaves an hourly time series.
t2m = ds["t2m"].squeeze() - 273.15
t2m.attrs = {"long_name": "2 m temperature", "units": "°C"}  # replaces the GRIB ones (K)
print(t2m.to_series().describe().round(1))

# %%
# Plot the hourly series with xarray.
t2m.plot(color="#2a78d6", linewidth=1, figsize=(9, 4))
plt.title(f"ERA5 2 m temperature, geohash {site}, July 2024")
plt.grid(axis="y", color="#e1e0d9")
plt.tight_layout()
plt.show()
