"""Yearly mean NDVI for a site, retrieved with cubesip.

For one geohash cell the script:

1. requests Sentinel-2 L2A red (B04), near infrared (B08) and the scene classification
   (SCL) from openEO, one cubesip request per year. Each year is written to the manifest,
   so running the script again reads the files from disk instead of downloading;
2. keeps only clear pixels (SCL vegetation, not-vegetated or water), dropping clouds,
   cloud shadows and snow, and computes NDVI = (B08 - B04) / (B08 + B04);
3. averages NDVI over the cell for every scene with enough clear pixels, then per month,
   then per year (mean of the monthly means, so the many clear summer scenes do not
   outweigh the winter);
4. saves a chart (PNG) and a table (CSV) and prints the table.

CDSE stores these bands with scale 0.0001 and offset 0, so the scale cancels out and NDVI
can be computed on the downloaded values directly.

Run from the repository root, with cubesip.yaml configured:

    uv run --group examples python examples/ndvi_yearly.py
    uv run --group examples python examples/ndvi_yearly.py --lat 45.30 --lon 9.50
    uv run --group examples python examples/ndvi_yearly.py --geohash u0nd9w --years 2021 2024
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
import xarray as xr

import cubesip
from cubesip.spatial import find_dims

# Default site: cropland in the Beauce plain (France). Pass --lat/--lon or --geohash.
DEFAULT_LAT, DEFAULT_LON = 48.30, 1.65

# Sentinel-2 scene classification classes counted as clear:
# 4 vegetation, 5 not vegetated, 6 water.
CLEAR_SCL_CLASSES = [4, 5, 6]

# Chart colours (light theme): one accent for the yearly mean, muted grey for context.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--geohash", help="geohash of the site, e.g. u0nd9w")
    where.add_argument("--lat", type=float, help="latitude of the site (use with --lon)")
    parser.add_argument("--lon", type=float, help="longitude of the site")
    parser.add_argument(
        "--precision", type=int, default=6, help="geohash length for --lat/--lon (6 = ~1 km)"
    )
    parser.add_argument(
        "--years", type=int, nargs=2, default=[2019, 2024], metavar=("FIRST", "LAST")
    )
    parser.add_argument(
        "--max-cloud-cover", type=float, default=80, help="skip scenes cloudier than this (%%)"
    )
    parser.add_argument(
        "--min-clear",
        type=float,
        default=0.5,
        help="minimum fraction of clear pixels for a scene to count",
    )
    parser.add_argument("--target-dir", default="data", help="where NetCDF files are stored")
    parser.add_argument("--output-dir", default="data", help="where the PNG and CSV are saved")
    parser.add_argument("--config", help="cubesip config file (default: ./cubesip.yaml)")
    parser.add_argument("--show", action="store_true", help="also open the chart in a window")
    args = parser.parse_args()
    if args.lat is not None and args.lon is None:
        parser.error("--lat needs --lon")
    if args.geohash is None:
        lat = DEFAULT_LAT if args.lat is None else args.lat
        lon = DEFAULT_LON if args.lon is None else args.lon
        args.geohash = cubesip.geohash.encode(lat, lon, args.precision)
    return args


def fetch_year(year: int, args: argparse.Namespace) -> xr.Dataset:
    """Sentinel-2 B04/B08/SCL for one calendar year (downloaded or read from the cache)."""
    request = cubesip.Request.from_json(
        {
            "driver": "openeo",
            "dataset": "SENTINEL2_L2A",
            "variables": ["B04", "B08", "SCL"],
            "geohash": args.geohash,
            "time": {"start": f"{year}-01-01", "end": f"{year}-12-31"},
            "target_dir": args.target_dir,
            "options": {"max_cloud_cover": args.max_cloud_cover},
        }
    )
    report = cubesip.check(request, args.config)
    if not report.ok:
        raise SystemExit(report.summary())
    if report.cache_hit is None:
        print(f"{year}: downloading ({report.size.detail}) ...", flush=True)
    result = cubesip.run(request, args.config)
    print(f"{year}: {'from cache' if result.from_cache else 'saved'} {result.path.name}")
    return result.dataset


def scene_ndvi(ds: xr.Dataset, min_clear: float) -> pd.Series:
    """Mean NDVI over the cell for each scene with at least ``min_clear`` clear pixels."""
    dims = find_dims(ds)
    red, nir, scl = ds["B04"], ds["B08"], ds["SCL"]
    clear = scl.isin(CLEAR_SCL_CLASSES) & ((nir + red) > 0)
    ndvi = ((nir - red) / (nir + red)).where(clear)

    space = (dims.y, dims.x)
    clear_fraction = clear.mean(dim=space)
    mean_ndvi = ndvi.mean(dim=space).where(clear_fraction >= min_clear)
    series = mean_ndvi.to_series().dropna()
    series.index = pd.DatetimeIndex(series.index).normalize()
    series.index.name = "date"
    return series.groupby(level=0).mean()  # one value per day


def yearly_table(scenes: pd.Series) -> pd.DataFrame:
    """Yearly mean of monthly means, with the number of months and scenes behind it."""
    monthly = scenes.resample("MS").mean().dropna()
    by_year = monthly.groupby(monthly.index.year)
    table = pd.DataFrame(
        {
            "ndvi_mean": by_year.mean(),
            "months_with_data": by_year.size(),
            "clear_scenes": scenes.groupby(scenes.index.year).size(),
        }
    )
    table.index.name = "year"
    return table.round({"ndvi_mean": 3})


def plot(scenes: pd.Series, table: pd.DataFrame, geohash: str, path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Context: every clear scene, small and muted.
    ax.scatter(
        scenes.index,
        scenes.values,
        s=10,
        color=INK_MUTED,
        alpha=0.45,
        linewidths=0,
        label="Clear scene (cell mean)",
        zorder=2,
    )

    # The story: one point per year, placed mid-year, joined by a 2px line.
    mid_year = pd.to_datetime([f"{year}-07-01" for year in table.index])
    ax.plot(
        mid_year,
        table["ndvi_mean"],
        color=ACCENT,
        linewidth=2,
        solid_joinstyle="round",
        solid_capstyle="round",
        marker="o",
        markersize=7,
        markerfacecolor=ACCENT,
        markeredgecolor=SURFACE,  # surface ring keeps markers legible over the dots
        markeredgewidth=2,
        label="Yearly mean",
        zorder=3,
    )

    # Label only the latest year, in text ink beside its marker.
    last_year, last_value = table.index[-1], table["ndvi_mean"].iloc[-1]
    ax.annotate(
        f"{last_year}: {last_value:.2f}",
        xy=(mid_year[-1], last_value),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        color=INK_PRIMARY,
        fontsize=10,
        fontweight="semibold",
    )

    low = min(0.0, float(scenes.min()) - 0.05)
    ax.set_ylim(low, 1.0)
    ax.set_ylabel("NDVI", color=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, length=0, pad=6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    first, last = table.index[0], table.index[-1]
    ax.set_xlim(pd.Timestamp(f"{first}-01-01"), pd.Timestamp(f"{last}-12-31"))

    fig.subplots_adjust(left=0.08, right=0.95, top=0.84, bottom=0.08)
    fig.text(
        0.02,
        0.96,
        f"Mean NDVI per year, geohash {geohash}",
        va="top",
        color=INK_PRIMARY,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.905,
        "Sentinel-2 L2A, clouds masked with SCL; yearly mean of monthly means",
        va="top",
        color=INK_SECONDARY,
        fontsize=10,
    )
    legend = ax.legend(loc="lower left", frameon=False, ncols=2, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.savefig(path, facecolor=SURFACE)


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    first, last = args.years
    box = cubesip.geohash.bbox(args.geohash)
    print(
        f"Site {args.geohash}: lat {box.south:.4f}..{box.north:.4f}, "
        f"lon {box.west:.4f}..{box.east:.4f}; years {first}-{last}"
    )

    scenes = pd.concat(
        [scene_ndvi(fetch_year(year, args), args.min_clear) for year in range(first, last + 1)]
    ).sort_index()
    if scenes.empty:
        raise SystemExit("no scene had enough clear pixels; try a lower --min-clear")

    table = yearly_table(scenes)
    print()
    print(table.to_string())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"ndvi_yearly_{args.geohash}_{first}-{last}.csv"
    png_path = csv_path.with_suffix(".png")
    table.to_csv(csv_path)
    plot(scenes, table, args.geohash, png_path)
    print(f"\nTable: {csv_path}\nChart: {png_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
