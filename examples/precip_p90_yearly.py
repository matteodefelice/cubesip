"""Yearly 90th percentile of daily precipitation for a site, from ERA5 on Earth Data Hub.

For one geohash cell the script:

1. requests ERA5 hourly total precipitation (``tp``) from Earth Data Hub, one cubesip
   request per year. Each year is written to the manifest, so running the script again
   reads the files from disk instead of downloading;
2. averages over the ERA5 0.25 deg cells that touch the geohash (usually one) and builds
   daily totals in mm. ERA5 ``tp`` at time t is the rain of the hour ending at t, so the
   hour ending at 00:00 belongs to the previous day; days without all 24 hours (the edges
   of the period) are dropped;
3. for every year computes the 90th percentile of the daily totals on wet days
   (>= 1 mm, the usual climate-index convention; ``--wet-day-threshold 0`` uses every day);
4. saves a chart (PNG) and a table (CSV) and prints the table.

Run from the repository root, with an Earth Data Hub token in cubesip.yaml:

    uv run --group examples python examples/precip_p90_yearly.py
    uv run --group examples python examples/precip_p90_yearly.py --latlon 45.30 9.50
    uv run --group examples python examples/precip_p90_yearly.py --years 1995 2024 --percentile 95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import cubesip
from cubesip.spatial import find_dims

# Default site (lat, lon): same cropland cell as examples/ndvi_yearly.py.
DEFAULT_SITE = (48.30, 1.65)
DATASET = "era5/era5-single-levels-atmosphere-v0"

# Chart colours (light theme): one accent for the yearly percentile, muted grey for context.
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
    where.add_argument("--geohash", help="geohash of the site, e.g. u094v8")
    where.add_argument(
        "--latlon",
        type=float,
        nargs=2,
        default=DEFAULT_SITE,
        metavar=("LAT", "LON"),
        help="site coordinates (default: cropland in the Beauce plain, France)",
    )
    parser.add_argument(
        "--precision", type=int, default=6, help="geohash length for --latlon (6 = ~1 km)"
    )
    parser.add_argument(
        "--years", type=int, nargs=2, default=[2015, 2024], metavar=("FIRST", "LAST")
    )
    parser.add_argument("--percentile", type=float, default=90, help="percentile (0-100)")
    parser.add_argument(
        "--wet-day-threshold",
        type=float,
        default=1.0,
        help="only days with at least this much rain (mm) count; 0 = every day",
    )
    parser.add_argument("--target-dir", default="data", help="where NetCDF files are stored")
    parser.add_argument("--output-dir", default="data", help="where the PNG and CSV are saved")
    parser.add_argument("--config", help="cubesip config file (default: ./cubesip.yaml)")
    parser.add_argument("--show", action="store_true", help="also open the chart in a window")
    args = parser.parse_args()
    if not 0 < args.percentile < 100:
        parser.error("--percentile must be between 0 and 100")
    args.geohash = args.geohash or cubesip.geohash.encode(*args.latlon, args.precision)
    return args


def fetch_year(year: int, args: argparse.Namespace, config: cubesip.Config) -> xr.Dataset:
    """ERA5 hourly ``tp`` for one calendar year (downloaded or read from the cache)."""
    print(f"{year}: ", end="", flush=True)
    result = cubesip.run(
        {
            "driver": "edh",
            "dataset": DATASET,
            "variables": ["tp"],
            "geohash": args.geohash,
            "time": {"start": f"{year}-01-01", "end": f"{year}-12-31"},
            "target_dir": args.target_dir,
        },
        config,
    )
    print(f"{'from cache' if result.from_cache else 'downloaded'} {result.path.name}")
    return result.dataset


def hourly_mm(ds: xr.Dataset) -> pd.Series:
    """Hourly precipitation (mm), averaged over the ERA5 cells touching the geohash."""
    dims = find_dims(ds)
    tp = ds["tp"].mean(dim=(dims.y, dims.x))
    return tp.to_series().astype("float64") * 1000.0  # m -> mm


def daily_mm(hourly: pd.Series) -> pd.Series:
    """Daily totals (mm).

    A value at time t is the rain of the hour ending at t, so day d collects the hours in
    (d 00:00, d+1 00:00]: bins closed on the right and labelled by their start.
    """
    days = hourly.resample("D", closed="right", label="left")
    totals, hours = days.sum(), days.count()
    daily = totals[hours == 24]  # drop incomplete days at the edges of the period
    daily.index.name = "date"
    return daily


def yearly_table(daily: pd.Series, wet: pd.Series, percentile: float) -> pd.DataFrame:
    wet_by_year = wet.groupby(wet.index.year)
    by_year = daily.groupby(daily.index.year)
    table = pd.DataFrame(
        {
            f"p{percentile:g}_mm": wet_by_year.quantile(percentile / 100),
            "wet_days": wet_by_year.size(),
            "max_mm": by_year.max(),
            "total_mm": by_year.sum(),
            "days": by_year.size(),
        }
    )
    table.index.name = "year"
    return table.round(1)


def plot(
    wet: pd.Series,
    yearly: pd.Series,
    percentile: float,
    geohash: str,
    days_label: str,
    path: Path,
) -> None:
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

    # Context: every day that counts, small and muted.
    ax.scatter(
        wet.index,
        wet.values,
        s=8,
        color=INK_MUTED,
        alpha=0.35,
        linewidths=0,
        label="Daily total",
        zorder=2,
    )

    # The story: one point per year, placed mid-year, joined by a 2px line.
    mid_year = pd.to_datetime([f"{year}-07-01" for year in yearly.index])
    ax.plot(
        mid_year,
        yearly.values,
        color=ACCENT,
        linewidth=2,
        solid_joinstyle="round",
        solid_capstyle="round",
        marker="o",
        markersize=7,
        markerfacecolor=ACCENT,
        markeredgecolor=SURFACE,  # surface ring keeps markers legible over the dots
        markeredgewidth=2,
        label=f"Yearly {percentile:g}th percentile",
        zorder=3,
    )

    # Label only the latest year, in text ink beside its marker.
    ax.annotate(
        f"{yearly.index[-1]}: {yearly.iloc[-1]:.1f} mm",
        xy=(mid_year[-1], yearly.iloc[-1]),
        xytext=(10, 0),
        textcoords="offset points",
        va="center",
        color=INK_PRIMARY,
        fontsize=10,
        fontweight="semibold",
    )

    ax.set_ylim(0, float(wet.max()) * 1.05)
    ax.set_ylabel("Daily precipitation (mm)", color=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, length=0, pad=6)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=max(1, len(yearly) // 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    first, last = yearly.index[0], yearly.index[-1]
    ax.set_xlim(pd.Timestamp(f"{first}-01-01"), pd.Timestamp(f"{last}-12-31"))

    fig.subplots_adjust(top=0.84)  # room for the title block
    fig.text(
        0.02,
        0.96,
        f"{percentile:g}th percentile of daily precipitation per year, geohash {geohash}",
        va="top",
        color=INK_PRIMARY,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.905,
        f"ERA5 total precipitation (Earth Data Hub), {days_label}; "
        "mean of the 0.25° cells touching the geohash",
        va="top",
        color=INK_SECONDARY,
        fontsize=10,
    )
    legend = ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # "tight" grows the image to fit every label, e.g. the end label past the axes.
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.2)


def main() -> None:
    args = parse_args()
    config = cubesip.load_config(args.config)
    first, last = args.years
    box = cubesip.geohash.bbox(args.geohash)
    print(
        f"Site {args.geohash}: lat {box.south:.4f}..{box.north:.4f}, "
        f"lon {box.west:.4f}..{box.east:.4f}; years {first}-{last}"
    )

    hourly = pd.concat(
        [hourly_mm(fetch_year(year, args, config)) for year in range(first, last + 1)]
    ).sort_index()
    daily = daily_mm(hourly)
    threshold = args.wet_day_threshold
    if threshold > 0:
        wet, days_label = daily[daily >= threshold], f"wet days (>= {threshold:g} mm)"
    else:
        wet, days_label = daily, "all days"
    table = yearly_table(daily, wet, args.percentile)
    print()
    print(table.to_string())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"precip_p{args.percentile:g}_yearly_{args.geohash}_{first}-{last}"
    csv_path = out / f"{stem}.csv"
    png_path = out / f"{stem}.png"
    table.to_csv(csv_path)
    yearly = table[f"p{args.percentile:g}_mm"]
    plot(wet, yearly, args.percentile, args.geohash, days_label, png_path)
    print(f"\nTable: {csv_path}\nChart: {png_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
