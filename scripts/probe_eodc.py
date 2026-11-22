"""Probe the EODC openEO back-end: can cubesip's openEO machinery use it as is?

Public checks (no account needed):
  1. the API answers (/.well-known/openeo) and which version is in production
  2. capabilities: synchronous processing (POST /result), batch jobs, billing plans
  3. login: OIDC providers, default clients, device-code grant; basic auth endpoint
  4. output formats: is netCDF offered, and under which exact name
  5. processes: load_collection parameters (properties), load_stac
  6. collections: count, and whether they match the public STAC (stac.eodc.eu)
  7. collection metadata read with cubesip's helpers (bands, resolution, CRS, derived_from)
  8. acquisition dates counted on the public STAC, and the size estimate cubesip would make

With --download (needs an EODC account; the first run prints a device-code login URL):
  9. login and account details (roles)
 10. a small synchronous download, timed
 11. the NetCDF read back: dimensions, time steps, and whether cubesip finds the CRS
 12. with --properties: a second download with the filter, to see if it is applied

Usage:
  uv run python scripts/probe_eodc.py
  uv run python scripts/probe_eodc.py --download
  uv run python scripts/probe_eodc.py --download \
      --properties '{"sat:orbit_state": {"eq": "descending"}}'

Exit code 1 if any check FAILs. Downloads go to --out (default ./data/probe_eodc, git-ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

import requests

from cubesip import geohash as gh
from cubesip import spatial
from cubesip.drivers.openeo import (
    BYTES_PER_VALUE,
    _single_epsg,
    available_bands,
    box_size_m,
    derived_from_target,
    property_filters,
    resolution_m,
    spatial_extent,
    temporal_extent,
)
from cubesip.request import Request

DEFAULT_URL = "https://openeo.eodc.eu"
DEFAULT_STAC = "https://stac.eodc.eu/api/v1"
DEFAULT_GEOHASH = gh.encode(lat=48.21, lon=16.37, precision=6)  # Vienna; EODC data is EU/AT
DEVICE_GRANTS = (
    "urn:ietf:params:oauth:grant-type:device_code",
    "urn:ietf:params:oauth:grant-type:device_code+pkce",
)

results: list[tuple[str, str, str]] = []


def report(status: str, check: str, detail: str = "") -> None:
    """Record and print one result line: PASS, FAIL, WARN or INFO."""
    results.append((status, check, detail))
    print(f"[{status:4}] {check}" + (f": {detail}" if detail else ""), flush=True)


def get_json(url: str, timeout: float, **kwargs) -> tuple[int, Any]:
    """GET a URL and return (status code, parsed JSON or None)."""
    response = requests.get(url, timeout=timeout, **kwargs)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


# -- public checks ---------------------------------------------------------------------------


def check_reachable(url: str, timeout: float) -> str | None:
    """1. Discovery document; returns the base URL of the production API version."""
    try:
        status, body = get_json(f"{url}/.well-known/openeo", timeout)
    except requests.RequestException as exc:
        report("FAIL", "API reachable", f"{url}: {type(exc).__name__}: {exc}")
        return None
    if status != 200 or not body:
        report("FAIL", "API reachable", f"HTTP {status} from {url}/.well-known/openeo")
        return None
    versions = body.get("versions") or []
    listed = ", ".join(
        f"{v.get('api_version')}{'' if v.get('production') else ' (dev)'}" for v in versions
    )
    production = [v for v in versions if v.get("production")] or versions
    if not production:
        report("FAIL", "API reachable", "no versions listed")
        return None
    best = max(production, key=lambda v: tuple(int(p) for p in str(v["api_version"]).split(".")))
    report("PASS", "API reachable", f"versions: {listed}; using {best['url']}")
    return best["url"].rstrip("/")


def check_capabilities(base: str, timeout: float) -> None:
    """2. Synchronous processing, batch jobs and billing."""
    _, caps = get_json(f"{base}/", timeout)
    caps = caps or {}
    report(
        "INFO",
        "back-end",
        f"{caps.get('title')} api {caps.get('api_version')}, backend {caps.get('backend_version')}",
    )
    endpoints = {(e["path"], m) for e in caps.get("endpoints", []) for m in e.get("methods", [])}
    if ("/result", "POST") in endpoints:
        report("PASS", "synchronous processing", "POST /result listed")
    else:
        report(
            "FAIL",
            "synchronous processing",
            "POST /result not listed: cubesip's download() cannot work, batch jobs needed",
        )
    report("INFO", "batch jobs", "listed" if ("/jobs", "POST") in endpoints else "not listed")
    billing = caps.get("billing")
    if billing:
        plans = ", ".join(
            f"{p.get('name')}{' (paid)' if p.get('paid') else ''}" for p in billing.get("plans", [])
        )
        report("WARN", "billing", f"currency {billing.get('currency')}; plans: {plans}")


def check_auth(base: str, timeout: float) -> None:
    """3. OIDC providers and clients; basic auth."""
    status, body = get_json(f"{base}/credentials/oidc", timeout)
    providers = (body or {}).get("providers") or []
    if status != 200 or not providers:
        report("FAIL", "OIDC providers", f"HTTP {status}, none listed")
    for provider in providers:
        clients = provider.get("default_clients")
        if clients is None:
            odd = [k for k in provider if k.lower() == "default_clients" and k != "default_clients"]
            note = f" (non-standard key {odd[0]!r}: openeo client will not see it)" if odd else ""
            report(
                "WARN",
                f"OIDC provider {provider.get('id')!r}",
                f"issuer {provider.get('issuer')}; no default client{note}: set auth.client_id",
            )
            continue
        for client in clients:
            grants = client.get("grant_types") or []
            device = any(g in grants for g in DEVICE_GRANTS)
            report(
                "PASS" if device else "WARN",
                f"OIDC provider {provider.get('id')!r}",
                f"issuer {provider.get('issuer')}; default client {client.get('id')}; "
                f"device-code {'allowed' if device else 'NOT allowed'} (grants: {grants})",
            )
    status, _ = get_json(f"{base}/credentials/basic", timeout)
    report(
        "INFO",
        "basic auth",
        {401: "endpoint exists (401 without credentials)", 404: "not offered"}.get(
            status, f"HTTP {status}"
        ),
    )


def check_formats(base: str, timeout: float) -> str | None:
    """4. Output formats; returns the exact netCDF format name."""
    _, body = get_json(f"{base}/file_formats", timeout)
    outputs = list(((body or {}).get("output") or {}).keys())
    netcdf = next((name for name in outputs if name.lower() == "netcdf"), None)
    if netcdf:
        report("PASS", "netCDF output", f"offered as {netcdf!r} (all: {outputs})")
    else:
        report("FAIL", "netCDF output", f"not offered; outputs: {outputs}")
    return netcdf


def check_processes(base: str, timeout: float) -> None:
    """5. load_collection parameters and load_stac."""
    _, body = get_json(f"{base}/processes", timeout)
    processes = {p["id"]: p for p in (body or {}).get("processes", [])}
    report("INFO", "processes", f"{len(processes)} listed")
    load = processes.get("load_collection")
    if not load:
        report("FAIL", "load_collection", "not listed")
    else:
        params = [p["name"] for p in load.get("parameters", [])]
        status = "PASS" if "properties" in params else "WARN"
        report(
            status,
            "load_collection parameters",
            f"{params}"
            + (
                ""
                if status == "PASS"
                else ": filters (max_cloud_cover, properties) may be rejected or ignored"
            ),
        )
    report(
        "INFO",
        "load_stac",
        "listed (stac driver possible)"
        if "load_stac" in processes
        else "not listed (cannot back the stac driver)",
    )


def check_collections(base: str, stac: str, collection: str, timeout: float) -> bool:
    """6. Collection list vs the public STAC."""
    status, body = get_json(f"{base}/collections", timeout)
    ids = {c["id"] for c in (body or {}).get("collections", [])}
    if status != 200 or not ids:
        report("FAIL", "collections", f"HTTP {status}, none listed")
        return False
    try:
        _, stac_body = get_json(f"{stac}/collections?limit=500", timeout)
        stac_ids = {c["id"] for c in (stac_body or {}).get("collections", [])}
        only_api, only_stac = sorted(ids - stac_ids), sorted(stac_ids - ids)
        report(
            "INFO",
            "collections vs STAC",
            f"{len(ids)} in openEO, {len(stac_ids)} in STAC; only in openEO: "
            f"{only_api[:10]}; only in STAC: {only_stac[:10]}",
        )
    except requests.RequestException as exc:
        report("WARN", "collections vs STAC", f"STAC not readable: {exc}")
    found = collection in ids
    report(
        "PASS" if found else "FAIL",
        f"collection {collection}",
        "listed" if found else "not listed in /collections",
    )
    return found


def check_metadata(base: str, req: Request) -> dict | None:
    """7. Collection metadata read with openeo + cubesip helpers."""
    import openeo

    try:
        meta = openeo.connect(base).describe_collection(req.dataset)
    except Exception as exc:
        report("FAIL", "describe_collection", f"{type(exc).__name__}: {exc}")
        return None
    bands = available_bands(meta)
    missing = [b for b in req.variables if bands is not None and b not in bands]
    if bands is None:
        report("WARN", "bands", "not listed in metadata; cubesip cannot check them")
    elif missing:
        report("FAIL", "bands", f"missing {missing}; available {bands}")
    else:
        report("PASS", "bands", f"{req.variables} found among {bands}")
    res = resolution_m(meta, req.variables)
    report(
        "PASS" if res else "WARN",
        "resolution",
        f"{res:g} m" if res else "not in metadata: size estimate will be 'unknown'",
    )
    report("INFO", "single EPSG in metadata", str(_single_epsg(meta)))
    target = derived_from_target(meta)
    report(
        "PASS" if target else "WARN",
        "derived_from link",
        f"{target}"
        if target
        else "absent: the openeo driver cannot count dates; an EODC "
        "driver should search the public STAC instead",
    )
    return meta


def count_stac_dates(stac: str, req: Request, filters: dict | None = None) -> int | None:
    """Distinct acquisition dates on the STAC, optionally filtered like load_collection would."""
    from pystac_client import Client

    search = Client.open(stac).search(
        collections=[req.dataset],
        bbox=list(req.bbox),
        datetime=f"{req.time.start.isoformat()}/{req.time.end.isoformat()}",
        limit=200,
        max_items=5000,
    )
    dates = set()
    for item in search.items_as_dicts():
        props = item.get("properties", {})
        if filters and not all(
            prop in props and test(props[prop]) for prop, test in filters.items()
        ):
            continue
        dates.add((props.get("datetime") or props.get("start_datetime") or "")[:10])
    return len(dates)


def check_size(stac: str, req: Request, meta: dict | None) -> int | None:
    """8. Date count on the public STAC and the size estimate cubesip would make."""
    try:
        steps = count_stac_dates(stac, req)
        report(
            "PASS" if steps else "WARN",
            "dates on public STAC",
            f"{steps} distinct date(s) in {req.time.start}..{req.time.end}",
        )
    except Exception as exc:
        report("WARN", "dates on public STAC", f"{type(exc).__name__}: {exc}")
        steps = None
    res = resolution_m(meta, req.variables) if meta else None
    if res and steps is not None:
        width, height = box_size_m(req)
        nx, ny = max(1, round(width / res)), max(1, round(height / res))
        nbytes = nx * ny * len(req.variables) * max(steps, 1) * BYTES_PER_VALUE
        report(
            "INFO",
            "size estimate",
            f"{nx}x{ny} px, {len(req.variables)} band(s), {steps} step(s): {nbytes / 1e6:.1f} MB",
        )
    return steps


# -- authenticated checks --------------------------------------------------------------------


def check_login(base: str, provider_id: str | None, client_id: str | None):
    """9. Login (stored refresh token, else device code) and account info."""
    import openeo

    connection = openeo.connect(base)
    try:
        connection.authenticate_oidc(provider_id=provider_id, client_id=client_id)
    except Exception as exc:
        report("FAIL", "login", f"{type(exc).__name__}: {exc}")
        return None
    try:
        me = connection.describe_account()
        report(
            "PASS",
            "login",
            f"user {me.get('user_id')}; roles {me.get('roles')}; "
            f"plan {me.get('default_plan')}; budget {me.get('budget')}",
        )
    except Exception as exc:
        report("WARN", "login", f"logged in, but /me failed: {exc}")
    return connection


def download(
    connection, req: Request, fmt: str, out: Path, label: str, filters: dict | None = None
) -> Path | None:
    """10./12. One synchronous download, timed."""
    cube = connection.load_collection(
        req.dataset,
        spatial_extent=spatial_extent(req),
        temporal_extent=temporal_extent(req),
        bands=list(req.variables),
        properties=filters or None,
    )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{req.dataset}_{req.geohash}_{label}.nc"
    start = time.monotonic()
    try:
        cube.download(path, format=fmt)
    except Exception as exc:
        report(
            "FAIL",
            f"download ({label})",
            f"after {time.monotonic() - start:.0f} s: {type(exc).__name__}: {exc}",
        )
        return None
    seconds = time.monotonic() - start
    size = path.stat().st_size if path.is_file() else 0
    report(
        "PASS" if size else "FAIL",
        f"download ({label})",
        f"{size / 1e6:.2f} MB in {seconds:.0f} s -> {path}",
    )
    return path if size else None


def inspect(path: Path, req: Request, stac_steps: int | None) -> int | None:
    """11. Read the NetCDF back the way cubesip does."""
    import xarray as xr
    from pyproj import CRS

    ds = xr.open_dataset(path, decode_coords="all")
    report(
        "INFO",
        "NetCDF content",
        f"dims {dict(ds.sizes)}; vars {list(ds.data_vars)}; "
        f"coords {list(ds.coords)}; global attrs {list(ds.attrs)}",
    )
    missing = [v for v in req.variables if v not in ds.data_vars]
    report(
        "FAIL" if missing else "PASS",
        "bands as variables",
        f"missing {missing}" if missing else "one variable per band",
    )
    steps = None
    try:
        dims = spatial.find_dims(ds)
        steps = ds.sizes[dims.time] if dims.time else 1
        compare = "" if stac_steps is None else f" (STAC count: {stac_steps})"
        report(
            "INFO",
            "dimensions",
            f"time={dims.time} y={dims.y} x={dims.x}; {steps} time step(s){compare}",
        )
    except ValueError as exc:
        report("FAIL", "dimensions", str(exc))
    try:
        report("PASS", "CRS found by cubesip", spatial.native_crs(ds))
    except ValueError as exc:
        attr = ds.attrs.get("crs")
        if attr:
            crs = CRS.from_user_input(attr)
            report(
                "FAIL",
                "CRS found by cubesip",
                f"{exc}; but global attr crs={crs.to_string()!r} parses: read ds.attrs['crs'] "
                "as a fallback in spatial.native_crs",
            )
        else:
            report("FAIL", "CRS found by cubesip", f"{exc}; no global crs attribute either")
    ds.close()
    return steps


# -- main ------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL, help="openEO back-end (default %(default)s)")
    parser.add_argument("--stac", default=DEFAULT_STAC, help="public STAC (default %(default)s)")
    parser.add_argument("--collection", default="SENTINEL1_SIG0_20M")
    parser.add_argument("--bands", nargs="+", default=["VV", "VH"])
    parser.add_argument("--geohash", default=DEFAULT_GEOHASH, help="default: Vienna, level 6")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 7, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 7, 7))
    parser.add_argument("--timeout", type=float, default=30, help="seconds per HTTP request")
    parser.add_argument("--download", action="store_true", help="log in and download")
    parser.add_argument("--provider-id", help="OIDC provider id (default: the back-end's first)")
    parser.add_argument("--client-id", help="OIDC client id (default: the provider's default)")
    parser.add_argument(
        "--properties",
        type=json.loads,
        help='filter to test, e.g. \'{"sat:orbit_state": {"eq": "descending"}}\'',
    )
    parser.add_argument("--out", type=Path, default=Path("data/probe_eodc"))
    args = parser.parse_args()

    req = Request(
        driver="openeo",
        dataset=args.collection,
        variables=args.bands,
        geohash=args.geohash,
        time={"start": args.start, "end": args.end},
        target_dir=args.out,
        options={"properties": args.properties} if args.properties else {},
    )
    print(
        f"Probing {args.url} with {req.dataset} {req.variables} @ {req.geohash} "
        f"{req.time.start}..{req.time.end}\n"
    )

    base = check_reachable(args.url.rstrip("/"), args.timeout)
    if base is None:
        check_size(args.stac, req, None)
        print("\nThe openEO API does not answer; only the public STAC was checked.")
        return 1
    check_capabilities(base, args.timeout)
    check_auth(base, args.timeout)
    fmt = check_formats(base, args.timeout)
    check_processes(base, args.timeout)
    check_collections(base, args.stac, req.dataset, args.timeout)
    meta = check_metadata(base, req)
    stac_steps = check_size(args.stac, req, meta)

    if args.download:
        connection = check_login(base, args.provider_id, args.client_id)
        if connection is not None and fmt:
            path = download(connection, req, fmt, args.out, "plain")
            steps = inspect(path, req, stac_steps) if path else None
            if args.properties and steps is not None:
                filters = property_filters(req.options)
                expected = count_stac_dates(args.stac, req, filters)
                fpath = download(connection, req, fmt, args.out, "filtered", filters)
                fsteps = inspect(fpath, req, expected) if fpath else None
                if fsteps is not None:
                    applied = fsteps == expected or (fsteps < steps and expected < steps)
                    report(
                        "PASS" if applied else "FAIL",
                        "properties filter",
                        f"{steps} step(s) unfiltered, {fsteps} filtered, {expected} expected "
                        f"from STAC: {'applied' if applied else 'IGNORED'}",
                    )
    else:
        print("\n(skipped login and download; add --download to run them)")

    counts = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{counts['PASS']} passed, {counts['WARN']} warnings, {counts['FAIL']} failed")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
