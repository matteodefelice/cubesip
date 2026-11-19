"""Pre-flight check of a request: is it cached, does the dataset exist, how big is it."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cubesip.config import Config
from cubesip.drivers import DatasetInfo, DatasetNotFoundError, Driver, SizeEstimate, get_driver
from cubesip.manifest import CacheHit, Manifest
from cubesip.request import Request
from cubesip.spatial import EmptySelectionError

log = logging.getLogger(__name__)


@dataclass
class CheckReport:
    request: Request
    cache_hit: CacheHit | None = None
    dataset: DatasetInfo | None = None  # None on a cache hit or when the dataset is missing
    size: SizeEstimate | None = None  # not computed on a cache hit
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when run() can go ahead: either the data are already cached, or the dataset was
        found and has every requested variable.
        """
        if self.error is not None:
            return False
        if self.cache_hit is not None:
            return True
        return self.dataset is not None and not self.dataset.missing_variables

    def summary(self) -> str:
        """Build a short, readable text about the check: what was asked, whether it is cached or
        ready to download, what was found about the dataset, the expected size and any warnings.
        """
        r = self.request
        period = f"{r.time.start} to {r.time.end}" if r.time else "static"
        lines = [
            f"cubesip check: {r.driver} {r.dataset} [{', '.join(r.variables)}] "
            f"@ {r.geohash}, {period}"
        ]
        if self.cache_hit is not None:
            lines.append(
                f"  status : cached (geohash {self.cache_hit.geohash}) in {self.cache_hit.path}"
            )
        elif self.ok:
            lines.append("  status : ready to download")
        else:
            lines.append(f"  status : NOT READY - {self.error or 'see below'}")
        if self.dataset is not None:
            missing = ", ".join(self.dataset.missing_variables) or "none"
            lines.append(
                f"  dataset: found, CRS {self.dataset.crs or 'native (set by the source)'}, "
                f"missing variables: {missing}"
            )
            if self.dataset.missing_variables and self.dataset.available_variables:
                lines.append(f"  available: {', '.join(self.dataset.available_variables)}")
        if self.size is not None:
            size = format_bytes(self.size.nbytes) if self.size.nbytes is not None else "unknown"
            detail = f", {self.size.detail}" if self.size.detail else ""
            lines.append(f"  size   : {size} ({self.size.method}{detail})")
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)

    def __str__(self) -> str:
        """Printing a report shows its summary."""
        return self.summary()


def format_bytes(nbytes: int | float) -> str:
    """Turn a number of bytes into a readable size such as '12.3 MB' (counting in steps of 1000)."""
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    raise AssertionError("unreachable")


def check(request: Request, config: Config, driver: Driver | None = None) -> CheckReport:
    """Check a request step by step without downloading anything. First look in the manifest for a
    file that already covers it; if there is none, ask the source whether the dataset and variables
    exist; finally estimate how big the download would be and warn if it is too large. Problems are
    written into the report instead of being raised.
    """
    driver = driver or get_driver(request.driver, config)
    report = CheckReport(request=request)
    try:
        driver.validate(request)
    except ValueError as exc:
        report.error = str(exc)
        return report

    key = driver.dataset_key(request.dataset)
    backend = driver.backend_url
    report.cache_hit = Manifest(config.manifest).find_covering(request, key, backend)
    if report.cache_hit is not None:
        log.info("cache hit: %s", report.cache_hit.path)
        if backend and not report.cache_hit.backend:
            report.warnings.append(
                "the cached file was recorded before cubesip tracked back-ends; it is assumed "
                f"to come from {backend} (run with force=True to download it again)"
            )
        return report

    try:
        report.dataset = driver.describe(request)
    except (DatasetNotFoundError, PermissionError, ValueError) as exc:
        report.error = str(exc)
        return report
    report.warnings += report.dataset.notes
    if report.dataset.missing_variables:
        report.error = f"variables not available: {', '.join(report.dataset.missing_variables)}"
        return report

    try:
        report.size = driver.estimate_size(request)
    except (EmptySelectionError, ValueError, KeyError) as exc:
        report.error = str(exc)
        return report
    if report.size.nbytes is None:
        report.warnings.append(f"size could not be estimated: {report.size.detail}")
    limit = driver.max_request_bytes
    if limit is not None and report.size.nbytes is not None and report.size.nbytes > limit:
        report.warnings.append(
            f"estimated size {format_bytes(report.size.nbytes)} exceeds the {driver.name} limit "
            f"of {format_bytes(limit)}: the request may fail; consider a smaller area or period"
        )
    return report
