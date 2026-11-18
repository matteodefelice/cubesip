"""Pre-flight check of a request: does the dataset exist, is it cached, how big is it.

Implemented in milestone M2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cubesip.config import Config
from cubesip.drivers import DatasetInfo, SizeEstimate
from cubesip.manifest import CacheHit
from cubesip.request import Request


@dataclass
class CheckReport:
    request: Request
    dataset: DatasetInfo | None  # None if the dataset was not found
    cache_hit: CacheHit | None
    size: SizeEstimate | None  # not computed when served from cache
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if run() can proceed (cache hit, or dataset found with all variables)."""
        if self.cache_hit is not None:
            return True
        return self.dataset is not None and not self.dataset.missing_variables


def check(request: Request, config: Config) -> CheckReport:
    """1) manifest cache lookup (geohash ancestors, time and variable coverage);
    2) if not cached: dataset/variables exist on the source; 3) size estimate + warnings."""
    raise NotImplementedError("M2")
