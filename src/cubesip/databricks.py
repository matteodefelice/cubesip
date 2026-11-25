"""Databricks helpers: move files into Unity Catalog volumes with ``dbutils``.

xarray / netCDF4 cannot write directly to a ``/Volumes/...`` path, so when a request sets
``stage_locally`` the NetCDF file is first written to the local disk (e.g. ``/tmp``) and then
moved to its target with ``dbutils.fs.mv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_dbutils() -> Any:
    """Return the Databricks ``dbutils`` object. Raises a RuntimeError outside Databricks."""
    try:
        from databricks.sdk.runtime import dbutils  # Databricks Runtime 13+

        return dbutils
    except ImportError:
        pass
    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        return DBUtils(SparkSession.builder.getOrCreate())
    except ImportError as exc:
        raise RuntimeError(
            "stage_locally=True needs Databricks dbutils, which could not be found; "
            "set stage_locally to false when not running on Databricks"
        ) from exc


def move_to_target(local_path: Path, target_path: Path) -> None:
    """Move a file from the local disk to its target (e.g. a Unity Catalog volume) with
    ``dbutils.fs.mv``, creating the target folder first.
    """
    dbutils = get_dbutils()
    dbutils.fs.mkdirs(Path(target_path).parent.as_posix())
    dbutils.fs.mv(f"file:{Path(local_path).as_posix()}", Path(target_path).as_posix())
