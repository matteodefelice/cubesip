"""YAML configuration: credentials for the openEO back-ends / Earth Data Hub and the manifest.

The real file (``cubesip.yaml``) is git-ignored; ``cubesip.example.yaml`` is the template.
Secrets are held as ``SecretStr`` so they never show up in reprs, logs or tracebacks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

ENV_VAR = "CUBESIP_CONFIG"
DEFAULT_LOCATIONS = (
    Path("cubesip.yaml"),
    Path.home() / ".config" / "cubesip" / "cubesip.yaml",
)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenEOAuth(_Section):
    method: Literal["oidc", "client_credentials", "basic"] = "oidc"
    provider_id: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    username: str | None = None
    password: SecretStr | None = None


CDSE_URL = "https://openeo.dataspace.copernicus.eu"
CDSE_FEDERATION_URL = "https://openeofed.dataspace.copernicus.eu"


class OpenEOConfig(_Section):
    """One openEO back-end: where it is, how to log in, the synchronous download limit."""

    url: str = CDSE_URL
    auth: OpenEOAuth = Field(default_factory=OpenEOAuth)
    max_sync_bytes: int = 500_000_000


class StacConfig(_Section):
    # Config section of the openEO back-end that runs load_stac for the stac driver.
    backend: Literal["openeo", "cdse_federation"] = "openeo"


class EarthDataHubConfig(_Section):
    url: str = "https://data.earthdatahub.destine.eu"
    token: SecretStr | None = None


class Config(_Section):
    manifest: Path
    openeo: OpenEOConfig = Field(default_factory=OpenEOConfig)
    cdse_federation: OpenEOConfig = Field(
        default_factory=lambda: OpenEOConfig(url=CDSE_FEDERATION_URL)
    )
    stac: StacConfig = Field(default_factory=StacConfig)
    earthdatahub: EarthDataHubConfig = Field(default_factory=EarthDataHubConfig)


def find_config(path: str | Path | None = None) -> Path:
    """Find the config file to use. A path given here wins; otherwise the CUBESIP_CONFIG environment
    variable; otherwise cubesip.yaml in the current folder, then in ~/.config/cubesip. Raises
    FileNotFoundError if none exists.
    """
    if path is not None:
        candidates = [Path(path)]
    elif os.environ.get(ENV_VAR):
        candidates = [Path(os.environ[ENV_VAR])]
    else:
        candidates = list(DEFAULT_LOCATIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No cubesip config found (tried: {tried}). Copy cubesip.example.yaml to cubesip.yaml."
    )


def load_config(path: str | Path | None = None) -> Config:
    """Read the YAML config file, check that it is valid and return it as a Config. A relative
    manifest path is taken relative to the folder of the config file.
    """
    config_path = find_config(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    # Empty strings in the template mean "not set".
    for section in ("openeo", "cdse_federation", "stac", "earthdatahub"):
        _drop_empty(raw.get(section))
    config = Config.model_validate(raw)
    if not config.manifest.is_absolute():
        config.manifest = (config_path.parent / config.manifest).resolve()
    return config


def _drop_empty(node: object) -> None:
    """Remove settings left as empty strings (as in the template), so they count as 'not set' and
    get their default values. Nested sections are cleaned too.
    """
    if isinstance(node, dict):
        for key in [k for k, v in node.items() if v == ""]:
            del node[key]
        for value in node.values():
            _drop_empty(value)
