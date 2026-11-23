import shutil
from pathlib import Path

import pytest

from cubesip.config import ENV_VAR, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_loads(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cubesip.yaml"
    shutil.copy(ROOT / "cubesip.example.yaml", cfg_file)
    config = load_config(cfg_file)
    assert config.manifest == (tmp_path / "data" / "manifest.csv").resolve()
    assert config.data_dir == (tmp_path / "data").resolve()
    assert config.openeo.auth.method == "oidc"
    assert config.cdse_federation.url == "https://openeofed.dataspace.copernicus.eu"
    assert config.stac.backend == "openeo"
    assert config.earthdatahub.token is None  # empty string in template means "not set"


def test_new_sections_are_optional(tmp_path: Path) -> None:
    """A config written before cdse_federation / stac existed still loads, with defaults."""
    cfg_file = tmp_path / "cubesip.yaml"
    cfg_file.write_text("manifest: m.csv\nopeneo:\n  url: https://openeo.example.eu\n")
    config = load_config(cfg_file)
    assert config.cdse_federation.url == "https://openeofed.dataspace.copernicus.eu"
    assert config.stac.backend == "openeo"


def test_data_dir_defaults_next_to_config(tmp_path: Path) -> None:
    """A config without data_dir still loads; files go to ./data beside the config file."""
    cfg_file = tmp_path / "cubesip.yaml"
    cfg_file.write_text("manifest: m.csv\n")
    assert load_config(cfg_file).data_dir == (tmp_path / "data").resolve()


def test_data_dir_absolute_is_kept(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    cfg_file = tmp_path / "cubesip.yaml"
    cfg_file.write_text(f"manifest: m.csv\ndata_dir: '{target.as_posix()}'\n")
    assert load_config(cfg_file).data_dir == target


def test_secrets_never_printed(tmp_path: Path) -> None:
    cfg_file = tmp_path / "cubesip.yaml"
    cfg_file.write_text("manifest: m.csv\nearthdatahub:\n  token: super-secret-token\n")
    config = load_config(cfg_file)
    assert config.earthdatahub.token.get_secret_value() == "super-secret-token"
    assert "super-secret-token" not in repr(config)


def test_env_var_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "other.yaml"
    cfg_file.write_text("manifest: m.csv\n")
    monkeypatch.setenv(ENV_VAR, str(cfg_file))
    assert load_config().manifest.name == "m.csv"


def test_missing_config_is_explained(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cubesip.example.yaml"):
        load_config(tmp_path / "nope.yaml")
