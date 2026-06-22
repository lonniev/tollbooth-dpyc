"""Tests for resolve_service_version — the fleet-wide version resolver."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

from tollbooth.version import resolve_service_version


def test_prefers_installed_metadata():
    with patch("importlib.metadata.version", return_value="1.2.3"):
        assert resolve_service_version("any-dist") == "1.2.3"


def test_falls_back_to_pyproject_when_not_installed(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "9.9.9"\n'
    )
    init_file = tmp_path / "src" / "pkg" / "__init__.py"
    init_file.parent.mkdir(parents=True)
    init_file.write_text("")
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        # Walks up from the source file to the repo-root pyproject.
        assert resolve_service_version("x", str(init_file)) == "9.9.9"


def test_flat_layout_file_hint_finds_sibling_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "flat"\nversion = "0.11.1"\n'
    )
    server = tmp_path / "server.py"
    server.write_text("")
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        assert resolve_service_version("flat", str(server)) == "0.11.1"


def test_returns_default_when_unresolvable(tmp_path):
    lonely = tmp_path / "lonely.py"  # no pyproject anywhere up-tree
    lonely.write_text("")
    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        assert resolve_service_version("missing", str(lonely)) == "0.0.0"


def test_wheel_dogfoods_its_own_resolver():
    import tollbooth

    # In the test env the wheel is installed, so metadata governs; either way it
    # must be a real version, never the unresolved sentinel.
    assert tollbooth.__version__ not in ("", "0.0.0")
