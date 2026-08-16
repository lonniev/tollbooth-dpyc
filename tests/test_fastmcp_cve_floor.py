"""fastmcp keyring extra must not admit CVE-affected releases (issue #230).

GHSA-m8x7-r2rg-vh5g / GHSA-rww4-4w9c-7733 / GHSA-vv7q-7jx5-f767 are fixed in
fastmcp 3.2.0. The keyring optional-extra is the SDK's published floor for
that dependency; a looser pin (e.g. ``>=3.0``) lets consumers resolve 3.1.x
and stay exposed. This test reads the declared specifier — it does not
install fastmcp — so the gate holds even when the keyring extra is absent
from the test env.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 not in requires-python
    import tomli as tomllib  # type: ignore[no-redef]

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging ships with the test env via pip/uv
    pytest.skip("packaging is required for specifier checks", allow_module_level=True)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# First release that closes all three GHSAs named in #230.
_PATCHED_FLOOR = Version("3.2.0")
_VULNERABLE_SAMPLES = ("3.0.0", "3.1.0", "3.1.1")
_SAFE_SAMPLES = ("3.2.0", "3.4.4", "3.4.7")


def _keyring_fastmcp_spec() -> SpecifierSet:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]["keyring"]
    for entry in extras:
        if entry == "fastmcp" or entry.startswith("fastmcp"):
            # ``fastmcp>=3.2.0`` → SpecifierSet(">=3.2.0"); bare name is open-ended.
            _, _, spec = entry.partition("fastmcp")
            return SpecifierSet(spec or ">=0")
    raise AssertionError("keyring extra does not declare a fastmcp dependency")


def test_keyring_extra_blocks_cve_affected_fastmcp_releases() -> None:
    spec = _keyring_fastmcp_spec()
    for raw in _VULNERABLE_SAMPLES:
        assert Version(raw) not in spec, (
            f"keyring fastmcp pin {spec!s} still admits vulnerable {raw} "
            f"(CVEs fixed only in >= {_PATCHED_FLOOR})"
        )
    for raw in _SAFE_SAMPLES:
        assert Version(raw) in spec, (
            f"keyring fastmcp pin {spec!s} unexpectedly rejects patched {raw}"
        )


def test_uv_lock_metadata_matches_pyproject_fastmcp_floor() -> None:
    """Keep the committed lock's requires-dist in step with pyproject.toml."""
    lock = (_REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    # The tollbooth-dpyc package metadata block records the published floor.
    match = re.search(
        r'\{ name = "fastmcp", marker = "extra == \'keyring\'", specifier = "([^"]+)" \}',
        lock,
    )
    assert match is not None, "uv.lock missing keyring fastmcp requires-dist entry"
    lock_spec = SpecifierSet(match.group(1))
    py_spec = _keyring_fastmcp_spec()
    assert lock_spec == py_spec, (
        f"uv.lock keyring fastmcp specifier {lock_spec!s} != pyproject {py_spec!s}"
    )
