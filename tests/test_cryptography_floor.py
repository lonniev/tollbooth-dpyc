"""Guard the cryptography security floor against GHSA-m2h6-j472-rp4c.

The wheel declares an explicit floor so fresh installs cannot land a
version whose X.509 name-constraint verifier accepts over-broad wildcard
DNS SANs (fixed in cryptography 49.0.0). The lockfile must resolve at or
above that floor too — a loose floor with a pinned vulnerable lock is
still a ship of the CVE.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
# GHSA-m2h6-j472-rp4c / CVE-2026-69248: fixed in cryptography 49.0.0.
_FIXED = Version("49.0.0")
_VULNERABLE = Version("48.0.0")


def _cryptography_requirement() -> Requirement:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    matches = [d for d in deps if d.startswith("cryptography")]
    assert matches, "cryptography must be an explicit direct dependency"
    assert len(matches) == 1, f"expected one cryptography pin, found {matches!r}"
    return Requirement(matches[0])


def _locked_cryptography_version() -> Version:
    text = (REPO_ROOT / "uv.lock").read_text()
    match = re.search(
        r'(?m)^name = "cryptography"\nversion = "([^"]+)"',
        text,
    )
    assert match, "uv.lock must pin a cryptography package version"
    return Version(match.group(1))


def test_cryptography_floor_rejects_ghsa_m2h6_j472_rp4c_affected_range():
    req = _cryptography_requirement()
    assert _VULNERABLE not in req.specifier, (
        f"cryptography floor {req} still admits vulnerable {_VULNERABLE} "
        f"(GHSA-m2h6-j472-rp4c is fixed in {_FIXED})"
    )
    assert _FIXED in req.specifier, (
        f"cryptography floor {req} must admit the patched release {_FIXED}"
    )


def test_uv_lock_resolves_cryptography_at_or_above_fixed_release():
    locked = _locked_cryptography_version()
    assert locked >= _FIXED, (
        f"uv.lock pins cryptography=={locked}, which is below the "
        f"GHSA-m2h6-j472-rp4c fix floor {_FIXED}"
    )
