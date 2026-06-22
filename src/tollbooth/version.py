"""Single, fleet-wide way for a service to report its own version.

The source of truth is always ``pyproject [project].version``. We read it from
installed distribution metadata when the package is pip-installed, and fall
back to reading the source ``pyproject.toml`` when it is not — e.g. FastMCP
Cloud / Horizon running a flat checkout without installing the project, where
``importlib.metadata`` raises ``PackageNotFoundError``.

Every operator and Authority calls this so version reporting is identical and
reliable across the fleet; there is no hand-maintained version constant to
drift, and ``/release`` bumping pyproject is the only lever.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["resolve_service_version"]


def resolve_service_version(
    dist_name: str, source_hint: str | os.PathLike[str] | None = None
) -> str:
    """Resolve a service's own version from pyproject ``[project].version``.

    Args:
        dist_name: The distribution name (pyproject ``[project].name``), used
            for the installed-metadata lookup.
        source_hint: Typically the caller's ``__file__``. When the package is
            not installed, the search for ``pyproject.toml`` walks up from here
            (or from the current working directory if omitted).

    Returns:
        The version string, or ``"0.0.0"`` if it cannot be determined.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(dist_name)
    except PackageNotFoundError:
        pass

    import tomllib

    start = Path(source_hint).resolve() if source_hint else Path.cwd().resolve()
    # A file hint contributes its parent dirs; a dir hint contributes itself too.
    search_dirs = list(start.parents) if start.is_file() else [start, *start.parents]
    for parent in search_dirs:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            try:
                with pyproject.open("rb") as fh:
                    return str(tomllib.load(fh)["project"]["version"])
            except (KeyError, OSError, tomllib.TOMLDecodeError):
                break
    return "0.0.0"
