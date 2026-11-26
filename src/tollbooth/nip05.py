"""NIP-05 address resolution: ``name@domain`` → npub.

Protocol (NIP-05):
  1. Parse ``name@domain``
  2. GET ``https://{domain}/.well-known/nostr.json?name={name}``
  3. Read ``names[name.lower()]`` as a 64-char hex pubkey
  4. Encode as bech32 ``npub1...``

Uses httpx (already a project dep). Mirrors the LNURL address-parse style
in ``tollbooth.lnurl`` so Lightning and Nostr identifiers share one shape.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)

# A NIP-05 local-part: letters, digits, underscore, hyphen, period.
# Domains are more permissive (labels + dots); we only refuse empty / path-like.
_NAME_RE = re.compile(r"^[a-z0-9_\-\.]+$", re.IGNORECASE)
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class Nip05ResolutionError(Exception):
    """Failed to resolve a NIP-05 address to an npub."""


def looks_like_nip05(value: str) -> bool:
    """True when *value* is shaped like ``name@domain`` (not an npub)."""
    if not value or value.startswith("npub1") or "@" not in value:
        return False
    parts = value.split("@")
    if len(parts) != 2:
        return False
    name, domain = parts
    return bool(name) and bool(domain) and "." in domain and "/" not in domain


def _parse_nip05_address(address: str) -> tuple[str, str]:
    """Parse ``name@domain`` into (name, domain). Raises on invalid format."""
    if "@" not in address:
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (missing @): {address!r}"
        )
    parts = address.split("@")
    if len(parts) != 2:
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (multiple @): {address!r}"
        )
    name, domain = parts[0].strip(), parts[1].strip()
    if not name:
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (empty name): {address!r}"
        )
    if not domain or "." not in domain or "/" in domain:
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (bad domain): {address!r}"
        )
    if not _NAME_RE.match(name):
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (bad name): {address!r}"
        )
    return name, domain.lower()


def resolve_nip05(address: str, *, timeout: float = 10.0) -> str:
    """Resolve a NIP-05 ``name@domain`` address to a bech32 npub.

    Args:
        address: NIP-05 identifier in ``name@domain`` form.
        timeout: HTTP timeout in seconds.

    Returns:
        Bech32 ``npub1...`` for the hex pubkey published under that name.

    Raises:
        Nip05ResolutionError: On parse, network, or mapping failure.
    """
    # Normalize before the cache key so Alice@Example.com and alice@example.com
    # share one lookup. timeout is intentionally not part of the key — the
    # default path is what validation + resolve share within one tool call.
    name, domain = _parse_nip05_address(address)
    return _resolve_nip05_cached(name.lower(), domain, timeout)


def clear_nip05_cache() -> None:
    """Drop cached NIP-05 lookups (tests, or after a known remapping)."""
    _resolve_nip05_cached.cache_clear()


@lru_cache(maxsize=256)
def _resolve_nip05_cached(name: str, domain: str, timeout: float) -> str:
    """Cached core of :func:`resolve_nip05` (name already lowercased)."""
    url = f"https://{domain}/.well-known/nostr.json"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, params={"name": name})
    except httpx.HTTPError as exc:
        raise Nip05ResolutionError(
            f"NIP-05 lookup failed for {name}@{domain}: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise Nip05ResolutionError(
            f"NIP-05 endpoint returned HTTP {resp.status_code}: {url}"
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise Nip05ResolutionError(
            f"NIP-05 endpoint returned non-JSON for {name}@{domain}"
        ) from exc

    if not isinstance(data, dict):
        raise Nip05ResolutionError(
            f"NIP-05 endpoint returned unexpected payload for {name}@{domain}"
        )

    names = data.get("names")
    if not isinstance(names, dict):
        raise Nip05ResolutionError(
            f"NIP-05 response missing 'names' map for {name}@{domain}"
        )

    # NIP-05: name comparison is case-insensitive; servers often store lowercase.
    hex_pk = names.get(name)
    if not isinstance(hex_pk, str) or not _HEX64_RE.match(hex_pk):
        raise Nip05ResolutionError(
            f"NIP-05 name {name!r} not found (or not a hex pubkey) at {domain}"
        )

    try:
        from pynostr.key import PublicKey  # type: ignore[import-untyped]

        return PublicKey(bytes.fromhex(hex_pk.lower())).bech32()
    except Exception as exc:
        raise Nip05ResolutionError(
            f"NIP-05 pubkey for {name}@{domain} is not a valid key: {exc}"
        ) from exc
