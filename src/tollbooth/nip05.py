"""NIP-05 address resolution: ``name@domain`` → npub.

Protocol (NIP-05):
  1. Parse ``name@domain``
  2. GET ``https://{domain}/.well-known/nostr.json?name={name}``
  3. Read ``names[name.lower()]`` as a 64-char hex pubkey
  4. Encode as bech32 ``npub1...``

Uses httpx (already a project dep). Mirrors the LNURL address-parse style
in ``tollbooth.lnurl`` so Lightning and Nostr identifiers share one shape.

The caller names the domain, and tool arguments are adversarial, so the
lookup is fenced:

- Redirects are refused, as NIP-05 requires of fetchers.
- The domain must be a public DNS hostname: no IP literal, no port, no
  ``localhost`` or private-use suffix, and every address it resolves to must
  be globally routable. This keeps the server from probing its own network.
- A resolved npub is cached for ``CACHE_TTL_S`` only, so a remapped or
  revoked address takes effect within minutes, not at the next restart.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import threading
import time

import httpx

logger = logging.getLogger(__name__)

# A NIP-05 local-part: letters, digits, underscore, hyphen, period.
# Domains are more permissive (labels + dots); we only refuse empty / path-like.
_NAME_RE = re.compile(r"^[a-z0-9_\-\.]+$", re.IGNORECASE)
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
# One DNS label: letters, digits, inner hyphens, at most 63 characters.
_LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")
# Suffixes that name private or local networks, never a public NIP-05 host.
_PRIVATE_SUFFIXES = (
    "localhost", "local", "localdomain", "internal", "intranet", "lan",
    "home", "corp", "private", "home.arpa", "in-addr.arpa", "ip6.arpa",
)

#: Seconds a resolved npub is trusted before it is looked up again.
CACHE_TTL_S = 300.0
_CACHE_MAX = 256
_DEFAULT_TIMEOUT_S = 5.0

_cache: dict[tuple[str, str], tuple[str, float]] = {}
_cache_lock = threading.Lock()


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
    domain = domain.lower().rstrip(".")
    _check_public_hostname(domain, address)
    return name, domain


def _check_public_hostname(domain: str, address: str) -> None:
    """Refuse any domain that is not a plain public DNS hostname."""
    if len(domain) > 253:
        raise Nip05ResolutionError(f"Invalid NIP-05 address (domain too long): {address!r}")
    try:
        ipaddress.ip_address(domain.strip("[]"))
    except ValueError:
        pass
    else:
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (an IP address is not a NIP-05 domain): {address!r}"
        )
    labels = domain.split(".")
    if len(labels) < 2 or not all(_LABEL_RE.match(label) for label in labels):
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (domain must be a hostname, no port): {address!r}"
        )
    if labels[-1].isdigit():
        raise Nip05ResolutionError(f"Invalid NIP-05 address (bad domain): {address!r}")
    if any(domain == s or domain.endswith("." + s) for s in _PRIVATE_SUFFIXES):
        raise Nip05ResolutionError(
            f"Invalid NIP-05 address (private-network domain): {address!r}"
        )


def _check_resolves_public(domain: str) -> None:
    """Refuse a domain whose DNS answers include a non-public address."""
    try:
        infos = socket.getaddrinfo(domain, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise Nip05ResolutionError(f"NIP-05 domain does not resolve: {domain}") from exc
    for info in infos:
        ip = ipaddress.ip_address(str(info[4][0]).split("%", 1)[0])
        if not ip.is_global:
            raise Nip05ResolutionError(
                f"NIP-05 domain resolves to a non-public address: {domain}"
            )


def resolve_nip05(address: str, *, timeout: float = _DEFAULT_TIMEOUT_S) -> str:
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
    # share one lookup.
    name, domain = _parse_nip05_address(address)
    key = (name.lower(), domain)
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[1] > now:
            return hit[0]
    npub = _lookup(key[0], domain, timeout)
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            for k in [k for k, (_, exp) in _cache.items() if exp <= now] or [next(iter(_cache))]:
                _cache.pop(k, None)
        _cache[key] = (npub, now + CACHE_TTL_S)
    return npub


def clear_nip05_cache() -> None:
    """Drop cached NIP-05 lookups (tests, or after a known remapping)."""
    with _cache_lock:
        _cache.clear()


def _not_a_nip05_domain(name: str, domain: str) -> str:
    """The domain answered, but with something other than a NIP-05 names map."""
    return (
        f"{domain} doesn't publish NIP-05 names. "
        f"Check the part of {name}@{domain} after the @."
    )


def _lookup(name: str, domain: str, timeout: float) -> str:
    """Fetch and validate one NIP-05 mapping (name already lowercased)."""
    _check_resolves_public(domain)
    url = f"https://{domain}/.well-known/nostr.json"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            resp = client.get(url, params={"name": name})
    except httpx.HTTPError as exc:
        raise Nip05ResolutionError(
            f"NIP-05 lookup failed for {name}@{domain}: {type(exc).__name__}"
        ) from exc

    if 300 <= resp.status_code < 400:
        raise Nip05ResolutionError(
            f"NIP-05 endpoint redirected; NIP-05 forbids following redirects: {url}"
        )
    if resp.status_code != 200:
        raise Nip05ResolutionError(
            f"{domain} doesn't publish NIP-05 names (HTTP {resp.status_code}). "
            f"Check the part of {name}@{domain} after the @."
        )

    try:
        data = resp.json()
    except Exception as exc:
        raise Nip05ResolutionError(_not_a_nip05_domain(name, domain)) from exc

    names = data.get("names") if isinstance(data, dict) else None
    if not isinstance(names, dict):
        raise Nip05ResolutionError(_not_a_nip05_domain(name, domain))

    # NIP-05: name comparison is case-insensitive; servers often store lowercase.
    hex_pk = names.get(name)
    if not isinstance(hex_pk, str) or not _HEX64_RE.match(hex_pk):
        raise Nip05ResolutionError(
            f"{domain} has no NIP-05 name {name!r}. Check the part of "
            f"{name}@{domain} before the @."
        )

    try:
        from pynostr.key import PublicKey  # type: ignore[import-untyped]

        return PublicKey(bytes.fromhex(hex_pk.lower())).bech32()
    except Exception as exc:
        raise Nip05ResolutionError(
            f"NIP-05 pubkey for {name}@{domain} is not a valid key: {exc}"
        ) from exc
