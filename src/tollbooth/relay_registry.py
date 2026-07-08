"""DPYC relay registry — the single source of truth for Nostr relays.

Every Tollbooth server draws its relay set from ``relays.json`` in the
``dpyc-community`` repository, fetched over GitHub raw HTTPS and cached in
process with a 3-day TTL. There is NO hardcoded relay list anywhere else —
change the set in ``dpyc-community/relays.json`` and the whole federation
follows on its next cache refresh.

Resolution semantics (see ``dpyc-community/RELAYS.md``):

- Fresh cache (< 3 days) → returned directly, no fetch.
- Stale or absent cache → refetch.
- Refresh fails but a previously-fetched copy exists → serve that
  last-known-good copy (stale-if-error), with a short backoff so a GitHub
  outage does not cause a fetch on every single call.
- Cold start with no cache and an unreachable registry → ``RelayRegistryError``
  (fail closed — the same trust model the membership registry uses).

This module is synchronous by design: every relay consumer in the wheel
(Secure Courier, bootstrap, profile, audit) is synchronous (blocking
websockets / thread pools), so a sync ``httpx.Client`` slots in without
forcing async through those paths.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_RELAYS_URL = (
    "https://raw.githubusercontent.com/lonniev/dpyc-community/main/relays.json"
)

# 3 days — clients fetch only when their cache is stale.
DEFAULT_RELAY_TTL_SECONDS = 3 * 24 * 60 * 60  # 259200

# After a failed refresh, wait this long before hitting the network again
# (serving the stale set in the meantime) so an outage can't cause a fetch
# on every call.
_FAIL_BACKOFF_SECONDS = 60.0


class RelayRegistryError(Exception):
    """Raised when relays cannot be resolved and no cached set exists (fail closed)."""


class RelayRegistry:
    """Cached fetch of the DPYC supported-relay set via HTTP.

    Fetches ``relays.json`` from the registry URL and caches the resolved,
    ordered (primary-first) relay URLs with a monotonic-clock TTL.
    """

    def __init__(
        self,
        url: str = DEFAULT_RELAYS_URL,
        cache_ttl_seconds: int = DEFAULT_RELAY_TTL_SECONDS,
    ) -> None:
        self._url = url
        self._ttl = cache_ttl_seconds
        self._cache: list[str] | None = None
        self._cache_time: float = 0.0
        self._retry_after: float = 0.0

    def relays(self) -> list[str]:
        """Return the ordered relay URLs (primary first), refreshing if stale.

        Raises ``RelayRegistryError`` only on a cold cache with an unreachable
        registry; otherwise a stale set is served rather than failing.
        """
        now = time.monotonic()

        if self._cache is not None and (now - self._cache_time) < self._ttl:
            return self._cache

        # Stale cache but still inside the post-failure backoff window: serve
        # the last-known-good set without hammering the network.
        if self._cache is not None and now < self._retry_after:
            return self._cache

        try:
            relays = self._do_fetch()
        except Exception as e:
            if self._cache is not None:
                logger.warning(
                    "Relay registry refresh failed (%s); serving last-known-good "
                    "set of %d relay(s).",
                    e,
                    len(self._cache),
                )
                self._retry_after = now + _FAIL_BACKOFF_SECONDS
                return self._cache
            raise RelayRegistryError(
                f"Relay registry unreachable and no cached relays: {e}"
            ) from e

        self._cache = relays
        self._cache_time = now
        self._retry_after = 0.0
        return relays

    def _do_fetch(self) -> list[str]:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self._url)
            resp.raise_for_status()
            data: Any = resp.json()

        if not isinstance(data, dict) or not isinstance(data.get("relays"), list):
            raise RelayRegistryError("relays.json missing a 'relays' list.")

        primary: list[str] = []
        rest: list[str] = []
        for entry in data["relays"]:
            if isinstance(entry, dict):
                url = entry.get("url")
                is_primary = bool(entry.get("primary"))
            elif isinstance(entry, str):
                url = entry
                is_primary = False
            else:
                continue
            if not isinstance(url, str) or not url.startswith("wss://"):
                continue
            (primary if is_primary else rest).append(url)

        # Primary first, then array order; de-duplicate while preserving order.
        ordered: list[str] = []
        for url in primary + rest:
            if url not in ordered:
                ordered.append(url)

        if not ordered:
            raise RelayRegistryError("relays.json contained no valid wss:// relays.")
        return ordered

    def invalidate(self) -> None:
        """Force the next call to re-fetch."""
        self._cache = None
        self._cache_time = 0.0
        self._retry_after = 0.0


# Process-global singleton so every consumer (courier, bootstrap, profile,
# audit) shares one 3-day cache — a single fetch serves the whole process.
_registry = RelayRegistry()


def get_relays() -> list[str]:
    """Return the DPYC supported relay URLs (primary first) from the registry.

    Cached process-wide with a 3-day TTL. Raises ``RelayRegistryError`` only
    on a cold cache with an unreachable registry.
    """
    return _registry.relays()


def invalidate_relays_cache() -> None:
    """Force the next ``get_relays()`` to re-fetch (mainly for tests)."""
    _registry.invalidate()
