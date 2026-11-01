"""DPYC relay registry — the Nostr relay set, sourced from the Oracle.

Every Tollbooth server draws its relay set from the DPYC Oracle
(``get_relays`` over MCP), cached in process. The authoritative data lives in
``dpyc-community/relays.json``; the Oracle serves it. When the Oracle is silent
and nothing is cached, that same public file is read directly: the only time an
operator touches GitHub, and only for addresses, which grant no trust. Every
event read from a relay is still checked for its author and signature.

Resolution semantics:

- Fresh cache (< 3 days) → returned directly, no fetch.
- Stale or absent cache → refetch from the Oracle.
- Refresh fails but a previously-fetched (or seeded) copy exists → serve that
  last-known-good copy (stale-if-error), with a short backoff so an Oracle
  outage does not cause a fetch on every call.
- Cold start with no cache and an unreachable Oracle → the published list.
- That is unreachable too → ``RelayRegistryError`` (fail closed).

The bootstrap path fetches the relay set from the Oracle once (async) and
**seeds** it here via ``seed_relays`` so later synchronous consumers (Secure
Courier, profile, audit) read the warm cache without another round-trip.

This module is synchronous by design: every relay consumer in the wheel is
synchronous (blocking websockets / thread pools) and runs on a worker thread,
never on the event loop. On a cache-miss it drives the async Oracle client via
``asyncio.run`` — safe on a worker thread, which has no running loop.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# 3 days — clients fetch only when their cache is stale.
# One hour. The set itself changes rarely, but its ORDER now moves with
# relay outages — a three-day cache meant the Oracle could demote a dead
# relay and no operator would notice until Thursday. A report that
# actually changes the order also invalidates this cache immediately
# (see relay_reports.flush_relay_failures); the TTL is the backstop.
DEFAULT_RELAY_TTL_SECONDS = 60 * 60  # 259200

# After a failed refresh, wait this long before hitting the network again
# (serving the stale set in the meantime) so an outage can't cause a fetch
# on every call.
_FAIL_BACKOFF_SECONDS = 60.0


_PUBLISHED_RELAYS_URL = (
    "https://raw.githubusercontent.com/lonniev/dpyc-community/main/relays.json"
)
_PUBLISHED_RELAYS_TIMEOUT_SECONDS = 5.0


class RelayRegistryError(Exception):
    """Raised when relays cannot be resolved and no cached set exists (fail closed)."""


def fetch_published_relays() -> list[str]:
    """Read ``relays.json`` directly, primary first, for when the Oracle is silent."""
    response = httpx.get(_PUBLISHED_RELAYS_URL, timeout=_PUBLISHED_RELAYS_TIMEOUT_SECONDS)
    response.raise_for_status()
    entries = response.json().get("relays")
    if not isinstance(entries, list):
        raise RelayRegistryError("Published relay list has no 'relays' array.")
    primary: list[str] = []
    rest: list[str] = []
    for entry in entries:
        url = entry.get("url") if isinstance(entry, dict) else entry
        if isinstance(url, str) and url.startswith("wss://"):
            (primary if isinstance(entry, dict) and entry.get("primary") else rest).append(url)
    relays = list(dict.fromkeys(primary + rest))
    if not relays:
        raise RelayRegistryError("Published relay list contained no wss:// relays.")
    return relays


class RelayRegistry:
    """Cached relay set sourced from the Oracle's ``get_relays`` tool."""

    def __init__(self, cache_ttl_seconds: int = DEFAULT_RELAY_TTL_SECONDS) -> None:
        self._ttl = cache_ttl_seconds
        self._cache: list[str] | None = None
        self._cache_time: float = 0.0
        self._retry_after: float = 0.0

    def seed(self, relays: list[str]) -> None:
        """Populate the cache from an already-fetched relay set.

        Called by the async bootstrap path after it fetches relays from the
        Oracle, so synchronous consumers reuse the set instead of re-fetching.
        """
        if relays:
            self._cache = list(relays)
            self._cache_time = time.monotonic()
            self._retry_after = 0.0

    def is_warm(self) -> bool:
        """True when a non-stale relay set is cached, so no fetch is needed."""
        return self._cache is not None and (time.monotonic() - self._cache_time) < self._ttl

    def relays(self) -> list[str]:
        """Return the ordered relay URLs (primary first), refreshing if stale.

        Raises ``RelayRegistryError`` only on a cold cache with both the Oracle
        and the published list unreachable; otherwise a stale set is served.
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
        except Exception as e:  # noqa: BLE001 — any failure means try the next source
            if self._cache is not None:
                logger.warning(
                    "Relay registry refresh failed (%s); serving last-known-good "
                    "set of %d relay(s).",
                    e,
                    len(self._cache),
                )
                self._retry_after = now + _FAIL_BACKOFF_SECONDS
                return self._cache
            try:
                relays = fetch_published_relays()
            except Exception as published_error:  # noqa: BLE001
                raise RelayRegistryError(
                    f"Relay registry unreachable and no cached relays: {e}; "
                    f"published list: {published_error}"
                ) from e
            logger.warning("Oracle silent (%s); using the published relay list.", e)

        self._cache = relays
        self._cache_time = now
        self._retry_after = 0.0
        return relays

    def _do_fetch(self) -> list[str]:
        """Fetch the relay set from the Oracle (synchronous bridge)."""
        from tollbooth.oracle_client import default_oracle_client

        try:
            relays = asyncio.run(default_oracle_client().get_relays())
        except RuntimeError as e:
            # Only happens if called from within a running event loop — the
            # async bootstrap path seeds instead of calling here, so this is a
            # misuse, surfaced clearly rather than silently.
            raise RelayRegistryError(
                f"get_relays() called from an async context ({e}); "
                "seed the registry or await the Oracle client directly."
            ) from e

        if not relays:
            raise RelayRegistryError("Oracle returned no relays.")
        return relays

    def invalidate(self) -> None:
        """Force the next call to re-fetch."""
        self._cache = None
        self._cache_time = 0.0
        self._retry_after = 0.0


# Process-global singleton so every consumer (courier, bootstrap, profile,
# audit) shares one 3-day cache — a single fetch (or bootstrap seed) serves the
# whole process.
_registry = RelayRegistry()


def get_relays() -> list[str]:
    """Return the DPYC supported relay URLs (primary first) from the Oracle.

    Cached process-wide. Raises ``RelayRegistryError`` only on a cold cache with
    both the Oracle and the published list unreachable.
    """
    return _registry.relays()


def seed_relays(relays: list[str]) -> None:
    """Seed the process-wide relay cache from an already-fetched set."""
    _registry.seed(relays)


async def ensure_relays_seeded() -> None:
    """Warm the process-wide relay cache from the Oracle, from an async context.

    The async counterpart of the synchronous :func:`get_relays`. Every actor that
    starts *inside* an event loop must call this once at startup so the synchronous
    relay consumers (Secure Courier, profile, audit) read a warm cache and never
    reach the ``asyncio.run`` bridge in ``RelayRegistry._do_fetch`` — which raises
    inside a running loop.

    Self-provisioning actors (Authorities, ``vault_source='env'``) skip the
    certified-operator bootstrap that would otherwise seed relays, so this is the
    seam that keeps their Secure Courier working. Idempotent (a no-op when the cache
    is already warm) and best-effort: an Oracle blip leaves the cache cold and a
    later synchronous consumer surfaces a clear error, rather than crashing here.
    """
    if _registry.is_warm():
        return
    try:
        from tollbooth.oracle_client import default_oracle_client

        relays = await default_oracle_client().get_relays()
        _registry.seed(relays)
    except Exception as e:  # noqa: BLE001 — best-effort warm; the sync path surfaces a cold miss
        logger.warning("Relay pre-seed from Oracle failed (%s); trying the published list.", e)
        try:
            _registry.seed(await asyncio.to_thread(fetch_published_relays))
        except Exception as published_error:  # noqa: BLE001
            logger.warning("Published relay list unavailable (%s); cache left cold.", published_error)


def invalidate_relays_cache() -> None:
    """Force the next ``get_relays()`` to re-fetch (mainly for tests)."""
    _registry.invalidate()
