"""DPYC relay registry — the Nostr relay set, sourced from the Oracle.

Every Tollbooth server draws its relay set from the DPYC Oracle
(``get_relays`` over MCP), cached in process with a 3-day TTL. Operators are
nsec-only and must never read GitHub directly, so the relay list — like every
other community fact — comes from the Oracle, the one component allowed to read
``dpyc-community``. The authoritative data still lives in
``dpyc-community/relays.json``; the Oracle serves it.

Resolution semantics:

- Fresh cache (< 3 days) → returned directly, no fetch.
- Stale or absent cache → refetch from the Oracle.
- Refresh fails but a previously-fetched (or seeded) copy exists → serve that
  last-known-good copy (stale-if-error), with a short backoff so an Oracle
  outage does not cause a fetch on every call.
- Cold start with no cache and an unreachable Oracle → ``RelayRegistryError``
  (fail closed).

The bootstrap path fetches the relay set from the Oracle once (async) and
**seeds** it here via ``seed_relays`` so later synchronous consumers (Secure
Courier, profile, audit) read the warm cache without another round-trip.

This module is synchronous by design: every relay consumer in the wheel is
synchronous (blocking websockets / thread pools). On a cache-miss it drives the
async Oracle client via ``asyncio.run`` — safe because these callers run
outside an event loop.
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)

# 3 days — clients fetch only when their cache is stale.
DEFAULT_RELAY_TTL_SECONDS = 3 * 24 * 60 * 60  # 259200

# After a failed refresh, wait this long before hitting the network again
# (serving the stale set in the meantime) so an outage can't cause a fetch
# on every call.
_FAIL_BACKOFF_SECONDS = 60.0


class RelayRegistryError(Exception):
    """Raised when relays cannot be resolved and no cached set exists (fail closed)."""


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

        Raises ``RelayRegistryError`` only on a cold cache with an unreachable
        Oracle; otherwise a stale set is served rather than failing.
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

    Cached process-wide with a 3-day TTL. Raises ``RelayRegistryError`` only
    on a cold cache with an unreachable Oracle.
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
        logger.warning("Relay pre-seed from Oracle failed (%s); cache left cold.", e)


def invalidate_relays_cache() -> None:
    """Force the next ``get_relays()`` to re-fetch (mainly for tests)."""
    _registry.invalidate()
