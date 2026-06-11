"""Generic in-memory session cache with TTL expiry.

Operators store domain-specific session objects (OAuth clients, API wrappers)
keyed by Horizon user_id. The cache handles TTL expiry and integrates with
OperatorRuntime's Neon vault for cross-restart persistence.

Usage::

    from tollbooth.session_cache import SessionCache

    # In your operator module:
    sessions: SessionCache[MySession] = SessionCache(ttl_seconds=3600)

    sessions.set("user123", my_session)
    s = sessions.get("user123")  # None if expired
    sessions.clear("user123")
"""

from __future__ import annotations

import logging
import time
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SessionCache(Generic[T]):
    """TTL-based in-memory session cache.

    Type parameter T is the operator's session object (e.g., a dataclass
    holding an API client, credentials, etc.).
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int | None = None) -> None:
        """Create a cache.

        Args:
            ttl_seconds: How long an entry stays valid after its last ``set``.
            max_size: Optional hard cap on the number of live entries. When set,
                ``set`` evicts the least-recently-written entries to stay within
                the bound. ``None`` (default) is unbounded — TTL is the only
                bound, which suits small per-operator caches.
        """
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._entries: dict[str, tuple[T, float]] = {}

    def _sweep_expired(self) -> int:
        """Drop every expired entry. Returns the count removed.

        Cheap O(n) hygiene so a ``set``-heavy workload (where ``get`` never
        revisits old keys to lazily expire them) cannot accumulate stale
        entries unbounded.
        """
        now = time.time()
        expired = [
            key for key, (_, created_at) in self._entries.items()
            if (now - created_at) > self._ttl
        ]
        for key in expired:
            del self._entries[key]
        return len(expired)

    def get(self, key: str) -> T | None:
        """Return the session for *key*, or None if absent/expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        session, created_at = entry
        if (time.time() - created_at) > self._ttl:
            del self._entries[key]
            return None
        return session

    def set(self, key: str, session: T) -> T:
        """Store *session* under *key* with a fresh TTL. Returns *session*.

        Opportunistically sweeps expired entries first, then enforces
        ``max_size`` (if configured) by evicting the least-recently-written
        entries. Re-setting an existing key moves it to most-recent.
        """
        self._sweep_expired()
        # Move-to-end on re-set so eviction is least-recently-written, not
        # first-inserted (dict updates keep the original position otherwise).
        self._entries.pop(key, None)
        self._entries[key] = (session, time.time())
        if self._max_size is not None:
            while len(self._entries) > self._max_size:
                oldest_key = next(iter(self._entries))
                del self._entries[oldest_key]
        return session

    def clear(self, key: str) -> T | None:
        """Remove and return the session for *key*, or None if absent."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        return entry[0]

    def clear_all(self) -> int:
        """Remove all sessions. Returns the count of cleared entries."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def __len__(self) -> int:
        """Return the number of stored (possibly expired) entries."""
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        """Check if *key* has a non-expired session."""
        return self.get(key) is not None

    @property
    def ttl_seconds(self) -> int:
        return self._ttl
