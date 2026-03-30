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

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[T, float]] = {}

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
        """Store *session* under *key* with a fresh TTL. Returns *session*."""
        self._entries[key] = (session, time.time())
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
