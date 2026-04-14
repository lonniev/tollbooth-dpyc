"""Proven npub cache — channel-bound npub ownership proof.

The proof is bound to the MCP session (FastMCP session_id) that
completed the proof exchange. A different session must prove
independently — proof does not leak across channels.

Cache key: ``"{session_id}:{npub}"``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from tollbooth.session_cache import SessionCache

logger = logging.getLogger(__name__)

DEFAULT_PROVEN_TTL = 3600  # 1 hour


def _cache_key(session_id: str, npub: str) -> str:
    return f"{session_id}:{npub}"


@dataclass(frozen=True)
class ProvenNpub:
    """Record that an npub owner proved ownership on a specific channel."""

    session_id: str
    npub: str
    verified_at: float
    expires_at: float


class ProvenNpubCache:
    """In-memory channel-bound npub ownership cache.

    Keyed by ``(session_id, npub)`` — proof on one MCP session
    does not extend to another.

    Args:
        ttl_seconds: How long a proven npub stays valid (default 3600).
    """

    def __init__(self, ttl_seconds: int = DEFAULT_PROVEN_TTL, **_: Any) -> None:
        self._ttl = ttl_seconds
        self._cache: SessionCache[ProvenNpub] = SessionCache(ttl_seconds=ttl_seconds)

    async def is_proven(self, session_id: str, npub: str) -> bool:
        """Check if an npub is proven on this session."""
        record = self._cache.get(_cache_key(session_id, npub))
        if record is None:
            return False
        if time.time() > record.expires_at:
            self._cache.clear(_cache_key(session_id, npub))
            return False
        return True

    async def mark_proven(self, session_id: str, npub: str) -> ProvenNpub:
        """Cache an npub as ownership-proven on this session.

        Args:
            session_id: The MCP transport session ID.
            npub: The patron's npub (bech32).

        Returns:
            The cached ``ProvenNpub`` record.
        """
        now = time.time()
        record = ProvenNpub(
            session_id=session_id,
            npub=npub,
            verified_at=now,
            expires_at=now + self._ttl,
        )
        self._cache.set(_cache_key(session_id, npub), record)
        logger.info(
            "Cached proven npub %s on session %s (expires in %ds)",
            npub[:20], session_id[:12], self._ttl,
        )
        return record

    def invalidate(self, session_id: str, npub: str) -> None:
        """Remove a proven npub from the cache."""
        self._cache.clear(_cache_key(session_id, npub))
