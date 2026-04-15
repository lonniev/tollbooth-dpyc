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
        key = _cache_key(session_id, npub)
        record = self._cache.get(key)
        if record is None:
            cached_keys = list(self._cache._entries.keys())
            logger.warning(
                "Proof cache MISS for session=%s npub=%s — "
                "key=%s not found. %d entries in cache: %s",
                session_id[:16], npub[:20], key[:40],
                len(cached_keys),
                [k[:40] for k in cached_keys],
            )
            return False
        if time.time() > record.expires_at:
            logger.warning(
                "Proof cache EXPIRED for session=%s npub=%s — "
                "verified_at=%.0f expires_at=%.0f now=%.0f (%.0fs overdue)",
                session_id[:16], npub[:20],
                record.verified_at, record.expires_at,
                time.time(), time.time() - record.expires_at,
            )
            self._cache.clear(key)
            return False
        remaining = record.expires_at - time.time()
        logger.debug(
            "Proof cache HIT for session=%s npub=%s (%.0fs remaining)",
            session_id[:16], npub[:20], remaining,
        )
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
