"""Proven npub cache — channel-bound npub ownership proof.

The proof is bound to the MCP session (FastMCP session_id) that
completed the proof exchange. A different session must prove
independently — proof does not leak across channels.

Cache key: ``"{session_id}:{npub}"``.

On ``mark_proven``, the record is written to both the in-memory
cache and the Neon vault (encrypted by the operator's nsec).
On ``is_proven`` cache miss, the vault is checked before rejecting —
surviving serverless cold starts that wipe in-memory state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from tollbooth.session_cache import SessionCache

logger = logging.getLogger(__name__)

DEFAULT_PROVEN_TTL = 3600  # 1 hour

_VAULT_KEY_PREFIX = "proven_npub:"


def _cache_key(session_id: str, npub: str) -> str:
    return f"{session_id}:{npub}"


def _vault_key(session_id: str, npub: str) -> str:
    return f"{_VAULT_KEY_PREFIX}{session_id}:{npub}"


@dataclass(frozen=True)
class ProvenNpub:
    """Record that an npub owner proved ownership on a specific channel."""

    session_id: str
    npub: str
    verified_at: float
    expires_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> ProvenNpub:
        return cls(**json.loads(raw))


class ProvenNpubCache:
    """Channel-bound npub ownership cache with vault persistence.

    In-memory ``SessionCache`` for hot lookups.  On cache miss,
    falls back to the Neon vault (encrypted at rest) so proofs
    survive serverless cold starts.

    Keyed by ``(session_id, npub)`` — proof on one MCP session
    does not extend to another.

    Args:
        ttl_seconds: How long a proven npub stays valid (default 3600).
        vault: Optional NeonVault for durable storage.
    """

    def __init__(
        self, ttl_seconds: int = DEFAULT_PROVEN_TTL, vault: Any | None = None, **_: Any
    ) -> None:
        self._ttl = ttl_seconds
        self._cache: SessionCache[ProvenNpub] = SessionCache(ttl_seconds=ttl_seconds)
        self._vault = vault

    async def is_proven(self, session_id: str, npub: str) -> bool:
        """Check if an npub is proven on this session."""
        key = _cache_key(session_id, npub)
        record = self._cache.get(key)

        if record is not None:
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

        # In-memory miss — try vault restore
        if self._vault:
            record = await self._vault_fetch(session_id, npub)
            if record is not None and time.time() < record.expires_at:
                self._cache.set(key, record)
                remaining = record.expires_at - time.time()
                logger.info(
                    "Proof cache RESTORED from vault for session=%s npub=%s (%.0fs remaining)",
                    session_id[:16], npub[:20], remaining,
                )
                return True
            if record is not None:
                logger.info(
                    "Vault proof expired for session=%s npub=%s — cleaning up",
                    session_id[:16], npub[:20],
                )
                await self._vault_delete(session_id, npub)

        cached_keys = list(self._cache._entries.keys())
        logger.warning(
            "Proof cache MISS for session=%s npub=%s — "
            "key=%s not found. %d entries in cache: %s",
            session_id[:16], npub[:20], key[:40],
            len(cached_keys),
            [k[:40] for k in cached_keys],
        )
        return False

    async def mark_proven(self, session_id: str, npub: str) -> ProvenNpub:
        """Cache an npub as ownership-proven on this session.

        Writes to both in-memory cache and vault (if configured).

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

        if self._vault:
            await self._vault_store(record)

        return record

    def invalidate(self, session_id: str, npub: str) -> None:
        """Remove a proven npub from the cache."""
        self._cache.clear(_cache_key(session_id, npub))

    # -- Vault helpers --------------------------------------------------------

    async def _vault_store(self, record: ProvenNpub) -> None:
        try:
            encrypted = self._vault._encrypt(record.to_json())
            await self._vault.set_config(
                _vault_key(record.session_id, record.npub), encrypted,
            )
            logger.debug(
                "Proof persisted to vault for session=%s npub=%s",
                record.session_id[:12], record.npub[:20],
            )
        except Exception as exc:
            logger.warning("Vault store for proven npub failed (non-fatal): %s", exc)

    async def _vault_fetch(self, session_id: str, npub: str) -> ProvenNpub | None:
        try:
            raw = await self._vault.get_config(_vault_key(session_id, npub))
            if raw is None:
                return None
            decrypted = self._vault._decrypt(raw)
            return ProvenNpub.from_json(decrypted)
        except Exception as exc:
            logger.warning("Vault fetch for proven npub failed (non-fatal): %s", exc)
            return None

    async def _vault_delete(self, session_id: str, npub: str) -> None:
        try:
            await self._vault.set_config(_vault_key(session_id, npub), "")
        except Exception:
            pass  # best-effort cleanup
