"""Proven npub cache — caches Schnorr-verified npub ownership.

Stateless AI agents (claude.ai, etc.) cannot generate kind-27235 proofs
because they have no nsec and no crypto runtime. This module lets the
operator verify a patron's proof once (delivered via Nostr DM) and cache
the result so subsequent ``debit_or_deny`` calls skip inline proof.

Usage::

    cache = ProvenNpubCache(runtime=runtime, ttl_seconds=3600)

    # Verify and cache a proof received via DM
    record = await cache.prove(npub, proof_json)

    # Check if an npub is already proven (memory, then vault)
    if await cache.is_proven(npub):
        ...  # skip inline proof requirement

    # Invalidate (e.g., on forget_credentials)
    cache.invalidate(npub)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from tollbooth.patron_session import PatronSessionCache

logger = logging.getLogger(__name__)

DEFAULT_PROVEN_TTL = 3600  # 1 hour


@dataclass(frozen=True)
class ProvenNpub:
    """Record that an npub owner replied via signed Nostr DM."""

    npub: str
    verified_at: float
    expires_at: float


class ProvenNpubCache:
    """In-memory + vault cache of proven npub ownership.

    Uses ``PatronSessionCache[ProvenNpub]`` for memory + Neon vault
    persistence, so proven npubs survive cold restarts.

    Args:
        runtime: The ``OperatorRuntime`` instance.
        ttl_seconds: How long a proven npub stays valid (default 3600).
    """

    def __init__(
        self,
        runtime: Any,
        ttl_seconds: int = DEFAULT_PROVEN_TTL,
    ) -> None:
        self._ttl = ttl_seconds
        self._cache: PatronSessionCache[ProvenNpub] = PatronSessionCache(
            runtime=runtime,
            service="npub_ownership",
            restore=self._restore,
            ttl_seconds=ttl_seconds,
        )

    async def _restore(self, creds: dict[str, Any]) -> ProvenNpub | None:
        """Rebuild a ProvenNpub from vault credentials."""
        try:
            record = ProvenNpub(
                npub=creds["npub"],
                verified_at=float(creds["verified_at"]),
                expires_at=float(creds["expires_at"]),
            )
        except (KeyError, ValueError, TypeError):
            return None

        if time.time() > record.expires_at:
            return None

        return record

    async def is_proven(self, npub: str) -> bool:
        """Check if an npub has a valid cached ownership proof."""
        record = await self._cache.get_or_restore(npub)
        if record is None:
            return False
        if time.time() > record.expires_at:
            self._cache.invalidate(npub)
            return False
        return True

    async def mark_proven(self, npub: str) -> ProvenNpub:
        """Cache an npub as ownership-proven.

        Called after Secure Courier confirms a signed DM from the patron.
        The DM itself is the proof — the patron's nsec signed it.

        Args:
            npub: The patron's npub (bech32).

        Returns:
            The cached ``ProvenNpub`` record.
        """
        now = time.time()
        record = ProvenNpub(
            npub=npub,
            verified_at=now,
            expires_at=now + self._ttl,
        )

        await self._cache.store(
            npub,
            record,
            vault_data={
                "npub": npub,
                "verified_at": str(now),
                "expires_at": str(now + self._ttl),
            },
        )

        logger.info("Cached proven npub %s (expires in %ds)", npub[:20], self._ttl)
        return record

    def invalidate(self, npub: str) -> None:
        """Remove a proven npub from the memory cache."""
        self._cache.invalidate(npub)
