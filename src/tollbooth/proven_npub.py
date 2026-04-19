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
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from tollbooth.session_cache import SessionCache

logger = logging.getLogger(__name__)

DEFAULT_PROVEN_TTL = 7200  # 2 hours
MAX_PROVEN_TTL = 86400  # 24 hours — hard cap, patron cannot exceed

# Sentinel: "patron did not specify a duration"
UNSET: Any = object()

# ---------------------------------------------------------------------------
# Human-friendly duration parser
# ---------------------------------------------------------------------------

_WORD_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "sixty": 60,
}

_UNIT_SECONDS: dict[str, int] = {
    "s": 1, "sec": 1, "second": 1, "seconds": 1,
    "m": 60, "min": 60, "minute": 60, "minutes": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "week": 604800, "weeks": 604800,
}

_DURATION_RE = re.compile(
    r"^\s*(\d+|[a-z]+)\s*(s|sec|seconds?|m|min|minutes?|h|hr|hrs|hours?|d|days?|w|weeks?)\s*$",
    re.IGNORECASE,
)


def parse_duration(text: str) -> int | None:
    """Parse a human-friendly duration string into seconds.

    Returns ``None`` for unlimited/never-expiring. Raises ``ValueError``
    for unrecognizable input.

    Examples::

        parse_duration("2h")         → 7200
        parse_duration("two days")   → 172800
        parse_duration("  30  min ") → 1800
        parse_duration("unlimited")  → None
        parse_duration("forever")    → None
    """
    cleaned = text.strip().lower()
    if not cleaned:
        raise ValueError(f"Empty duration string")
    if cleaned in ("unlimited", "never", "forever", "none", "no expiry", "no expiration"):
        return None
    m = _DURATION_RE.match(cleaned)
    if not m:
        raise ValueError(f"Cannot parse duration: {text!r}")
    amount_str, unit_str = m.group(1), m.group(2).lower()
    if amount_str.isdigit():
        amount = int(amount_str)
    else:
        amount = _WORD_NUMBERS.get(amount_str)
        if amount is None:
            raise ValueError(f"Cannot parse number: {amount_str!r} in {text!r}")
    unit_secs = _UNIT_SECONDS.get(unit_str)
    if unit_secs is None:
        raise ValueError(f"Unknown time unit: {unit_str!r} in {text!r}")
    result = amount * unit_secs
    if result > MAX_PROVEN_TTL:
        result = MAX_PROVEN_TTL
    return result

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

    async def mark_proven(
        self, session_id: str, npub: str, ttl_override: Any = UNSET,
    ) -> ProvenNpub:
        """Cache an npub as ownership-proven on this session.

        Writes to both in-memory cache and vault (if configured).
        The TTL is capped at ``MAX_PROVEN_TTL`` (24 hours) regardless
        of what the patron requests.

        Args:
            session_id: The MCP transport session ID.
            npub: The patron's npub (bech32).
            ttl_override: Seconds until expiry. ``None`` or values
                exceeding the cap are clamped to ``MAX_PROVEN_TTL``.
                Omit (or pass ``UNSET``) to use the cache default.

        Returns:
            The cached ``ProvenNpub`` record.
        """
        ttl = self._ttl if ttl_override is UNSET else ttl_override
        if ttl is None or ttl > MAX_PROVEN_TTL:
            ttl = MAX_PROVEN_TTL
        now = time.time()
        expires_at = now + ttl
        record = ProvenNpub(
            session_id=session_id,
            npub=npub,
            verified_at=now,
            expires_at=expires_at,
        )
        self._cache.set(_cache_key(session_id, npub), record)
        label = f"{ttl}s" if ttl is not None else "unlimited"
        logger.info(
            "Cached proven npub %s on session %s (expires in %s)",
            npub[:20], session_id[:12], label,
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
