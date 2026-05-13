"""Proven npub cache — poison-keyed npub ownership proof.

The proof is bound to the poison phrase that the calling application
received during the ``request_npub_proof`` / ``receive_npub_proof``
exchange. The application remembers the raw poison; the MCP stores
only ``sha256(poison):npub`` — never the raw poison itself.

Cache key: ``"{poison_hash}:{npub}"``.

On ``mark_proven``, the record is written to both the in-memory
cache and the Neon vault (encrypted by the operator's nsec).
On ``is_proven`` cache miss, the vault is checked before rejecting —
surviving serverless cold starts that wipe in-memory state.

Security: a vault compromise yields only hashed poisons — useless
without the raw values held exclusively by the calling application.
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
MAX_PROVEN_TTL = 604800  # 7 days — hard cap, patron cannot exceed

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
        raise ValueError("Empty duration string")
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


def _cache_key(poison_hash: str, npub: str) -> str:
    return f"{poison_hash}:{npub}"


def _vault_key(poison_hash: str, npub: str) -> str:
    return f"{_VAULT_KEY_PREFIX}{poison_hash}:{npub}"


@dataclass(frozen=True)
class ProvenNpub:
    """Record that an npub owner proved ownership via a poison phrase."""

    poison_hash: str
    npub: str
    verified_at: float
    expires_at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> ProvenNpub:
        return cls(**json.loads(raw))


class ProvenNpubCache:
    """Poison-keyed npub ownership cache with vault persistence.

    In-memory ``SessionCache`` for hot lookups.  On cache miss,
    falls back to the Neon vault (encrypted at rest) so proofs
    survive serverless cold starts.

    Keyed by ``(poison_hash, npub)`` — the calling application holds
    the raw poison phrase and supplies it on each paid tool call.
    The MCP never stores the raw poison.

    Args:
        ttl_seconds: How long a proven npub stays valid (default 2h).
        vault: Optional NeonVault for durable storage.
    """

    def __init__(
        self, ttl_seconds: int = DEFAULT_PROVEN_TTL, vault: Any | None = None, **_: Any
    ) -> None:
        self._ttl = ttl_seconds
        # SessionCache's global TTL must accommodate the longest possible
        # per-entry TTL.  Real expiry is checked in is_proven() using the
        # ProvenNpub.expires_at field, so the cache itself just needs to
        # avoid evicting entries before their patron-chosen duration.
        self._cache: SessionCache[ProvenNpub] = SessionCache(ttl_seconds=MAX_PROVEN_TTL)
        self._vault = vault

    async def is_proven(self, poison_hash: str, npub: str) -> bool:
        """Check if an npub is proven via the given poison hash."""
        key = _cache_key(poison_hash, npub)
        record = self._cache.get(key)

        if record is not None:
            if time.time() > record.expires_at:
                logger.warning(
                    "Proof cache EXPIRED for poison_hash=%s npub=%s — "
                    "verified_at=%.0f expires_at=%.0f now=%.0f (%.0fs overdue)",
                    poison_hash[:16], npub[:20],
                    record.verified_at, record.expires_at,
                    time.time(), time.time() - record.expires_at,
                )
                self._cache.clear(key)
                return False
            remaining = record.expires_at - time.time()
            logger.debug(
                "Proof cache HIT for poison_hash=%s npub=%s (%.0fs remaining)",
                poison_hash[:16], npub[:20], remaining,
            )
            return True

        # In-memory miss — try vault restore
        if self._vault:
            record = await self._vault_fetch(poison_hash, npub)
            if record is not None and time.time() < record.expires_at:
                self._cache.set(key, record)
                remaining = record.expires_at - time.time()
                logger.info(
                    "Proof cache RESTORED from vault for poison_hash=%s npub=%s (%.0fs remaining)",
                    poison_hash[:16], npub[:20], remaining,
                )
                return True
            if record is not None:
                logger.info(
                    "Vault proof expired for poison_hash=%s npub=%s — cleaning up",
                    poison_hash[:16], npub[:20],
                )
                await self._vault_delete(poison_hash, npub)

        cached_keys = list(self._cache._entries.keys())
        logger.warning(
            "Proof cache MISS for poison_hash=%s npub=%s — "
            "key=%s not found. %d entries in cache: %s",
            poison_hash[:16], npub[:20], key[:40],
            len(cached_keys),
            [k[:40] for k in cached_keys],
        )
        return False

    async def mark_proven(
        self, poison_hash: str, npub: str, ttl_override: Any = UNSET,
    ) -> ProvenNpub:
        """Cache an npub as ownership-proven via a poison phrase.

        Writes to both in-memory cache and vault (if configured).
        The TTL is capped at ``MAX_PROVEN_TTL`` (7 days) regardless
        of what the patron requests.

        Args:
            poison_hash: SHA-256 hex digest of the raw poison phrase.
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
            poison_hash=poison_hash,
            npub=npub,
            verified_at=now,
            expires_at=expires_at,
        )
        self._cache.set(_cache_key(poison_hash, npub), record)
        label = f"{ttl}s" if ttl is not None else "unlimited"
        logger.info(
            "Cached proven npub %s with poison_hash %s (expires in %s)",
            npub[:20], poison_hash[:12], label,
        )

        if self._vault:
            await self._vault_store(record)

        return record

    def invalidate(self, poison_hash: str, npub: str) -> None:
        """Remove a proven npub from the cache."""
        self._cache.clear(_cache_key(poison_hash, npub))

    async def proof_status(self, poison_hash: str, npub: str) -> dict[str, Any]:
        """Read-only lookup of a proof's current state.

        Mirrors :meth:`is_proven` but does NOT mutate cache state.
        Used by the ``check_proof_status`` standard tool so calling
        agents can ask "is this proof_token still going to work?"
        without burning credits on a guaranteed failure.

        Returns a dict with ``status`` (``"valid"`` | ``"expired"`` |
        ``"unknown"``) and ``expires_in_seconds`` (the remaining TTL,
        runtime-derived from the stored ``ProvenNpub.expires_at``).
        """
        key = _cache_key(poison_hash, npub)
        record = self._cache.get(key)

        if record is None and self._vault is not None:
            # Try vault — but do NOT delete on expiry (read-only path)
            record = await self._vault_fetch(poison_hash, npub)

        if record is None:
            return {"status": "unknown", "expires_in_seconds": 0}

        remaining = int(record.expires_at - time.time())
        if remaining <= 0:
            return {"status": "expired", "expires_in_seconds": 0}
        return {"status": "valid", "expires_in_seconds": remaining}

    # -- Vault helpers --------------------------------------------------------

    async def _vault_store(self, record: ProvenNpub) -> None:
        try:
            encrypted = self._vault._encrypt(record.to_json())
            await self._vault.set_config(
                _vault_key(record.poison_hash, record.npub), encrypted,
            )
            logger.debug(
                "Proof persisted to vault for poison_hash=%s npub=%s",
                record.poison_hash[:12], record.npub[:20],
            )
        except Exception as exc:
            logger.warning("Vault store for proven npub failed (non-fatal): %s", exc)

    async def _vault_fetch(self, poison_hash: str, npub: str) -> ProvenNpub | None:
        try:
            raw = await self._vault.get_config(_vault_key(poison_hash, npub))
            if raw is None:
                return None
            decrypted = self._vault._decrypt(raw)
            return ProvenNpub.from_json(decrypted)
        except Exception as exc:
            logger.warning("Vault fetch for proven npub failed (non-fatal): %s", exc)
            return None

    async def _vault_delete(self, poison_hash: str, npub: str) -> None:
        try:
            await self._vault.set_config(_vault_key(poison_hash, npub), "")
        except Exception:
            pass  # best-effort cleanup
