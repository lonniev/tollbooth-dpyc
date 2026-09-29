"""Proof-grant revocation — a per-npub watermark, vault-backed.

A proof grant (``identity_credential.sign_proof_grant``) is self-verifying:
the gate checks the Operator's signature, the patron binding, the challenge
hash and the expiry with no server-side state. The one thing a self-verifying
bearer cannot carry is *revocation*, so this module holds the single fact the
gate still needs from storage: **the moment a patron last revoked every grant
issued to them**. A grant created at or before that moment is refused.

Why a watermark and not per-grant tombstones: grants are bearer tokens the
caller holds, so ``forget_credentials`` cannot enumerate them. One timestamp
per npub covers every grant ever issued to that patron, needs no enumeration,
and costs one vault row per npub.

Hot-path cost: the watermark is held in memory per npub and re-read from the
vault only after ``refresh_seconds``; a paid call does not touch the vault
once the npub is warm. The vault write on revoke is authoritative — if it
fails, the revoke fails loudly rather than pretending.

Fail-closed: when the vault cannot be read at all, the gate cannot know
whether the patron revoked, so ``revoked_before`` raises
``RevocationStoreUnavailable`` and the gate refuses with a situation the
caller can retry. Authorizing on a guess is the one thing this store must
never do.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROVEN_TTL = 7200  # 2 hours — default grant lifetime when the patron names none
MAX_PROVEN_TTL = 2592000  # 30 days — hard cap on any grant, patron cannot exceed

DEFAULT_REFRESH_SECONDS = 60.0
"""How long a per-npub watermark read stays good before the vault is re-read."""

_VAULT_KEY_PREFIX = "proof_grant_revoked_before:"


class RevocationStoreUnavailable(Exception):
    """The vault could not answer whether a patron revoked their grants."""


# ---------------------------------------------------------------------------
# Human-friendly duration parser (the patron names their consent window in
# the DM reply: ``cache_duration = @@@two weeks@@@``)
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
    "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600,
    "d": 86400, "day": 86400, "days": 86400,
    "w": 604800, "wk": 604800, "week": 604800, "weeks": 604800,
}

_DURATION_RE = re.compile(
    r"^\s*(\d+|" + "|".join(_WORD_NUMBERS) + r")\s*([a-zA-Z]+)\s*$"
)


def parse_duration(text: str) -> int | None:
    """Parse a human-friendly duration string into seconds.

    Returns ``None`` for unlimited/never-expiring (the caller clamps that to
    ``MAX_PROVEN_TTL``). Raises ``ValueError`` for unrecognizable input.

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
        word = _WORD_NUMBERS.get(amount_str)
        if word is None:
            raise ValueError(f"Cannot parse number: {amount_str!r} in {text!r}")
        amount = word
    unit_secs = _UNIT_SECONDS.get(unit_str)
    if unit_secs is None:
        raise ValueError(f"Unknown time unit: {unit_str!r} in {text!r}")
    return min(amount * unit_secs, MAX_PROVEN_TTL)


def grant_ttl_seconds(raw_duration: str) -> int:
    """The grant lifetime a patron asked for, clamped to the cap.

    Empty → ``DEFAULT_PROVEN_TTL``; unparseable → the default as well (the
    reply is human-typed, and a typo must not strand the patron); unlimited or
    above the cap → ``MAX_PROVEN_TTL``.
    """
    raw = (raw_duration or "").strip()
    if not raw:
        return DEFAULT_PROVEN_TTL
    try:
        parsed = parse_duration(raw)
    except ValueError:
        return DEFAULT_PROVEN_TTL
    if parsed is None or parsed > MAX_PROVEN_TTL:
        return MAX_PROVEN_TTL
    return max(int(parsed), 1)


def _vault_key(npub: str) -> str:
    return f"{_VAULT_KEY_PREFIX}{npub}"


class ProofGrantRevocations:
    """Per-npub revocation watermark for proof grants.

    Args:
        vault: Optional NeonVault. Without one the watermark is in-memory
            only — fine for tests and for runtimes with no persistence.
        refresh_seconds: How long a read watermark stays good before the
            vault is consulted again.
    """

    def __init__(
        self, vault: Any | None = None, *, refresh_seconds: float = DEFAULT_REFRESH_SECONDS,
    ) -> None:
        self._vault = vault
        self._refresh = refresh_seconds
        # npub → (watermark or None, read_at)
        self._marks: dict[str, tuple[int | None, float]] = {}

    async def revoke_all(self, npub: str) -> int:
        """Refuse every grant issued to ``npub`` up to now. Returns the watermark.

        The vault write is authoritative: a failure raises so the caller can
        report that the revoke did not take, instead of a silent in-memory-only
        revoke that a cold start would forget.
        """
        watermark = int(time.time())
        if self._vault is not None:
            payload = json.dumps({"npub": npub, "revoked_before": watermark})
            await self._vault.set_config(_vault_key(npub), self._vault._encrypt(payload))
        self._marks[npub] = (watermark, time.time())
        logger.info("Proof grants revoked for %s… (before %d)", npub[:20], watermark)
        return watermark

    async def revoked_before(self, npub: str) -> int | None:
        """The patron's revocation watermark, or ``None`` if they never revoked.

        Served from memory while fresh; otherwise read from the vault. Raises
        ``RevocationStoreUnavailable`` when the vault cannot answer — the
        caller must fail closed.
        """
        now = time.time()
        cached = self._marks.get(npub)
        if cached is not None and now - cached[1] < self._refresh:
            return cached[0]
        if self._vault is None:
            return cached[0] if cached is not None else None
        try:
            raw = await self._vault.get_config(_vault_key(npub))
            watermark: int | None = None
            if raw:
                watermark = int(json.loads(self._vault._decrypt(raw))["revoked_before"])
        except Exception as exc:
            raise RevocationStoreUnavailable(str(exc)) from exc
        self._marks[npub] = (watermark, now)
        return watermark

    async def is_revoked(self, npub: str, grant_created_at: int) -> bool:
        """True when a grant minted at ``grant_created_at`` falls under the watermark."""
        watermark = await self.revoked_before(npub)
        return watermark is not None and int(grant_created_at) <= watermark
