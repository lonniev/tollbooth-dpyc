"""Authority certificate verification — shared infrastructure."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Protocol identifiers this Operator understands.
UNDERSTOOD_PROTOCOLS: frozenset[str] = frozenset({"dpyp-01-base-certificate"})


class CertificateError(Exception):
    """Raised when a certificate fails validation."""


class _JTIStore:
    """Thread-safe in-memory certificate ID store for anti-replay protection."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}  # jti -> expiry timestamp
        self._lock = threading.Lock()

    def check_and_record(self, jti: str, exp: float) -> bool:
        """Record a JTI. Returns True if new, False if already seen (replay)."""
        self._cleanup()
        with self._lock:
            if jti in self._seen:
                return False
            self._seen[jti] = exp
            return True

    def _cleanup(self) -> None:
        """Remove expired JTIs."""
        now = time.time()
        with self._lock:
            self._seen = {j: e for j, e in self._seen.items() if e > now}


# Module-level singleton — shared across all calls within one process.
_jti_store = _JTIStore()


def verify_certificate_auto(
    token: str,
    *,
    authority_npub: str = "",
    understood_protocols: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Verify a Nostr event certificate signed by the Authority.

    Args:
        token: Nostr event JSON string.
        authority_npub: Authority's bech32 npub for Schnorr verification.
        understood_protocols: Protocol identifiers this Operator accepts.

    Returns:
        Claims dict: operator_id, amount_sats, tax_paid_sats,
        net_sats, jti, dpyc_protocol.

    Raises:
        CertificateError: On verification failure or missing authority_npub.
    """
    if not authority_npub:
        raise CertificateError(
            "No authority_npub configured — cannot verify certificate."
        )
    from tollbooth.nostr_certificate import verify_nostr_certificate

    return verify_nostr_certificate(
        token.strip(), authority_npub, understood_protocols=understood_protocols,
    )


def reset_jti_store() -> None:
    """Reset the JTI store — for testing only."""
    global _jti_store
    _jti_store = _JTIStore()
