"""Authority certificate verification — Ed25519 JWT validation with anti-replay."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class CertificateError(Exception):
    """Raised when a certificate fails validation."""


class _JTIStore:
    """Thread-safe in-memory JTI (JWT ID) store for anti-replay protection."""

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


def verify_certificate(token: str, public_key_pem: str) -> dict[str, Any]:
    """Verify an Authority-signed Ed25519 JWT certificate.

    Args:
        token: The JWT string from certify_purchase.
        public_key_pem: The Authority's Ed25519 public key in PEM format.

    Returns:
        Dict with extracted claims: operator_id, amount_sats, tax_paid_sats,
        net_sats, jti.

    Raises:
        CertificateError: On invalid, expired, tampered, or replayed certificates.
    """
    try:
        import jwt
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError as e:
        raise CertificateError(
            f"Missing dependency for certificate verification: {e}. "
            "Install with: pip install 'PyJWT[crypto]'"
        ) from e

    # Load the public key
    try:
        public_key = load_pem_public_key(public_key_pem.encode())
    except (ValueError, TypeError) as e:
        raise CertificateError(f"Invalid authority public key: {e}") from e

    # Decode and verify the JWT
    try:
        claims = jwt.decode(token, public_key, algorithms=["EdDSA"])
    except jwt.ExpiredSignatureError as e:
        raise CertificateError("Certificate has expired.") from e
    except jwt.InvalidSignatureError as e:
        raise CertificateError("Certificate signature is invalid — possible tampering.") from e
    except jwt.DecodeError as e:
        raise CertificateError(f"Certificate could not be decoded: {e}") from e
    except jwt.InvalidTokenError as e:
        raise CertificateError(f"Invalid certificate: {e}") from e

    # Extract required fields
    jti = claims.get("jti")
    if not jti:
        raise CertificateError("Certificate missing jti claim.")

    exp = claims.get("exp")
    if not exp:
        raise CertificateError("Certificate missing exp claim.")

    # Anti-replay check
    if not _jti_store.check_and_record(jti, float(exp)):
        raise CertificateError(f"Certificate replay detected — jti {jti} already used.")

    return {
        "operator_id": claims.get("operator_id", ""),
        "amount_sats": claims.get("amount_sats", 0),
        "tax_paid_sats": claims.get("tax_paid_sats", 0),
        "net_sats": claims.get("net_sats", 0),
        "jti": jti,
    }


def reset_jti_store() -> None:
    """Reset the JTI store — for testing only."""
    global _jti_store
    _jti_store = _JTIStore()
