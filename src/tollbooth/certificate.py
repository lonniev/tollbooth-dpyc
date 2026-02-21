"""Authority certificate verification — Ed25519 JWT validation with anti-replay."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Protocol identifiers this Operator understands.
UNDERSTOOD_PROTOCOLS: frozenset[str] = frozenset({"dpyp-01-base-certificate"})


def normalize_public_key(raw: str) -> str:
    """Accept a bare base64 key string or full PEM and return valid PEM.

    Operators can set AUTHORITY_PUBLIC_KEY to just the base64 body
    (e.g. ``MCowBQYDK2VwAyEA...``) — no PEM headers needed. The
    variable name tells you it's a public key.
    """
    stripped = raw.strip()
    if stripped.startswith("-----"):
        return stripped
    return f"-----BEGIN PUBLIC KEY-----\n{stripped}\n-----END PUBLIC KEY-----"


def key_fingerprint(raw: str) -> str:
    """Return last 8 chars of the base64 key body for display."""
    stripped = raw.strip()
    if stripped.startswith("-----"):
        lines = [ln for ln in stripped.splitlines() if not ln.startswith("-----")]
        b64 = "".join(lines).strip()
    else:
        b64 = stripped
    return b64[-8:] if len(b64) >= 8 else b64


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


def verify_certificate(
    token: str,
    public_key_pem: str,
    *,
    understood_protocols: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Verify an Authority-signed Ed25519 JWT certificate.

    Args:
        token: The JWT string from certify_purchase.
        public_key_pem: The Authority's Ed25519 public key — either bare base64
            or full PEM format. The variable name indicates it's a public key.
        understood_protocols: Protocol identifiers this Operator accepts.
            Defaults to ``UNDERSTOOD_PROTOCOLS``.

    Returns:
        Dict with extracted claims: operator_id, amount_sats, tax_paid_sats,
        net_sats, jti, dpyc_protocol.

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

    # Normalize bare base64 to PEM
    pem = normalize_public_key(public_key_pem)

    # Load the public key
    try:
        public_key = load_pem_public_key(pem.encode())
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

    # Protocol version check
    protos = understood_protocols or UNDERSTOOD_PROTOCOLS
    proto = claims.get("dpyc_protocol")
    if not proto:
        raise CertificateError(
            "Certificate missing dpyc_protocol claim — "
            "Authority may be running incompatible version"
        )
    if proto not in protos:
        raise CertificateError(
            f"Unsupported protocol '{proto}'. "
            f"This Operator supports: {', '.join(sorted(protos))}"
        )

    return {
        "operator_id": claims.get("sub", ""),
        "amount_sats": claims.get("amount_sats", 0),
        "tax_paid_sats": claims.get("tax_paid_sats", 0),
        "net_sats": claims.get("net_sats", 0),
        "jti": jti,
        "dpyc_protocol": proto,
    }


def reset_jti_store() -> None:
    """Reset the JTI store — for testing only."""
    global _jti_store
    _jti_store = _JTIStore()
