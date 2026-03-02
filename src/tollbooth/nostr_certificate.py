"""Authority certificate verification — Nostr event (Schnorr/BIP-340) validation with anti-replay.

Verifies kind 30079 parameterized replaceable events signed by the Authority's
Nostr key.

Dependencies: ``pynostr`` (for event parsing and Schnorr verification).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import tollbooth.certificate as _cert_mod
from tollbooth.certificate import CertificateError, UNDERSTOOD_PROTOCOLS

logger = logging.getLogger(__name__)

NOSTR_CERT_KIND = 30079
"""NIP-33 parameterized replaceable event kind for tollbooth certificates."""

NOSTR_CERT_TAG = "tollbooth-cert"
NOSTR_CERT_LABEL = "dpyc.tollbooth"


def _get_tag_value(tags: list[list[str]], key: str) -> str | None:
    """Extract the first value for a given tag key from a Nostr event's tags."""
    for tag in tags:
        if len(tag) >= 2 and tag[0] == key:
            return tag[1]
    return None


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to hex pubkey string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey.from_npub(npub).hex()


def verify_nostr_certificate(
    event_json: str,
    authority_npub: str,
    *,
    understood_protocols: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Verify a Nostr event certificate signed by the Authority.

    Args:
        event_json: JSON string of the signed Nostr event.
        authority_npub: The Authority's npub (bech32) for signature verification.
        understood_protocols: Protocol identifiers this Operator accepts.
            Defaults to ``UNDERSTOOD_PROTOCOLS``.

    Returns:
        Dict with extracted claims: operator_id, amount_sats, fee_sats,
        net_sats, jti, dpyc_protocol.

    Raises:
        CertificateError: On invalid, expired, tampered, or replayed certificates.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError as e:
        raise CertificateError(
            f"Missing dependency for Nostr certificate verification: {e}. "
            "Install with: pip install 'tollbooth-dpyc[nostr]'"
        ) from e

    # Parse event JSON
    try:
        event_dict = json.loads(event_json)
    except (json.JSONDecodeError, TypeError) as e:
        raise CertificateError(f"Certificate event is not valid JSON: {e}") from e

    # Construct Event object
    try:
        event = Event.from_dict(event_dict)
    except Exception as e:
        raise CertificateError(f"Invalid Nostr event structure: {e}") from e

    # 1. Verify Schnorr signature
    try:
        if not event.verify():
            raise CertificateError(
                "Certificate signature is invalid — possible tampering."
            )
    except CertificateError:
        raise
    except Exception as e:
        raise CertificateError(f"Signature verification failed: {e}") from e

    # 2. Verify signer is the expected Authority
    try:
        authority_hex = _npub_to_hex(authority_npub)
    except Exception as e:
        raise CertificateError(f"Invalid authority npub: {e}") from e

    if event.pubkey != authority_hex:
        raise CertificateError(
            "Certificate not signed by registered Authority."
        )

    # 3. Verify event kind
    if event.kind != NOSTR_CERT_KIND:
        raise CertificateError(
            f"Not a tollbooth certificate event (kind {event.kind}, expected {NOSTR_CERT_KIND})."
        )

    # 4. Check expiration (NIP-40)
    expiration_str = _get_tag_value(event.tags, "expiration")
    if not expiration_str:
        raise CertificateError("Certificate missing expiration tag.")
    try:
        expiration = int(expiration_str)
    except (ValueError, TypeError) as e:
        raise CertificateError(f"Invalid expiration tag: {e}") from e
    if expiration < time.time():
        raise CertificateError("Certificate has expired.")

    # 5. Extract JTI from d-tag
    jti = _get_tag_value(event.tags, "d")
    if not jti:
        raise CertificateError("Certificate missing d-tag (JTI).")

    # 6. Anti-replay check (shared anti-replay store — access via module
    #    so reset_jti_store() is visible)
    if not _cert_mod._jti_store.check_and_record(jti, float(expiration)):
        raise CertificateError(
            f"Certificate replay detected — jti {jti} already used."
        )

    # 7. Extract claims from content
    try:
        claims = json.loads(event.content)
    except (json.JSONDecodeError, TypeError) as e:
        raise CertificateError(f"Certificate content is not valid JSON: {e}") from e

    # 8. Protocol version check
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
        "fee_sats": claims.get("fee_sats", claims.get("tax_paid_sats", 0)),
        "net_sats": claims.get("net_sats", 0),
        "jti": jti,
        "dpyc_protocol": proto,
    }
