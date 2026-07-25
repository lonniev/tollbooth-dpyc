"""DPYC Identity Credential — Nostr kind 30080 Schnorr-signed identity attestation.

Operators issue identity credentials to citizens during Secure Courier
onboarding.  A credential proves that a citizen's npub was verified by a
registered operator.  The hot path (tool calls) never contacts the Oracle;
the cold path (purchases/top-ups) uses the credential for chain-walk
verification and ban checks.

Structure (Nostr event):
    kind: 30080 (NIP-33 parameterized replaceable)
    pubkey: operator hex pubkey (issuer)
    content: JSON {"citizen_npub", "operator_npub", "issued_at"}
    tags: [d: jti, p: citizen_hex, t: dpyc-identity, L: dpyc.identity, expiration: unix_ts]
    sig: Schnorr/BIP-340

Dependencies: ``pynostr`` — install with ``pip install tollbooth-dpyc[nostr]``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

IDENTITY_CREDENTIAL_KIND = 30080
"""NIP-33 parameterized replaceable event kind for DPYC identity credentials."""

IDENTITY_CREDENTIAL_TAG = "dpyc-identity"
IDENTITY_CREDENTIAL_LABEL = "dpyc.identity"

# Default credential TTL: 30 days
DEFAULT_CREDENTIAL_TTL_SECONDS = 30 * 24 * 3600


class IdentityCredentialError(Exception):
    """Raised when an identity credential fails validation."""


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to hex pubkey string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey.from_npub(npub).hex()


def _hex_to_npub(hex_pubkey: str) -> str:
    """Convert a hex pubkey to bech32 npub string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey(bytes.fromhex(hex_pubkey)).bech32()


def sign_identity_credential(
    citizen_npub: str,
    operator_nsec: str,
    *,
    ttl_seconds: int = DEFAULT_CREDENTIAL_TTL_SECONDS,
) -> str:
    """Sign a kind 30080 identity credential for a citizen.

    Args:
        citizen_npub: Citizen's bech32 npub to attest.
        operator_nsec: Operator's bech32 nsec for signing.
        ttl_seconds: Credential validity in seconds.

    Returns:
        JSON string of the signed Nostr event.

    Raises:
        IdentityCredentialError: On signing failure.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
        from pynostr.key import PrivateKey  # type: ignore[import-untyped]
    except ImportError as e:
        raise IdentityCredentialError(
            f"Missing dependency for identity credential signing: {e}. "
            "Install with: pip install tollbooth-dpyc[nostr]"
        ) from e

    try:
        private_key = PrivateKey.from_nsec(operator_nsec)
        operator_hex = private_key.public_key.hex()
        operator_npub = private_key.public_key.bech32()
    except Exception as e:
        raise IdentityCredentialError(f"Invalid operator nsec: {e}") from e

    try:
        citizen_hex = _npub_to_hex(citizen_npub)
    except Exception as e:
        raise IdentityCredentialError(f"Invalid citizen npub: {e}") from e

    now = int(time.time())
    expiration = now + ttl_seconds
    jti = str(uuid.uuid4())

    content = json.dumps(
        {
            "citizen_npub": citizen_npub,
            "operator_npub": operator_npub,
            "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        },
        separators=(",", ":"),
    )

    event = Event(
        kind=IDENTITY_CREDENTIAL_KIND,
        content=content,
        tags=[
            ["d", jti],
            ["p", citizen_hex],
            ["t", IDENTITY_CREDENTIAL_TAG],
            ["L", IDENTITY_CREDENTIAL_LABEL],
            ["expiration", str(expiration)],
        ],
        pubkey=operator_hex,
        created_at=now,
    )
    event.sign(private_key.hex())

    return json.dumps(event.to_dict())


def _get_tag_value(tags: list[list[str]], key: str) -> str | None:
    """Extract the first value for a given tag key."""
    for tag in tags:
        if len(tag) >= 2 and tag[0] == key:
            return tag[1]
    return None


def verify_identity_credential(
    event_json: str,
    expected_citizen_npub: str | None = None,
) -> dict[str, Any]:
    """Verify a kind 30080 identity credential.

    Checks:
      1. Schnorr signature validity
      2. Event kind == 30080
      3. NIP-40 expiration not past
      4. d-tag (JTI) present
      5. Citizen npub matches expected (if provided)

    Args:
        event_json: JSON string of the signed Nostr event.
        expected_citizen_npub: If provided, verify the credential is for
            this specific citizen.

    Returns:
        Dict with claims: citizen_npub, operator_npub, operator_hex,
        issued_at, jti, expiration.

    Raises:
        IdentityCredentialError: On any verification failure.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError as e:
        raise IdentityCredentialError(
            f"Missing dependency for identity credential verification: {e}. "
            "Install with: pip install tollbooth-dpyc[nostr]"
        ) from e

    # Parse event JSON
    try:
        event_dict = json.loads(event_json)
    except (json.JSONDecodeError, TypeError) as e:
        raise IdentityCredentialError(
            f"Credential is not valid JSON: {e}"
        ) from e

    # Construct Event
    try:
        event = Event.from_dict(event_dict)
    except Exception as e:
        raise IdentityCredentialError(
            f"Invalid Nostr event structure: {e}"
        ) from e

    # 1. Verify Schnorr signature
    try:
        if not event.verify():
            raise IdentityCredentialError(
                "Credential signature invalid — possible tampering."
            )
    except IdentityCredentialError:
        raise
    except Exception as e:
        raise IdentityCredentialError(
            f"Signature verification failed: {e}"
        ) from e

    # 2. Verify event kind
    if event.kind != IDENTITY_CREDENTIAL_KIND:
        raise IdentityCredentialError(
            f"Not an identity credential (kind {event.kind}, "
            f"expected {IDENTITY_CREDENTIAL_KIND})."
        )

    # 3. Check expiration (NIP-40)
    expiration_str = _get_tag_value(event.tags, "expiration")
    if not expiration_str:
        raise IdentityCredentialError("Credential missing expiration tag.")
    try:
        expiration = int(expiration_str)
    except (ValueError, TypeError) as e:
        raise IdentityCredentialError(
            f"Invalid expiration tag: {e}"
        ) from e
    if expiration < time.time():
        raise IdentityCredentialError("Credential has expired.")

    # 4. Extract JTI from d-tag
    jti = _get_tag_value(event.tags, "d")
    if not jti:
        raise IdentityCredentialError("Credential missing d-tag (JTI).")

    # 5. Extract claims from content
    try:
        claims = json.loads(event.content)
    except (json.JSONDecodeError, TypeError) as e:
        raise IdentityCredentialError(
            f"Credential content is not valid JSON: {e}"
        ) from e

    citizen_npub = claims.get("citizen_npub", "")
    operator_npub = claims.get("operator_npub", "")

    # 6. Verify citizen matches expected
    if expected_citizen_npub and citizen_npub != expected_citizen_npub:
        raise IdentityCredentialError(
            "Credential citizen_npub does not match expected npub."
        )

    # 7. Verify signer matches claimed operator
    try:
        operator_hex = _npub_to_hex(operator_npub)
    except Exception:  # noqa: BLE001
        operator_hex = ""
    if operator_hex and event.pubkey != operator_hex:
        raise IdentityCredentialError(
            "Credential signer does not match claimed operator_npub."
        )

    return {
        "citizen_npub": citizen_npub,
        "operator_npub": operator_npub,
        "operator_hex": event.pubkey,
        "issued_at": claims.get("issued_at", ""),
        "jti": jti,
        "expiration": expiration,
    }


async def verify_credential_chain(
    event_json: str,
    citizen_npub: str,
    registry: Any,
) -> dict[str, Any]:
    """Verify an identity credential and that its issuer is a registered operator.

    Performs full verification:
      1. ``verify_identity_credential()`` — sig, kind, expiration, citizen match
      2. Registry chain walk — issuer (operator) is a registered DPYC member

    Args:
        event_json: JSON string of the signed credential event.
        citizen_npub: Expected citizen npub to verify against.
        registry: A ``DPYCRegistry`` instance for membership verification.

    Returns:
        Claims dict from ``verify_identity_credential()``.

    Raises:
        IdentityCredentialError: On credential or chain verification failure.
    """
    claims = verify_identity_credential(event_json, citizen_npub)

    operator_npub = claims["operator_npub"]
    try:
        await registry.check_membership(operator_npub)
    except Exception as e:
        raise IdentityCredentialError(
            f"Credential issuer {operator_npub} is not a registered "
            f"DPYC member: {e}"
        ) from e

    return claims
