"""QR Credential Card — scan-and-paste credential reuse for Secure Courier.

Returning users can scan a QR code or paste an ``ncred1...`` string to
reactivate credentials without re-typing.  The card is a self-contained
Nostr event (kind 21420) with NIP-44 encrypted credentials, Schnorr-signed
by the operator, and bech32-encoded with HRP ``ncred``.

Encoding flow::

    credentials dict
      → JSON → NIP-44 encrypt(operator_nsec, user_npub)
      → kind-21420 event with service/expiry tags
      → sign(operator_nsec)
      → JSON serialize → bech32("ncred", ...)
      → ncred1<data>

Decoding flow::

    ncred1<data>
      → bech32 decode → JSON → Event
      → verify signature
      → check expiration
      → NIP-44 decrypt(operator_nsec, tagged user_npub)
      → credentials dict

Dependencies: pynostr (core dep), segno (optional, for QR rendering).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Custom Nostr event kind for credential cards
CREDENTIAL_CARD_KIND = 21420

# Default expiry for credential cards
DEFAULT_EXPIRY_DAYS = 90

# ── Exceptions ─────────────────────────────────────────────────────────


class CredentialCardError(Exception):
    """Base error for credential card operations."""


class CredentialCardExpired(CredentialCardError):
    """Raised when a credential card has expired."""


class CredentialCardInvalid(CredentialCardError):
    """Raised when a credential card fails validation (bad sig, corrupt data, etc.)."""


# ── Bech32 ncred encoding ─────────────────────────────────────────────

try:
    from pynostr.bech32 import (  # type: ignore[import-untyped]
        Encoding,
        bech32_decode,
        bech32_encode,
        convertbits,
    )
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]

    _HAS_PYNOSTR = True
except ImportError:
    _HAS_PYNOSTR = False

try:
    from tollbooth.nip44 import decrypt as _nip44_decrypt
    from tollbooth.nip44 import encrypt as _nip44_encrypt

    _HAS_NIP44 = True
except ImportError:
    _HAS_NIP44 = False

_HRP = "ncred"


def _ncred_encode(event_json_bytes: bytes) -> str:
    """Encode raw bytes as ``ncred1<bech32data>``."""
    if not _HAS_PYNOSTR:
        raise CredentialCardError("pynostr required for ncred encoding")
    bits5 = convertbits(event_json_bytes, 8, 5)
    return bech32_encode(_HRP, bits5, Encoding.BECH32)


def _ncred_decode(ncred: str) -> dict[str, Any]:
    """Decode ``ncred1<bech32data>`` back to the event dict."""
    if not _HAS_PYNOSTR:
        raise CredentialCardError("pynostr required for ncred decoding")

    hrp, bits5, _spec = bech32_decode(ncred.lower())
    if hrp != _HRP:
        raise CredentialCardInvalid(
            f"Invalid credential card prefix: expected '{_HRP}', got '{hrp}'"
        )
    if bits5 is None:
        raise CredentialCardInvalid("Corrupt credential card: bech32 decode failed")

    raw = bytes(convertbits(bits5, 5, 8, False))
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CredentialCardInvalid(f"Corrupt credential card payload: {exc}") from exc


# ── Public API ─────────────────────────────────────────────────────────


def encode_credential_card(
    credentials: dict[str, str],
    service: str,
    operator_nsec: str,
    user_npub: str,
    *,
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
) -> str:
    """Build a kind-21420 credential card event and encode as ``ncred1...``.

    Args:
        credentials: Field name → value mapping to encrypt.
        service: Service identifier (e.g. ``"thebrain"``).
        operator_nsec: Operator's Nostr nsec (bech32).
        user_npub: Patron's npub (bech32) — used as NIP-44 encryption target.
        expiry_days: Card validity in days (default 90).

    Returns:
        ``ncred1<bech32data>`` string.

    Raises:
        CredentialCardError: If dependencies are missing or encoding fails.
    """
    if not _HAS_PYNOSTR:
        raise CredentialCardError("pynostr required for credential cards")
    if not _HAS_NIP44:
        raise CredentialCardError("NIP-44 module required for credential cards")

    try:
        pk = PrivateKey.from_nsec(operator_nsec)
    except Exception as exc:
        raise CredentialCardError(f"Invalid operator nsec: {exc}") from exc

    try:
        user_hex = PublicKey.from_npub(user_npub).hex()
    except Exception as exc:
        raise CredentialCardError(f"Invalid user npub: {exc}") from exc

    operator_privkey_hex = pk.hex()
    operator_pubkey_hex = pk.public_key.hex()

    now = int(time.time())
    expires_at = now + (expiry_days * 86400)

    # NIP-44 encrypt the credentials JSON
    encrypted_content = _nip44_encrypt(
        json.dumps(credentials),
        operator_privkey_hex,
        user_hex,
    )

    event = Event(
        kind=CREDENTIAL_CARD_KIND,
        content=encrypted_content,
        tags=[
            ["service", service],
            ["v", "1"],
            ["p", user_hex],
            ["expiration", str(expires_at)],
        ],
        pubkey=operator_pubkey_hex,
        created_at=now,
    )
    event.sign(operator_privkey_hex)

    event_json = json.dumps(event.to_dict(), separators=(",", ":"))
    return _ncred_encode(event_json.encode("utf-8"))


def decode_credential_card(
    ncred: str,
    operator_nsec: str,
) -> dict[str, Any]:
    """Decode and verify an ``ncred1...`` credential card.

    Args:
        ncred: The ``ncred1...`` bech32 string.
        operator_nsec: Operator's Nostr nsec for decryption and sig verification.

    Returns:
        Dict with keys: ``service``, ``credentials``, ``user_npub``,
        ``issued_at``, ``expires_at``.

    Raises:
        CredentialCardInvalid: Bad signature, wrong operator, corrupt data.
        CredentialCardExpired: Card past its expiration date.
    """
    if not _HAS_PYNOSTR:
        raise CredentialCardError("pynostr required for credential cards")
    if not _HAS_NIP44:
        raise CredentialCardError("NIP-44 module required for credential cards")

    # Decode bech32 → event dict
    event_dict = _ncred_decode(ncred)

    # Reconstruct Event for signature verification
    try:
        event = Event(
            id=event_dict["id"],
            kind=event_dict["kind"],
            content=event_dict["content"],
            tags=event_dict["tags"],
            pubkey=event_dict["pubkey"],
            created_at=event_dict["created_at"],
            sig=event_dict["sig"],
        )
    except (KeyError, TypeError) as exc:
        raise CredentialCardInvalid(
            f"Malformed credential card event: {exc}"
        ) from exc

    # Verify kind
    if event.kind != CREDENTIAL_CARD_KIND:
        raise CredentialCardInvalid(
            f"Wrong event kind: expected {CREDENTIAL_CARD_KIND}, got {event.kind}"
        )

    # Verify signature
    try:
        pk = PrivateKey.from_nsec(operator_nsec)
    except Exception as exc:
        raise CredentialCardError(f"Invalid operator nsec: {exc}") from exc

    if event.pubkey != pk.public_key.hex():
        raise CredentialCardInvalid(
            "Credential card was not signed by this operator"
        )

    if not event.verify():
        raise CredentialCardInvalid(
            "Credential card signature verification failed"
        )

    # Extract tags
    tags = {tag[0]: tag[1] for tag in event.tags if len(tag) >= 2}

    service = tags.get("service")
    if not service:
        raise CredentialCardInvalid("Credential card missing 'service' tag")

    user_hex = tags.get("p")
    if not user_hex:
        raise CredentialCardInvalid("Credential card missing 'p' tag (user pubkey)")

    expiration_str = tags.get("expiration")
    if not expiration_str:
        raise CredentialCardInvalid("Credential card missing 'expiration' tag")

    try:
        expires_at = int(expiration_str)
    except ValueError as exc:
        raise CredentialCardInvalid(
            f"Invalid expiration timestamp: {expiration_str}"
        ) from exc

    # Check expiration
    if time.time() > expires_at:
        raise CredentialCardExpired(
            f"Credential card expired at {expires_at} "
            f"({int(time.time()) - expires_at} seconds ago)"
        )

    # Decrypt credentials
    operator_privkey_hex = pk.hex()
    try:
        plaintext = _nip44_decrypt(event.content, operator_privkey_hex, user_hex)
        credentials = json.loads(plaintext)
    except Exception as exc:
        raise CredentialCardInvalid(
            f"Failed to decrypt credential card content: {exc}"
        ) from exc

    if not isinstance(credentials, dict):
        raise CredentialCardInvalid("Decrypted content is not a JSON object")

    user_npub = PublicKey(bytes.fromhex(user_hex)).bech32()

    return {
        "service": service,
        "credentials": credentials,
        "user_npub": user_npub,
        "issued_at": event.created_at,
        "expires_at": expires_at,
    }


def render_qr(ncred: str, *, scale: int = 6) -> bytes:
    """Render an ``ncred1...`` string as a QR code PNG.

    Uppercases the ncred string for optimal QR alphanumeric mode encoding.

    Args:
        ncred: The ``ncred1...`` bech32 string.
        scale: Module size in pixels (default 6).

    Returns:
        PNG image as bytes.

    Raises:
        CredentialCardError: If segno is not installed or rendering fails.
    """
    try:
        import segno  # type: ignore[import-untyped]
    except ImportError:
        raise CredentialCardError(
            "segno required for QR rendering. Install with: pip install segno"
        )

    import io

    # Uppercase for QR alphanumeric mode (better capacity)
    qr = segno.make(ncred.upper())
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=scale)
    return buf.getvalue()
