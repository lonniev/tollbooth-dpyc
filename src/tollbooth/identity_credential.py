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

# Proof-grant subtype of kind 30080 — Operator-signed bearer for nsec-less agents.
# Distinct tags so a citizen identity credential cannot be mistaken for a proof
# grant (and vice versa). The challenge nonce only *selects* this grant; it never
# unlocks one on its own.
PROOF_GRANT_TAG = "dpyc-proof-grant"
PROOF_GRANT_LABEL = "dpyc.proof_grant"
# A peer grant: the patron lets another npub read their account and pay for
# them at one Operator — never spend as them.
PEER_GRANT_TAG = "dpyc-peer-grant"
PEER_GRANT_LABEL = "dpyc.peer_grant"
PEER_GRANT_SCOPE = "read+fund"
# Who started a peer grant; the other party approved it by signed reply.
PEER_GRANT_ORIGINS = ("peer", "patron")
_PROOF_APPROVAL = ("patron_sig", "patron_event_id")
_PEER_APPROVAL = ("origin", "approver", "approver_sig", "approver_event_id")

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


def sign_proof_grant(
    patron_npub: str,
    operator_nsec: str,
    *,
    challenge_hash: str,
    patron_sig: str,
    patron_event_id: str,
    ttl_seconds: int = DEFAULT_CREDENTIAL_TTL_SECONDS,
) -> str:
    """Sign a kind-30080 Operator proof grant for an nsec-less agent.

    The grant attests that the Operator verified a patron-signed reply over a
    specific challenge nonce. The raw nonce never appears in the grant — only
    ``sha256(nonce)`` — so the nonce may later *select* the grant without being
    able to authorize alone.

    Tags:
        d: jti
        p: patron hex pubkey
        t: dpyc-proof-grant
        L: dpyc.proof_grant
        challenge: sha256(nonce) hex
        patron_sig: the patron's Schnorr signature on the verified reply
            (the NIP-17 seal, or the NIP-04 event)
        patron_event_id: id of that signed reply event
        expiration: unix ts (patron-chosen duration, capped by caller)

    Args:
        patron_npub: Patron's bech32 npub the grant is issued for.
        operator_nsec: Operator's bech32 nsec (or hex) for signing.
        challenge_hash: ``sha256(nonce).hexdigest()`` — never the raw nonce.
        patron_sig: The patron's Schnorr signature the Operator verified on
            the reply (never a placeholder — no verified signature, no grant).
        patron_event_id: Id of that verified reply event.
        ttl_seconds: Validity window in seconds.

    Returns:
        JSON string of the signed kind-30080 event.

    Raises:
        IdentityCredentialError: On signing failure.
    """
    return _sign_grant(
        patron_npub, operator_nsec,
        tag=PROOF_GRANT_TAG, label=PROOF_GRANT_LABEL,
        challenge_hash=challenge_hash, ttl_seconds=ttl_seconds,
        approval_tags=[["patron_sig", patron_sig], ["patron_event_id", patron_event_id]],
    )


def sign_peer_grant(
    patron_npub: str,
    peer_npub: str,
    operator_nsec: str,
    *,
    origin: str,
    challenge_hash: str,
    approver_sig: str,
    approver_event_id: str,
    ttl_seconds: int = DEFAULT_CREDENTIAL_TTL_SECONDS,
) -> str:
    """Sign a kind-30080 Operator peer grant: the patron lets *peer_npub* read
    their account and pay for them at this Operator.

    Either party may start it (*origin*, ``"peer"`` or ``"patron"``); the
    other approves it by a signed reply over the challenge, which the Operator
    verified. The starter's consent is the npub proof the Operator checked on
    its call. Tags: ``peer`` (hex), ``scope`` (always
    :data:`PEER_GRANT_SCOPE`), ``origin``, ``approver`` (the other party's
    hex), ``approver_sig`` and ``approver_event_id``. Its ``t`` tag is
    :data:`PEER_GRANT_TAG`, so it can never pass as a proof grant.
    """
    if origin not in PEER_GRANT_ORIGINS:
        raise IdentityCredentialError(f"origin must be one of {PEER_GRANT_ORIGINS}.")
    try:
        peer_hex = _npub_to_hex(peer_npub)
        patron_hex = _npub_to_hex(patron_npub)
    except Exception as e:
        raise IdentityCredentialError(f"Invalid npub: {e}") from e
    if peer_hex == patron_hex:
        raise IdentityCredentialError("A patron cannot be their own peer.")
    approver_hex = patron_hex if origin == "peer" else peer_hex
    return _sign_grant(
        patron_npub, operator_nsec,
        tag=PEER_GRANT_TAG, label=PEER_GRANT_LABEL,
        challenge_hash=challenge_hash, ttl_seconds=ttl_seconds,
        approval_tags=[
            ["origin", origin],
            ["approver", approver_hex],
            ["approver_sig", approver_sig],
            ["approver_event_id", approver_event_id],
        ],
        extra_tags=[["peer", peer_hex], ["scope", PEER_GRANT_SCOPE]],
        extra_content={"peer_npub": peer_npub, "scope": PEER_GRANT_SCOPE, "origin": origin},
    )


def _sign_grant(
    patron_npub: str,
    operator_nsec: str,
    *,
    tag: str,
    label: str,
    challenge_hash: str,
    ttl_seconds: int,
    approval_tags: list[list[str]],
    extra_tags: list[list[str]] | None = None,
    extra_content: dict[str, str] | None = None,
) -> str:
    """Sign one kind-30080 grant of type *tag* for *patron_npub*.

    *approval_tags* name the verified approval (who signed it, and its
    signature and event id); every value is required.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
        from pynostr.key import PrivateKey  # type: ignore[import-untyped]
    except ImportError as e:
        raise IdentityCredentialError(
            f"Missing dependency for proof-grant signing: {e}. "
            "Install with: pip install tollbooth-dpyc[nostr]"
        ) from e

    if not challenge_hash or not isinstance(challenge_hash, str):
        raise IdentityCredentialError("challenge_hash is required.")
    for name, value in approval_tags:
        if not value or not isinstance(value, str):
            raise IdentityCredentialError(f"{name} is required.")

    try:
        if operator_nsec.startswith("nsec1"):
            private_key = PrivateKey.from_nsec(operator_nsec)
        else:
            private_key = PrivateKey(bytes.fromhex(operator_nsec))
        operator_hex = private_key.public_key.hex()
        operator_npub = private_key.public_key.bech32()
    except Exception as e:
        raise IdentityCredentialError(f"Invalid operator nsec: {e}") from e

    try:
        patron_hex = _npub_to_hex(patron_npub)
    except Exception as e:
        raise IdentityCredentialError(f"Invalid patron npub: {e}") from e

    now = int(time.time())
    # Allow non-positive ttl so callers/tests can mint already-expired grants;
    # verification rejects those via the expiration check.
    expiration = now + int(ttl_seconds)
    jti = str(uuid.uuid4())

    content = json.dumps(
        {
            "patron_npub": patron_npub,
            "operator_npub": operator_npub,
            "issued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "challenge": challenge_hash,
            **(extra_content or {}),
        },
        separators=(",", ":"),
    )

    tags = [
        ["d", jti],
        ["p", patron_hex],
        ["t", tag],
        ["L", label],
        ["challenge", challenge_hash],
        ["expiration", str(expiration)],
        *approval_tags,
        *(extra_tags or []),
    ]

    event = Event(
        kind=IDENTITY_CREDENTIAL_KIND,
        content=content,
        tags=tags,
        pubkey=operator_hex,
        created_at=now,
    )
    event.sign(private_key.hex())
    return json.dumps(event.to_dict())


def verify_proof_grant(
    event_json: str,
    *,
    expected_patron_npub: str,
    challenge_hash: str | None = None,
    expected_operator_hex: str | None = None,
) -> dict[str, Any]:
    """Verify a kind-30080 Operator proof grant.

    Checks Operator Schnorr signature, kind, proof-grant tag, expiration,
    patron ``p`` tag / content match, optional challenge binding, and optional
    issuer pubkey match. Does **not** consult the vault — the grant is
    self-verifying on the hot path.

    Args:
        event_json: JSON string of the signed grant event.
        expected_patron_npub: The npub the caller claims to act as.
        challenge_hash: When provided, must equal the grant's ``challenge`` tag
            (``sha256(nonce)``). Omit only for status-style reads that already
            hold the grant.
        expected_operator_hex: When provided, the grant signer must equal this
            Operator pubkey (hex). Used by the gate so a foreign Operator's
            grant cannot authorize here.

    Returns:
        Dict with claims: patron_npub, operator_npub, operator_hex, jti,
        expiration, challenge, patron_sig, patron_event_id.

    Raises:
        IdentityCredentialError: On any verification failure. The exception
            message is a stable machine-readable reason code when possible
            (``expired``, ``wrong_npub``, ``challenge_mismatch``,
            ``revoked`` is reserved for the gate's tombstone check,
            ``bad_signature``, ``wrong_kind``, ``not_proof_grant``,
            ``operator_mismatch``, ``malformed``).
    """
    claims, _tags = _verify_grant(
        event_json, tag=PROOF_GRANT_TAG, wrong_tag_reason="not_proof_grant",
        expected_patron_npub=expected_patron_npub, challenge_hash=challenge_hash,
        expected_operator_hex=expected_operator_hex, approval=_PROOF_APPROVAL,
    )
    return claims


def verify_peer_grant(
    event_json: str,
    *,
    expected_patron_npub: str,
    expected_peer_npub: str,
    expected_operator_hex: str | None = None,
) -> dict[str, Any]:
    """Verify a kind-30080 Operator peer grant from *expected_patron_npub* to
    *expected_peer_npub*.

    Every check of :func:`verify_proof_grant`, plus: the ``t`` tag is
    :data:`PEER_GRANT_TAG` (``not_peer_grant`` otherwise), the ``peer`` tag is
    the expected peer (``wrong_peer``), ``scope`` is
    :data:`PEER_GRANT_SCOPE` (``wrong_scope``), and the ``approver`` is the
    party that did not start it (``wrong_approver``).

    Returns the proof-grant claims plus ``peer_npub`` and ``scope``.
    """
    claims, tags = _verify_grant(
        event_json, tag=PEER_GRANT_TAG, wrong_tag_reason="not_peer_grant",
        expected_patron_npub=expected_patron_npub, challenge_hash=None,
        expected_operator_hex=expected_operator_hex, approval=_PEER_APPROVAL,
    )
    try:
        expected_peer_hex = _npub_to_hex(expected_peer_npub)
    except Exception as e:
        raise IdentityCredentialError("wrong_peer") from e
    if _get_tag_value(tags, "peer") != expected_peer_hex:
        raise IdentityCredentialError("wrong_peer")
    if _get_tag_value(tags, "scope") != PEER_GRANT_SCOPE:
        raise IdentityCredentialError("wrong_scope")
    # The approver is the party that did not start it.
    origin = claims["origin"]
    if origin not in PEER_GRANT_ORIGINS:
        raise IdentityCredentialError("malformed")
    approver = _npub_to_hex(expected_patron_npub) if origin == "peer" else expected_peer_hex
    if claims["approver"] != approver:
        raise IdentityCredentialError("wrong_approver")
    return {**claims, "peer_npub": expected_peer_npub, "scope": PEER_GRANT_SCOPE}


def _verify_grant(
    event_json: str,
    *,
    tag: str,
    wrong_tag_reason: str,
    expected_patron_npub: str,
    challenge_hash: str | None,
    expected_operator_hex: str | None,
    approval: tuple[str, ...],
) -> tuple[dict[str, Any], list[list[str]]]:
    """The checks every kind-30080 grant passes; returns (claims, tags)."""
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError as e:
        raise IdentityCredentialError(
            f"Missing dependency for proof-grant verification: {e}. "
            "Install with: pip install tollbooth-dpyc[nostr]"
        ) from e

    try:
        event_dict = json.loads(event_json)
    except (json.JSONDecodeError, TypeError) as e:
        raise IdentityCredentialError("malformed") from e

    try:
        event = Event.from_dict(event_dict)
    except Exception as e:
        raise IdentityCredentialError("malformed") from e

    try:
        if not event.verify():
            raise IdentityCredentialError("bad_signature")
    except IdentityCredentialError:
        raise
    except Exception as e:
        raise IdentityCredentialError("bad_signature") from e

    if event.kind != IDENTITY_CREDENTIAL_KIND:
        raise IdentityCredentialError("wrong_kind")

    if _get_tag_value(event.tags, "t") != tag:
        # A grant of another type (or a plain identity credential) is refused.
        raise IdentityCredentialError(wrong_tag_reason)

    expiration_str = _get_tag_value(event.tags, "expiration")
    if not expiration_str:
        raise IdentityCredentialError("malformed")
    try:
        expiration = int(expiration_str)
    except (ValueError, TypeError) as e:
        raise IdentityCredentialError("malformed") from e
    if expiration < time.time():
        raise IdentityCredentialError("expired")

    jti = _get_tag_value(event.tags, "d")
    if not jti:
        raise IdentityCredentialError("malformed")

    challenge = _get_tag_value(event.tags, "challenge") or ""
    if not challenge:
        raise IdentityCredentialError("malformed")
    if challenge_hash is not None and challenge != challenge_hash:
        raise IdentityCredentialError("challenge_mismatch")

    approved = {name: _get_tag_value(event.tags, name) or "" for name in approval}
    if not all(approved.values()):
        raise IdentityCredentialError("malformed")

    try:
        expected_patron_hex = _npub_to_hex(expected_patron_npub)
    except Exception as e:
        raise IdentityCredentialError("wrong_npub") from e

    p_tag = _get_tag_value(event.tags, "p")
    if p_tag != expected_patron_hex:
        raise IdentityCredentialError("wrong_npub")

    try:
        claims = json.loads(event.content) if event.content else {}
    except (json.JSONDecodeError, TypeError):
        claims = {}
    content_patron = claims.get("patron_npub", "")
    if content_patron and content_patron != expected_patron_npub:
        raise IdentityCredentialError("wrong_npub")

    if expected_operator_hex and event.pubkey != expected_operator_hex:
        raise IdentityCredentialError("operator_mismatch")

    operator_npub = claims.get("operator_npub", "")
    if not operator_npub:
        try:
            operator_npub = _hex_to_npub(event.pubkey)
        except Exception:  # noqa: BLE001
            operator_npub = ""

    return {
        "patron_npub": expected_patron_npub,
        "operator_npub": operator_npub,
        "operator_hex": event.pubkey,
        "issued_at": claims.get("issued_at", ""),
        "created_at": int(event.created_at),
        "jti": jti,
        "expiration": expiration,
        "challenge": challenge,
        **approved,
    }, event.tags


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
