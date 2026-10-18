"""Identity proof — npub ownership, proven on the connection.

Two ways a request proves an npub:

1. **A signed request** (NIP-98): ``Authorization: Nostr <base64 kind-27235>``
   whose ``u``, ``method`` and ``payload`` tags bind the event to this one
   request (``sign_request`` / ``verify_request_signature``). Whoever holds the
   nsec needs no grant.
2. **A held grant**: an OAuth (holder) connection's wallet keeps, per npub, the
   Operator-signed kind-30080 grant ``receive_npub_proof`` issued after the
   patron's DM reply — the envelope ``{"grant": <event>, "nonce": "<phrase>"}``.
   The grant is self-verifying (Operator sig, patron binding, expiry,
   challenge = sha256(nonce)); ``require_held_grant`` checks it.

The checks are actor-agnostic — Operators, Authorities, and any future runtime
use the same functions. A runtime passes its own ``operator_hex`` (only its
grants authorize here) and its ``revocations`` store (a patron's
``forget_credentials`` watermark).

Dependencies: ``pynostr`` (available via ``tollbooth-dpyc[nostr]``).
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tollbooth.proven_npub import ProofGrantRevocations

logger = logging.getLogger(__name__)

PROOF_EVENT_KIND = 27235
"""NIP-98 HTTP Auth event kind, repurposed for MCP identity proofs."""

# Machine-readable grant-rejection reasons (surfaced on PROOF_INVALID).
PROOF_REASON_GRANT_EXPIRED = "expired"
PROOF_REASON_GRANT_WRONG_NPUB = "wrong_npub"
PROOF_REASON_GRANT_CHALLENGE = "challenge_mismatch"
PROOF_REASON_GRANT_REVOKED = "revoked"
PROOF_REASON_GRANT_BAD_SIGNATURE = "bad_signature"
PROOF_REASON_GRANT_WRONG_KIND = "wrong_kind"
PROOF_REASON_GRANT_NOT_GRANT = "not_proof_grant"
PROOF_REASON_GRANT_OPERATOR = "operator_mismatch"
PROOF_REASON_GRANT_MALFORMED = "malformed"
PROOF_REASON_GRANT_MISSING_NONCE = "missing_nonce"
PROOF_REASON_GRANT_OPERATOR_UNKNOWN = "operator_unknown"
"""This runtime could not establish its own identity, so it cannot tell its
grants from a stranger's. A server-side fault — never authorize past it."""
PROOF_REASON_REVOCATIONS_UNAVAILABLE = "revocations_unavailable"
"""The revocation store could not be read; whether the patron revoked is
unknown, so the gate refuses. Retryable."""

DEFAULT_WINDOW_SECONDS = 60
"""Maximum age (in seconds) of a valid proof event."""

MAX_PROOF_JSON_BYTES = 64 * 1024
"""Reject proof payloads larger than this before parsing.

A signed kind-27235 event is well under 2 KB; anything approaching this cap is
an adversarial oversized payload (tool arguments are untrusted AI input). Bound
the input before ``json.loads`` so a 10 MB string can't be fully parsed."""

def _signed_proof_event(nsec: str, tags: list[list[str]]) -> str:
    """A kind-27235 event over ``tags`` plus a per-proof nonce, signed by ``nsec``.

    The nonce keeps two proofs minted in the same wall-clock second from being
    byte-identical: their ids would collide and the replay guard would refuse
    the second. It is signed but otherwise inert.
    """
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]

    if nsec.startswith("nsec1"):
        pk = PrivateKey.from_nsec(nsec)
    else:
        pk = PrivateKey(bytes.fromhex(nsec))

    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()),
        tags=[*tags, ["nonce", secrets.token_hex(16)]],
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


def sign_request(nsec: str, *, method: str, url: str, body: bytes) -> str:
    """The ``Authorization`` value for one request signed by ``nsec`` (NIP-98).

    The event names the method, the URL and the hex sha256 of the body, so it
    authorizes this request and nothing else.
    """
    import base64
    import hashlib

    event = _signed_proof_event(nsec, [
        ["u", url],
        ["method", method.upper()],
        ["payload", hashlib.sha256(body).hexdigest()],
    ])
    return "Nostr " + base64.b64encode(event.encode()).decode()


PROVENANCE_ATTESTATION_TOOL = "npub_proof_request"
"""``u``-tag sentinel marking a kind-27235 event as a proof-request provenance
attestation, so it can never be mistaken for a signed request: it carries no
``method`` or ``payload`` tag."""


def create_provenance_attestation(
    operator_nsec: str,
    *,
    sender_pubkey_hex: str,
    subject_npub: str,
    service: str,
    challenge: str,
    reason: str | None = None,
    origin: str | None = None,
    verify_at: str | None = None,
) -> str:
    """Sign an Operator provenance attestation for a Secure-Courier request DM.

    A Secure-Courier DM must be *delivered* from a key other than the
    Operator's own npub in the self-addressed case — Nostr relays silently
    drop self-addressed DMs, so ``open_channel`` sends self-DMs from a
    throwaway ephemeral key (commit #93). That leaves the human staring at an
    unfamiliar sender npub with no verifiable tie to the Operator they believe
    they are dealing with.

    This attestation restores the tie *inside* the (NIP-44 encrypted) DM body:
    the Operator signs — with its **registered** identity key, the asset an
    impostor does not hold — a kind-27235 event that binds:

    - ``sender``: the pubkey the recipient actually sees on the DM (the
      ephemeral delivery key, or the Operator's own key on a patron DM). An
      attestation cannot be lifted onto a DM delivered by a different key.
    - ``subject``: the npub whose ownership/credentials the DM concerns.
    - ``service``: the credential-template / proof service the DM rides on.
    - ``challenge``: the one-time ``nonce`` for this exchange, so the
      attestation is bound to this DM and cannot be replayed against another.

    The recipient verifies the signature (``verify_provenance_attestation``)
    and resolves the recovered signer pubkey against the DPYC registry to
    render the trust state. Provenance is thus **Operator-attested**, never
    **requester-asserted** — the design principle this closes.

    Args:
        operator_nsec: The Operator's registered private key (bech32 or hex).
        sender_pubkey_hex: Hex pubkey the DM is delivered from (ephemeral key
            for self-DMs, else the Operator's own pubkey).
        subject_npub: The npub the request concerns (bech32).
        service: The service name the DM rides on.
        challenge: The one-time ``nonce`` for this exchange.
        reason: Optional human-readable purpose the Operator states for this
            request ("I'm working on your request XYZ and need the Operator to
            do ABC"). Signed into the attestation as a ``reason`` tag so the
            recipient can render it *verifiably bound to the signer* — a relay
            cannot forge or alter the stated purpose. Omitted when not given.
        origin: Optional, operator-OBSERVED provenance of the client that
            triggered the request (a compact "geo · coarse-ip · client" string
            harvested server-side from the transport — never client self-report).
            Signed into an ``origin`` tag so the recipient sees *where the
            request came from* to judge an unsolicited ask. Best-effort: omitted
            when the transport exposes nothing. Omitted when not given.
        verify_at: Optional, free-form statement of *where the initiating agent
            already showed the recipient this exact ``challenge`` (nonce)* —
            a URL, or a description like "your Claude.ai conversation". This is
            the OAuth 2.0 Device Grant ``verification_uri`` generalized: the
            recipient approves only if the code here matches the one displayed
            there, so an unsolicited request (whose code the recipient has never
            seen anywhere) is refused. Signed in so a relay cannot rewrite the
            stated venue. Omitted when not given.

    Returns:
        JSON string of the signed kind-27235 attestation event.
    """
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]

    if operator_nsec.startswith("nsec1"):
        pk = PrivateKey.from_nsec(operator_nsec)
    else:
        pk = PrivateKey(bytes.fromhex(operator_nsec))

    tags = [
        ["u", PROVENANCE_ATTESTATION_TOOL],
        ["sender", sender_pubkey_hex],
        ["subject", subject_npub],
        ["service", service],
        ["challenge", challenge],
        ["nonce", secrets.token_hex(16)],
    ]
    if reason:
        tags.append(["reason", reason])
    if origin:
        tags.append(["origin", origin])
    if verify_at:
        tags.append(["verify_at", verify_at])

    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()),
        tags=tags,
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


def verify_provenance_attestation(
    attestation_json: str,
    *,
    expected_sender_pubkey_hex: str,
    expected_subject_npub: str,
    expected_challenge: str,
) -> dict[str, Any]:
    """Verify an Operator provenance attestation against what the recipient saw.

    Confirms the attestation is a well-formed, correctly-signed kind-27235
    provenance event whose bound facts (sender, subject, challenge) match the
    DM the recipient actually received — closing the lift-and-replay window.

    This function performs **cryptographic** verification only. It deliberately
    does *not* consult the DPYC registry: the caller resolves the recovered
    ``operator_pubkey_hex`` against the registry to decide the trust state
    (registered + certified → green; registered but novel → amber; unresolvable
    → red). Keeping registry I/O out of this leaf makes it pure and testable,
    and lets the caller apply its own fail-closed policy.

    Unlike ``verify_proof`` there is **no freshness window and no replay
    consumption**: the attestation is an inbound, idempotently-readable fact a
    human may verify minutes or hours after the DM arrived (the same reason the
    proof-reply path dropped its wall-clock gate), and re-reading it is not a
    replay.

    Returns a dict with:
        ``valid``: bool — signature valid and all bound fields match.
        ``operator_pubkey_hex``: str | None — recovered signer, for registry
            resolution (present whenever the signature verified, even if a
            bound field mismatched).
        ``reason``: str — machine-readable failure cause when ``valid`` is
            False (``"pynostr_missing"``, ``"malformed"``, ``"bad_signature"``,
            ``"wrong_kind"``, ``"not_attestation"``, ``"sender_mismatch"``,
            ``"subject_mismatch"``, ``"challenge_mismatch"``), else ``"ok"``.
    """
    result: dict[str, Any] = {
        "valid": False,
        "operator_pubkey_hex": None,
        "reason": "ok",
    }

    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pynostr not installed — cannot verify provenance attestation")
        result["reason"] = "pynostr_missing"
        return result

    if (
        not isinstance(attestation_json, str)
        or len(attestation_json) > MAX_PROOF_JSON_BYTES
    ):
        result["reason"] = "malformed"
        return result

    try:
        event = Event.from_dict(json.loads(attestation_json))
    except Exception:  # noqa: BLE001
        result["reason"] = "malformed"
        return result

    try:
        if not event.verify():
            result["reason"] = "bad_signature"
            return result
    except Exception:  # noqa: BLE001
        result["reason"] = "bad_signature"
        return result

    # Signature is valid — the signer pubkey is now trustworthy for the caller
    # to resolve against the registry, regardless of any bound-field mismatch.
    result["operator_pubkey_hex"] = event.pubkey

    if event.kind != PROOF_EVENT_KIND:
        result["reason"] = "wrong_kind"
        return result

    def _tag(name: str) -> str | None:
        for tag in event.tags:
            if len(tag) >= 2 and tag[0] == name:
                return tag[1]
        return None

    if _tag("u") != PROVENANCE_ATTESTATION_TOOL:
        result["reason"] = "not_attestation"
        return result
    if _tag("sender") != expected_sender_pubkey_hex:
        result["reason"] = "sender_mismatch"
        return result
    if _tag("subject") != expected_subject_npub:
        result["reason"] = "subject_mismatch"
        return result
    if _tag("challenge") != expected_challenge:
        result["reason"] = "challenge_mismatch"
        return result

    result["valid"] = True
    return result


# Replay protection: track consumed proof event IDs within the window
_consumed_proofs: dict[str, float] = {}  # event_id → expiry time
_CONSUMED_CLEANUP_INTERVAL = 120  # seconds between cleanups
# Hard cap on the consumed-proof set. Entries are only inserted after a proof
# passes full signature + freshness verification, so this set cannot be filled
# with arbitrary junk — but a flood of validly-signed, distinct proofs arriving
# faster than the 120s cleanup could still grow it unbounded. The cap bounds
# memory; 10k live entries is far above any honest operator's concurrent proof
# volume within a single freshness window.
_CONSUMED_MAX_ENTRIES = 10000
_last_cleanup: float = 0.0


def _cleanup_consumed() -> None:
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CONSUMED_CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    expired = [k for k, v in _consumed_proofs.items() if now > v]
    for k in expired:
        del _consumed_proofs[k]


def _record_consumed(event_id: str, expiry: float) -> None:
    """Record a consumed proof id under a hard size cap.

    The interval-based ``_cleanup_consumed`` drains expired ids lazily; this
    cap is the backstop when distinct valid proofs arrive faster than that
    cleanup runs. When full, already-expired ids are purged first; if still
    full, the ids closest to expiry are evicted. Evicting soonest-to-expire is
    safe: an evicted id can only be replayed in the sliver before its proof's
    freshness window lapses, after which the age check rejects it regardless.
    """
    if len(_consumed_proofs) >= _CONSUMED_MAX_ENTRIES:
        now = time.time()
        for k in [k for k, v in _consumed_proofs.items() if now > v]:
            del _consumed_proofs[k]
        if len(_consumed_proofs) >= _CONSUMED_MAX_ENTRIES:
            overflow = len(_consumed_proofs) - _CONSUMED_MAX_ENTRIES + 1
            for k, _ in sorted(
                _consumed_proofs.items(), key=lambda kv: kv[1]
            )[:overflow]:
                del _consumed_proofs[k]
    _consumed_proofs[event_id] = expiry


# Machine-readable rejection reasons for an inline Schnorr (Tactic-2) proof.
# Coarse by design — one value per distinct rejection branch in
# ``_verify_proof_reason`` — so a caller can tell *why* a proof was refused
# without leaking key material, the full event, or registry-membership facts.
PROOF_REASON_MALFORMED_JSON = "malformed_json"
"""Payload was not raw JSON within bounds (e.g. base64-wrapped, NIP-98 framed)."""
PROOF_REASON_BAD_EVENT = "bad_event"
"""JSON parsed, but not a well-formed Nostr event."""
PROOF_REASON_WRONG_KIND = "wrong_kind"
"""Event kind was not 27235."""
PROOF_REASON_SIGNATURE_INVALID = "signature_invalid"
"""Schnorr signature did not verify over the event id."""
PROOF_REASON_EXPIRED = "expired"
"""``created_at`` was outside the freshness window (too old or too far future)."""
PROOF_REASON_REPLAYED = "replayed"
"""This event id was already consumed within the window."""
PROOF_REASON_PYNOSTR_MISSING = "pynostr_missing"
"""pynostr is not installed — proof cannot be verified (environment fault)."""
PROOF_REASON_REQUEST_MISMATCH = "request_mismatch"
"""A signed request's method, URL path or body hash did not match the request."""

# Human-readable one-liners for each rejection reason, safe to hand to the
# caller. Each says *what* was wrong without revealing key material, the event,
# or registry facts (``npub_mismatch`` = "signer ≠ the npub you claimed", never
# "that npub is/isn't registered").
_PROOF_REASON_MESSAGES = {
    PROOF_REASON_MALFORMED_JSON: (
        "Invalid request signature: the Authorization value must be "
        "'Nostr <base64 of a signed kind-27235 event>'."
    ),
    PROOF_REASON_BAD_EVENT: (
        "Invalid identity proof: the payload parsed as JSON but is not a "
        "well-formed Nostr event."
    ),
    PROOF_REASON_WRONG_KIND: (
        "Invalid identity proof: the event kind must be 27235."
    ),
    PROOF_REASON_SIGNATURE_INVALID: (
        "Invalid identity proof: the Schnorr signature did not verify."
    ),
    PROOF_REASON_EXPIRED: (
        "Invalid identity proof: created_at is outside the "
        f"{DEFAULT_WINDOW_SECONDS}-second freshness window (too old or too far "
        "in the future)."
    ),
    PROOF_REASON_REPLAYED: (
        "Invalid identity proof: this event id was already used. Mint a fresh "
        "event (add a random nonce tag to avoid same-second id collisions)."
    ),
    PROOF_REASON_PYNOSTR_MISSING: (
        "Cannot verify identity proof: proof verification is unavailable on "
        "this server."
    ),
    PROOF_REASON_REQUEST_MISMATCH: (
        "Invalid request signature: its method, URL path or body hash does not "
        "match this request."
    ),
}


def _check_proof_event(
    proof_json: str,
    *,
    binds: Any,
    mismatch_reason: str,
    window_seconds: int,
) -> tuple[str | None, str]:
    """The checks every kind-27235 request event passes, in order: shape,
    signature, kind, binding, freshness, single use.

    The signer IS the identity being proven. ``binds(event)`` says whether the
    event is bound to what is being authorized: the method, path and body of
    the request.

    Returns ``(reason, signer_hex)``: ``reason`` is ``None`` when the proof is
    good, else one of the ``PROOF_REASON_*`` constants. Do not loosen any check.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pynostr not installed — cannot verify identity proof")
        return PROOF_REASON_PYNOSTR_MISSING, ""

    if not isinstance(proof_json, str) or len(proof_json) > MAX_PROOF_JSON_BYTES:
        logger.debug(
            "identity_proof: rejecting oversized/invalid payload (%s bytes)",
            len(proof_json) if isinstance(proof_json, str) else "non-str",
        )
        return PROOF_REASON_MALFORMED_JSON, ""

    try:
        event_dict = json.loads(proof_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("identity_proof: invalid JSON")
        return PROOF_REASON_MALFORMED_JSON, ""

    try:
        event = Event.from_dict(event_dict)
    except Exception:  # noqa: BLE001
        logger.debug("identity_proof: invalid Nostr event structure")
        return PROOF_REASON_BAD_EVENT, ""

    try:
        if not event.verify():
            logger.debug("identity_proof: signature verification failed")
            return PROOF_REASON_SIGNATURE_INVALID, ""
    except Exception:  # noqa: BLE001
        logger.debug("identity_proof: signature verification error")
        return PROOF_REASON_SIGNATURE_INVALID, ""

    if event.kind != PROOF_EVENT_KIND:
        logger.debug("identity_proof: wrong kind %d (expected %d)", event.kind, PROOF_EVENT_KIND)
        return PROOF_REASON_WRONG_KIND, ""

    if not binds(event):
        logger.debug("identity_proof: event not bound to this request (%s)", mismatch_reason)
        return mismatch_reason, ""

    now = time.time()
    age = abs(now - event.created_at)
    if age > window_seconds:
        logger.debug("identity_proof: expired (age=%.1fs, window=%ds)", age, window_seconds)
        return PROOF_REASON_EXPIRED, ""

    # Replay protection: reject reused proof event IDs
    _cleanup_consumed()
    event_id = getattr(event, "id", None) or event_dict.get("id", "")
    if event_id and event_id in _consumed_proofs:
        logger.debug("identity_proof: replay rejected (event_id=%s)", event_id[:16])
        return PROOF_REASON_REPLAYED, ""
    if event_id:
        _record_consumed(event_id, now + window_seconds)

    return None, event.pubkey


def _tag_values(event: Any, name: str) -> list[str]:
    return [tag[1] for tag in event.tags if len(tag) >= 2 and tag[0] == name]


def verify_request_signature(
    authorization: str,
    *,
    method: str,
    path: str,
    body_sha256: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> tuple[str | None, str | None]:
    """Verify a NIP-98 signed request: who signed THIS request, if anyone.

    ``authorization`` is the value after ``Nostr `` — base64 of a kind-27235
    event whose ``method`` tag is the HTTP method, whose ``u`` tag is the
    request URL and whose ``payload`` tag is the hex sha256 of the body. The
    URL's path must match; its host may differ, because a site's same-origin
    proxy forwards a request signed for the site to the operator, and the body
    already names the operator's slug-prefixed tool.

    Returns ``(npub, None)`` for a good signature, else ``(None, reason)``.
    """
    import base64
    from urllib.parse import urlparse

    try:
        proof_json = base64.b64decode(authorization.strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None, PROOF_REASON_MALFORMED_JSON

    def binds(event: Any) -> bool:
        urls = _tag_values(event, "u")
        return (
            _tag_values(event, "method") == [method.upper()]
            and _tag_values(event, "payload") == [body_sha256]
            and len(urls) == 1
            and urlparse(urls[0]).path.rstrip("/") == path.rstrip("/")
        )

    reason, signer_hex = _check_proof_event(
        proof_json,
        binds=binds,
        mismatch_reason=PROOF_REASON_REQUEST_MISMATCH,
        window_seconds=window_seconds,
    )
    if reason is not None:
        return None, reason
    from tollbooth.nostr_credentials import _hex_to_npub

    return _hex_to_npub(signer_hex), None


def _parse_grant_envelope(envelope: str) -> tuple[str, str | None] | None:
    """Parse a proof-grant envelope or bare kind-30080 grant from ``envelope``.

    Returns ``(grant_json, nonce_or_none)`` when the payload is grant-shaped,
    else ``None`` so the caller can try Tactic-2 (kind 27235).

    Accepted shapes:
      - envelope ``{"grant": <event dict|json str>, "nonce": "<phrase>"}``
      - bare kind-30080 event JSON (nonce required separately — returns
        ``nonce=None`` so the gate can demand it)
    """
    if not isinstance(envelope, str) or not envelope or envelope[0] not in "{[":
        return None
    if len(envelope) > MAX_PROOF_JSON_BYTES:
        return None
    try:
        payload = json.loads(envelope)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    # Envelope: grant + nonce (the receive_npub_proof return shape).
    if "grant" in payload:
        grant = payload["grant"]
        if isinstance(grant, dict):
            grant_json = json.dumps(grant, separators=(",", ":"))
        elif isinstance(grant, str):
            grant_json = grant
        else:
            return None
        nonce = payload.get("nonce")
        if nonce is not None and not isinstance(nonce, str):
            nonce = str(nonce) if nonce else None
        return grant_json, nonce

    # Bare kind-30080 event.
    kind = payload.get("kind")
    if kind == 30080:
        return envelope, None
    return None


_GRANT_REASON_MESSAGES = {
    PROOF_REASON_GRANT_EXPIRED: (
        "Invalid proof grant: the grant has expired. Call request_npub_proof "
        "and receive_npub_proof for a fresh grant."
    ),
    PROOF_REASON_GRANT_WRONG_NPUB: (
        "Invalid proof grant: the grant is not bound to the npub you claimed."
    ),
    PROOF_REASON_GRANT_CHALLENGE: (
        "Invalid proof grant: the nonce does not select this grant "
        "(sha256(nonce) ≠ challenge)."
    ),
    PROOF_REASON_GRANT_REVOKED: (
        "Invalid proof grant: this grant was revoked. Call request_npub_proof "
        "and receive_npub_proof for a fresh grant."
    ),
    PROOF_REASON_GRANT_BAD_SIGNATURE: (
        "Invalid proof grant: the Operator Schnorr signature did not verify."
    ),
    PROOF_REASON_GRANT_WRONG_KIND: (
        "Invalid proof grant: the event kind must be 30080."
    ),
    PROOF_REASON_GRANT_NOT_GRANT: (
        "Invalid proof grant: the event is not a dpyc-proof-grant."
    ),
    PROOF_REASON_GRANT_OPERATOR: (
        "Invalid proof grant: the grant was issued by a different Operator."
    ),
    PROOF_REASON_GRANT_MALFORMED: (
        "Invalid proof grant: the payload is malformed."
    ),
    PROOF_REASON_GRANT_MISSING_NONCE: (
        "Invalid proof grant: a nonce is required to select the grant. "
        "Prove the npub again with request_npub_proof / receive_npub_proof."
    ),
    PROOF_REASON_GRANT_OPERATOR_UNKNOWN: (
        "This service could not establish its own identity, so it cannot "
        "verify who issued your proof grant. That is a fault on the "
        "service side, not yours — report it to the operator."
    ),
    PROOF_REASON_REVOCATIONS_UNAVAILABLE: (
        "Could not check whether this proof grant was revoked because the "
        "service's persistence layer did not answer. Retry shortly."
    ),
}


def _grant_denial(reason: str) -> dict[str, Any]:
    """Structured refusal of a proof grant, one line per reason, no key material."""
    return {
        "success": False,
        "error_code": _error_codes().PROOF_INVALID,
        "reason": reason,
        "error": _GRANT_REASON_MESSAGES.get(reason, f"Invalid proof grant: {reason}."),
    }


def _error_codes() -> Any:
    # Lazy: constants imports nothing from here, but keep the dependency one-way.
    from tollbooth.constants import ErrorCode

    return ErrorCode


async def require_held_grant(
    npub: str,
    envelope: str,
    *,
    revocations: ProofGrantRevocations | None = None,
    operator_hex: str | None = None,
) -> dict[str, Any] | None:
    """Verify the grant a holder connection holds for ``npub``. Returns
    ``None`` on success or a structured error dict to return verbatim.

    ``envelope`` is what ``receive_npub_proof`` put in the holder's wallet:
    ``{"grant": <kind-30080 event>, "nonce": "<phrase>"}``. The grant is
    signed by this runtime (``operator_hex``), names ``npub`` and commits to
    ``sha256(nonce)``. A grant is accepted only when ``operator_hex`` is known
    (a grant from a stranger, or from a runtime that cannot name itself,
    never authorizes) and, when ``revocations`` is supplied, only if it
    post-dates the patron's last ``forget_credentials``.

    An empty ``envelope`` means the connection does not hold ``npub``.
    """
    # Lazy import — constants is leaf-y, but identity_proof is too;
    # keep the dep one-way to be safe.
    from tollbooth.constants import ErrorCode

    if not npub.startswith("npub1") or len(npub) < 60:
        return {
            "success": False,
            "error_code": ErrorCode.NPUB_INVALID,
            "error": (
                "Invalid npub format. Must start with 'npub1' and be at "
                "least 60 characters."
            ),
        }
    if not envelope:
        return {
            "success": False,
            "error_code": ErrorCode.PROOF_REQUIRED,
            "error": "This connection does not hold that npub yet.",
            "next_steps": [
                ("Holding npub's nsec: sign each HTTP request with it (NIP-98, "
                "`Authorization: Nostr <base64 kind-27235>` with u, method and "
                "payload tags) and pass only npub."),
                ("Holding no key: connect with OAuth, call request_npub_proof, "
                "reply to the DM from your Nostr client, then call "
                "receive_npub_proof once. The connection then holds npub; pass "
                "only npub."),
            ],
        }

    grant_parsed = _parse_grant_envelope(envelope)
    if grant_parsed is None:
        return _grant_denial(PROOF_REASON_GRANT_MALFORMED)
    grant_json, nonce = grant_parsed
    if not nonce:
        return _grant_denial(PROOF_REASON_GRANT_MISSING_NONCE)
    # Fail closed: a runtime that cannot name itself cannot tell its own
    # grants from a stranger's, and must not guess.
    if not operator_hex:
        return _grant_denial(PROOF_REASON_GRANT_OPERATOR_UNKNOWN)

    import hashlib as _hashlib

    challenge_hash = _hashlib.sha256(nonce.encode()).hexdigest()
    from tollbooth.identity_credential import (
        IdentityCredentialError,
        verify_proof_grant,
    )

    try:
        claims = verify_proof_grant(
            grant_json,
            expected_patron_npub=npub,
            challenge_hash=challenge_hash,
            expected_operator_hex=operator_hex,
        )
    except IdentityCredentialError as exc:
        return _grant_denial(str(exc) or PROOF_REASON_GRANT_MALFORMED)

    # Revocation: the patron's forget_credentials watermark. Unknown is
    # not "not revoked" — if the store cannot answer, refuse and say why.
    if revocations is not None:
        from tollbooth.proven_npub import RevocationStoreUnavailable

        try:
            revoked = await revocations.is_revoked(npub, claims["created_at"])
        except RevocationStoreUnavailable as exc:
            logger.warning("proof-grant revocation check unavailable: %s", exc)
            return _grant_denial(PROOF_REASON_REVOCATIONS_UNAVAILABLE)
        if revoked:
            return _grant_denial(PROOF_REASON_GRANT_REVOKED)

    return None
