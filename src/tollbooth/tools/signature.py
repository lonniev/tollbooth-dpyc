"""Signatures over a statement: an npub agrees to exactly this text, and anyone can check it later.

Learned from Attestr (Vinney Cavallo's mutual-agreement draft): the parties
sign the sha256 of a *normalized* statement, both sign the same hash, and the
hash is what every later record refers to. No new event kind is used. A
signature is made one of two ways, each built on a Tollbooth protocol already
in service:

- **A signed request** (NIP-98). An agent holding its own key calls a tool
  whose arguments carry the statement's hash. The kind-27235 event signs the
  sha256 of the request body, so the event plus that body is the signature
  (:func:`signed_request_record`).
- **A signed reply** (the npub-proof DM). The Operator DMs the statement and
  its hash on the Secure Courier; the npub replies from any Nostr client. The
  reply's signed layer is the signature, and the Operator countersigns a
  witness binding that reply to the statement's hash and the one-time
  challenge (:func:`request_signature_tool`, :func:`receive_signature_tool`).

:func:`verify_signature_record` checks either kind of record from its own
contents, with no relay and no clock.

A DM reply's content is encrypted to the Operator, so anyone can check who
signed and when; showing what the reply said takes the conversation key. A
signed request carries its body in the clear.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from tollbooth.tools.proof import (
    SIGNATURE_SERVICE,
    ApprovalReply,
    collect_signed_approval,
    send_approval_request,
)

logger = logging.getLogger(__name__)

#: The longest statement a DM asks an npub to sign.
MAX_STATEMENT_CHARS = 4000
#: ``u``-tag sentinel marking a kind-27235 event as an Operator's witness to a
#: signed reply. Like the provenance attestation, it carries no ``method`` or
#: ``payload`` tag, so it can never pass as a signed request.
SIGNATURE_WITNESS_TOOL = "signature_witness"


def normalized_statement(text: str) -> str:
    """The statement as it is hashed: trimmed, with Unix line endings (as Attestr does)."""
    return text.strip().replace("\r\n", "\n")


def statement_hash(text: str) -> str:
    """The hex sha256 of the normalized statement: what both parties sign."""
    return hashlib.sha256(normalized_statement(text).encode("utf-8")).hexdigest()


def _binder() -> str | None:
    """Who may collect the reply: this holder connection, or the npub that signed this request."""
    from tollbooth.holder_auth import current_holder, signed_npub

    holder = current_holder()
    if holder:
        return holder
    npub = signed_npub()
    return f"signed:{npub}" if npub else None


def _channel(binder: str, digest: str) -> str:
    """One channel per caller and statement, so two signatures asked at once never collide."""
    return hashlib.sha256(f"{binder}|{digest}".encode()).hexdigest()[:16]


def _refused(error: str, code: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"success": False, "error": error}
    if code:
        out["error_code"] = code
    return out


# ---------------------------------------------------------------------------
# A signed request
# ---------------------------------------------------------------------------


def signed_request_record(digest: str) -> dict[str, Any] | None:
    """The request in flight as a signature over ``digest``, or None.

    None when the request is not signed, or its body does not name ``digest``.
    """
    from tollbooth.holder_auth import signed_request

    signed = signed_request()
    if signed is None or not digest or digest not in signed["body"]:
        return None
    return {
        "method": "signed_request",
        "signer": signed["npub"],
        "statement_hash": digest,
        "signed_at": int(signed["event"].get("created_at", 0)),
        "event": signed["event"],
        "body": signed["body"],
    }


# ---------------------------------------------------------------------------
# A signed reply
# ---------------------------------------------------------------------------


async def request_signature_tool(
    rt: Any, signer_npub: str, statement: str, *, service_name: str = "",
) -> dict[str, Any]:
    """DM ``signer_npub`` the statement and its hash, asking them to sign it by replying.

    Returns ``nonce`` and ``statement_hash``; pass both to
    :func:`receive_signature_tool` once the signer has replied.
    """
    err = rt.npub_validation_error(signer_npub, param="signer_npub")
    if err is not None:
        return err
    if not isinstance(statement, str) or not normalized_statement(statement):
        return _refused("There is no statement to sign.")
    text = normalized_statement(statement)
    if len(text) > MAX_STATEMENT_CHARS:
        return _refused(f"A statement to sign is at most {MAX_STATEMENT_CHARS} characters.")
    binder = _binder()
    if binder is None:
        return _refused(
            "Asking for a signature needs a connection that can collect the reply: "
            "an OAuth (holder) connection, or a request signed with your own key."
        )
    courier = await rt.courier()
    if courier is None:
        return _refused("Secure Courier not configured.")
    digest = statement_hash(text)
    where = service_name or "This service"
    greeting = (
        f"{where} asks you to sign this:\n\n{text}\n\n"
        f"sha256: {digest}\n\n"
        "Reply to sign it. Your signed reply is your signature."
    )
    sent = await send_approval_request(
        courier, rt.resolve_npub(signer_npub),
        service=SIGNATURE_SERVICE, channel=_channel(binder, digest), holder=binder,
        greeting=greeting, reason=f"Signature over sha256 {digest}", verify_at=None,
        label="Signature request", receive_tool="receive_signature",
    )
    if sent.get("success"):
        sent["statement_hash"] = digest
    return sent


async def receive_signature_tool(
    rt: Any, signer_npub: str, nonce: str, digest: str,
) -> dict[str, Any]:
    """Collect the signer's reply and return it as a signature record over ``digest``.

    Call once, after the signer says they replied. The record holds their
    signed reply and the Operator's witness binding it to ``digest``.
    """
    err = rt.npub_validation_error(signer_npub, param="signer_npub")
    if err is not None:
        return err
    if not nonce:
        return _refused("nonce is required — pass the nonce request_signature returned.")
    if not isinstance(digest, str) or len(digest) != 64:
        return _refused("statement_hash is required — pass the one request_signature returned.")
    binder = _binder()
    if binder is None:
        return _refused("Collect the signature on the connection that asked for it.")
    courier = await rt.courier()
    if courier is None:
        return _refused("Secure Courier not configured.")
    resolved = rt.resolve_npub(signer_npub)
    state = f"{SIGNATURE_SERVICE}#{_channel(binder, digest)}"
    reply = await collect_signed_approval(
        rt, courier._exchange, resolved, state, nonce,
        holder=binder, request_tool="request_signature",
    )
    if not isinstance(reply, ApprovalReply):
        return reply
    if not reply.patron_event:
        return _refused("The reply carried no signed event to keep, so it cannot stand as a signature.")

    from tollbooth.identity_proof import sign_signature_witness

    witness = sign_signature_witness(
        rt._get_nsec(),
        signer_npub=resolved,
        statement_hash=digest,
        reply_event_id=reply.patron_event_id,
        challenge_hash=hashlib.sha256(nonce.encode()).hexdigest(),
    )
    return {
        "success": True,
        "signature": {
            "method": "dm_reply",
            "signer": resolved,
            "statement_hash": digest,
            "signed_at": int(reply.patron_event.get("created_at", 0)),
            "reply_event": reply.patron_event,
            "witness": json.loads(witness),
        },
    }


# ---------------------------------------------------------------------------
# Checking a record
# ---------------------------------------------------------------------------


def _tags(event: dict[str, Any], name: str) -> list[str]:
    return [t[1] for t in event.get("tags") or [] if isinstance(t, list) and len(t) > 1 and t[0] == name]


def _verified_id(event: Any) -> str | None:
    from tollbooth.nostr_credentials import CourierValidationError, _verify_signed_layer

    if not isinstance(event, dict):
        return None
    try:
        return _verify_signed_layer(event, "signature")
    except (CourierValidationError, KeyError):
        return None


def verify_signature_record(
    record: Any, *, statement_hash: str, signer_npub: str, operator_npub: str = "",
) -> str | None:
    """Why ``record`` is not ``signer_npub``'s signature over ``statement_hash``, or None when it is.

    ``operator_npub``, for a DM reply, must be the npub that witnessed it.
    """
    from tollbooth.nostr_credentials import _npub_to_hex

    if not isinstance(record, dict) or record.get("statement_hash") != statement_hash:
        return "the record is not over this statement"
    signer_hex = _npub_to_hex(signer_npub)
    method = record.get("method")
    if method == "signed_request":
        event, body = record.get("event"), record.get("body")
        if not isinstance(body, str) or not isinstance(event, dict) or _verified_id(event) is None:
            return "the signed request does not verify"
        if event["pubkey"] != signer_hex or event.get("kind") != 27235:
            return "the signed request is not this signer's"
        if _tags(event, "payload") != [hashlib.sha256(body.encode("utf-8")).hexdigest()]:
            return "the body is not the one that was signed"
        if statement_hash not in body:
            return "the signed body does not name this statement"
        return None
    if method == "dm_reply":
        reply, witness = record.get("reply_event"), record.get("witness")
        if not isinstance(reply, dict) or not isinstance(witness, dict):
            return "the record lacks its reply or its witness"
        reply_id = _verified_id(reply)
        if reply_id is None or reply["pubkey"] != signer_hex:
            return "the reply is not this signer's signed event"
        if _verified_id(witness) is None:
            return "the witness does not verify"
        if operator_npub and witness["pubkey"] != _npub_to_hex(operator_npub):
            return "the witness is not this operator's"
        if (_tags(witness, "u") != [SIGNATURE_WITNESS_TOOL] or _tags(witness, "e") != [reply_id]
                or _tags(witness, "p") != [signer_hex] or _tags(witness, "sha256") != [statement_hash]):
            return "the witness does not bind this reply to this statement"
        return None
    return "unknown signature method"
