"""Peer grants: a patron lets another npub watch their account and pay for them.

A *peer* (a Parent, a Supervisor, a business console) proves only its own npub.
Once the patron approves it by Secure Courier DM, the peer may call
``check_balance``, ``account_statement``, ``purchase_credits`` and
``check_payment`` for the patron at this Operator — and nothing else: paid tools
and every other patron tool still need the patron's own proof
(``OperatorRuntime.require_caller_or_peer``). A patron may approve several peers.

The approval rides the proof machinery (``tools.proof``): the same signed reply,
pinned relay, revocation watermark and Operator-signed kind-30080 grant, with
``t=dpyc-peer-grant`` and the peer named in the grant.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

from tollbooth.tools.proof import (
    PEER_SERVICE,
    ApprovalReply,
    collect_signed_approval,
    send_approval_request,
)

logger = logging.getLogger(__name__)

# A peer grant lasts this long when the patron's reply names no duration.
DEFAULT_PEER_GRANT = "30d"


def _peer_channel(peer_npub: str) -> str:
    """One channel per (patron, peer): a peer's request never replaces another's."""
    return hashlib.sha256(peer_npub.encode()).hexdigest()[:16]


def _short(npub: str) -> str:
    return npub[:16] + "…" if len(npub) > 20 else npub


async def _validate_pair(
    rt: Any, patron_npub: str, peer_npub: str,
) -> tuple[str, str, dict[str, Any] | None]:
    """Resolve both npubs and require the caller to prove the peer."""
    for value, param in ((patron_npub, "patron_npub"), (peer_npub, "peer_npub")):
        err = rt.npub_validation_error(value, param=param)
        if err is not None:
            return "", "", err
    patron = rt.resolve_npub(patron_npub)
    peer = rt.resolve_npub(peer_npub)
    if patron == peer:
        return "", "", {
            "success": False,
            "error": "A patron cannot be their own peer: they already act as themselves.",
        }
    # The peer asks for itself: the caller must prove the peer npub.
    err = await rt.prove_caller(peer)
    if err is not None:
        return "", "", err
    return patron, peer, None


async def request_peer_grant_tool(
    rt: Any, patron_npub: str, peer_npub: str, *, service_name: str,
) -> dict[str, Any]:
    """DM the patron an approval: may *peer_npub* watch and pay for them here?"""
    patron, peer, err = await _validate_pair(rt, patron_npub, peer_npub)
    if err is not None:
        return err
    courier = await rt.courier()
    if courier is None:
        from tollbooth.constants import ErrorCode as _EC
        return {
            "success": False,
            "error_code": _EC.SECURE_COURIER_UNAVAILABLE,
            "error": "Secure Courier not configured.",
        }
    where = service_name or "this service"
    greeting = (
        f"{peer} asks to see your account at {where} and to pay for you.\n"
        "They cannot spend your balance.\n"
        "Approve only if you know them."
    )
    return await send_approval_request(
        courier, patron,
        service=PEER_SERVICE, channel=_peer_channel(peer), holder=peer,
        greeting=greeting, reason=f"Peer grant for {peer}", verify_at=None,
        label="Peer grant request", receive_tool="receive_peer_grant",
    )


async def receive_peer_grant_tool(
    rt: Any, patron_npub: str, peer_npub: str, nonce: str,
) -> dict[str, Any]:
    """Collect the patron's signed approval and record the peer grant."""
    from tollbooth.constants import ErrorCode as _EC
    if not nonce:
        return {
            "success": False,
            "error_code": _EC.NONCE_MISSING,
            "error": "nonce is required — pass the nonce request_peer_grant returned.",
        }
    patron, peer, err = await _validate_pair(rt, patron_npub, peer_npub)
    if err is not None:
        return err
    courier = await rt.courier()
    if courier is None:
        return {
            "success": False,
            "error_code": _EC.SECURE_COURIER_UNAVAILABLE,
            "error": "Secure Courier not configured.",
        }
    exchange = courier._exchange
    reply = await collect_signed_approval(
        rt, exchange, patron, f"{PEER_SERVICE}#{_peer_channel(peer)}", nonce,
        holder=peer, request_tool="request_peer_grant",
    )
    if not isinstance(reply, ApprovalReply):
        return reply

    from tollbooth.identity_credential import sign_peer_grant
    from tollbooth.proven_npub import grant_ttl_seconds
    ttl = grant_ttl_seconds(reply.payload.get("allow_for") or DEFAULT_PEER_GRANT)
    try:
        grant_json = sign_peer_grant(
            patron, peer, rt._get_nsec(),
            challenge_hash=hashlib.sha256(nonce.encode()).hexdigest(),
            patron_sig=reply.patron_sig,
            patron_event_id=reply.patron_event_id,
            ttl_seconds=ttl,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("peer-grant issuance failed: %s", exc)
        return {"success": False, "error": f"Failed to issue the peer grant: {exc}"}

    expires_at = int(time.time()) + ttl
    envelope = json.dumps({"grant": json.loads(grant_json)}, separators=(",", ":"))
    await (await rt.peer_grant_store()).put(patron, peer, envelope, expires_at)

    until = datetime.fromtimestamp(expires_at, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    try:
        exchange.send_dm(
            patron,
            f"{_short(peer)} may now see your account at {rt._service_name} "
            f"and pay for you, until {until}.\n"
            "They cannot spend your balance.\n"
            "Revoke it any time with revoke_peer_grant.",
        )
    except Exception:
        logger.debug("peer-grant confirmation DM failed", exc_info=True)

    # The grant stays in the store; it never enters the caller's transcript.
    return {
        "success": True,
        "patron_npub": patron,
        "peer_npub": peer,
        "scope": "read+fund",
        "expires_at": until,
        "popped_dms": reply.popped,
        "message": (
            f"{_short(peer)} may now call check_balance, account_statement, "
            f"purchase_credits and check_payment for {_short(patron)} here, until {until}."
        ),
    }


async def list_peer_grants_tool(rt: Any, npub: str) -> dict[str, Any]:
    """The patron's peers here, with when each grant ends. Only the patron sees it."""
    if err := await rt.require_caller_proof(npub):
        return err
    patron = rt.resolve_npub(npub)
    try:
        peers = await (await rt.peer_grant_store()).peers_of(patron)
    except Exception as exc:  # noqa: BLE001 — never surface the SQL
        logger.warning("peer grant store unavailable: %s", type(exc).__name__)
        return _store_unavailable()
    return {"success": True, "patron_npub": patron, "peers": peers}


async def revoke_peer_grant_tool(rt: Any, npub: str, peer_npub: str) -> dict[str, Any]:
    """Withdraw one peer's grant. The patron may, and so may the peer itself."""
    for value, param in ((npub, "npub"), (peer_npub, "peer_npub")):
        err = rt.npub_validation_error(value, param=param)
        if err is not None:
            return err
    patron = rt.resolve_npub(npub)
    peer = rt.resolve_npub(peer_npub)
    err = await rt.require_caller_proof(patron)
    if err is not None and await rt.prove_caller(peer) is not None:
        return err
    try:
        removed = await (await rt.peer_grant_store()).remove(patron, peer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("peer grant store unavailable: %s", type(exc).__name__)
        return _store_unavailable()
    return {
        "success": True,
        "revoked": removed,
        "message": (
            f"{_short(peer)} can no longer see or fund {_short(patron)} here."
            if removed else f"{_short(peer)} held no grant from {_short(patron)} here."
        ),
    }


def _store_unavailable() -> dict[str, Any]:
    from tollbooth.constants import ErrorCode
    return {
        "success": False,
        "error_code": ErrorCode.WARMING_UP,
        "error": "Peer grants could not be read just now. Retry in 10-15 seconds.",
    }
