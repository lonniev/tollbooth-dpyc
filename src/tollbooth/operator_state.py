"""An operator's own record of its bootstrap, kept on Nostr.

The Oracle may run, then not run, then run again. An operator bootstrapped while
it was up writes down what it learned: who its Authority is and which relays it
used. A cold start during an Oracle outage reads that record back and carries on.

The record is a NIP-33 replaceable event (kind 30078) the operator signs with its
own nsec and NIP-04-encrypts to itself. Anyone can publish under a ``d`` tag, so a
reader takes only an event whose author is itself and whose signature verifies.

The Oracle's answer always wins when it gives one. A record the Oracle last
confirmed more than :data:`MAX_GLIDE_SECONDS` ago is not used, and an explicit
"no Authority" from the Oracle withdraws it. Neon credentials never go in here:
the Authority's event stays their only source, so a rotated password cannot be
undone by an old copy.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from tollbooth.relay_fanout import fan_out

logger = logging.getLogger(__name__)

OPERATOR_STATE_TAG = "dpyc-operator-state"

# How long an operator may trust what the Oracle last confirmed. Long enough to
# glide through an outage; a ban cannot reach a gliding operator until it ends.
MAX_GLIDE_SECONDS = 30 * 24 * 60 * 60

# Rewrite an unchanged record this often, so its age keeps measuring the last
# time the Oracle confirmed it.
_REFRESH_SECONDS = 24 * 60 * 60

_READ_BUDGET_SECONDS = 6.0
_SETTLE_SECONDS = 1.5


@dataclass(frozen=True, slots=True)
class OperatorState:
    """What the Oracle last confirmed for this operator."""

    authority_npub: str
    relays: tuple[str, ...]
    confirmed_at: int


def _keys(nsec: str) -> tuple[str, str]:
    """``(private key hex, public key hex)`` from a hex or bech32 nsec."""
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]

    key = PrivateKey.from_nsec(nsec) if nsec.startswith("nsec1") else PrivateKey(bytes.fromhex(nsec))
    return key.hex(), key.public_key.hex()


def _d_tag(pubkey_hex: str) -> str:
    return f"{OPERATOR_STATE_TAG}:{pubkey_hex}"


def _publish(nsec: str, body: dict[str, Any], relays: list[str]) -> int:
    """Sign and broadcast the record; how many relays accepted it."""
    from pynostr.event import Event  # type: ignore[import-untyped]

    from tollbooth.bootstrap_relay import broadcast_signed_event
    from tollbooth.nip04 import encrypt as nip04_encrypt

    private_hex, public_hex = _keys(nsec)
    event = Event(
        pubkey=public_hex,
        kind=30078,
        content=nip04_encrypt(
            private_key_hex=private_hex,
            public_key_hex=public_hex,
            plaintext=json.dumps({"type": OPERATOR_STATE_TAG, **body}),
        ),
        created_at=int(time.time()),
        tags=[["d", _d_tag(public_hex)]],
    )
    event.sign(private_hex)
    return broadcast_signed_event(event.to_dict(), relays).accepted


def _own_body(event: dict[str, Any], private_hex: str, public_hex: str) -> dict[str, Any] | None:
    """The decrypted record, or ``None`` unless this operator wrote it."""
    from pynostr.event import Event  # type: ignore[import-untyped]

    from tollbooth.nip04 import decrypt as nip04_decrypt

    try:
        if event.get("pubkey") != public_hex or ["d", _d_tag(public_hex)] not in event.get("tags", []):
            return None
        if not Event.from_dict(event).verify():
            return None
        body = json.loads(nip04_decrypt(
            ciphertext_with_iv=event["content"],
            private_key_hex=private_hex,
            public_key_hex=public_hex,
        ))
    except Exception as exc:  # noqa: BLE001 — a relay can serve anything
        logger.warning("Operator state event rejected: %s", type(exc).__name__)
        return None
    return body if isinstance(body, dict) and body.get("type") == OPERATOR_STATE_TAG else None


def _read_one(
    relay_url: str, sub_filter: dict[str, Any], private_hex: str, public_hex: str,
) -> tuple[int, dict[str, Any] | None]:
    """The newest record one relay holds, as ``(created_at, body)``."""
    import websocket  # type: ignore[import-untyped]

    t0 = time.monotonic()
    ws = websocket.create_connection(relay_url, timeout=_READ_BUDGET_SECONDS)
    newest: tuple[int, dict[str, Any] | None] = (0, None)
    try:
        sub_id = f"operator-state-{int(time.time())}"
        ws.send(json.dumps(["REQ", sub_id, sub_filter]))
        while (remaining := _READ_BUDGET_SECONDS - (time.monotonic() - t0)) > 0:
            ws.settimeout(remaining)
            msg = json.loads(ws.recv())
            if msg[0] == "EOSE":
                break
            if msg[0] != "EVENT" or len(msg) < 3:
                continue
            created_at = int(msg[2].get("created_at", 0))
            if created_at > newest[0]:
                body = _own_body(msg[2], private_hex, public_hex)
                if body is not None:
                    newest = (created_at, body)
        ws.send(json.dumps(["CLOSE", sub_id]))
    finally:
        ws.close()
    return newest


def _latest_body(nsec: str, relays: list[str]) -> dict[str, Any] | None:
    """The newest record any answering relay holds, withdrawn or not."""
    private_hex, public_hex = _keys(nsec)
    sub_filter = {"kinds": [30078], "authors": [public_hex], "#d": [_d_tag(public_hex)]}
    outcomes = fan_out(
        relays,
        lambda url: _read_one(url, sub_filter, private_hex, public_hex),
        deadline=_READ_BUDGET_SECONDS,
        settle=_SETTLE_SECONDS,
        accept=lambda read: read[1] is not None,
    )
    newest: tuple[int, dict[str, Any] | None] = (0, None)
    for outcome in outcomes:
        if outcome.state == "ok" and outcome.value[0] > newest[0]:
            newest = outcome.value
    return newest[1]


def _as_state(body: dict[str, Any] | None) -> OperatorState | None:
    """A usable record from a decrypted body; ``None`` when withdrawn or malformed."""
    if body is None:
        return None
    authority = body.get("authority_npub")
    relays = body.get("relays")
    confirmed_at = body.get("confirmed_at")
    if not (
        isinstance(authority, str) and authority.startswith("npub1")
        and isinstance(relays, list)
        and isinstance(confirmed_at, int)
    ):
        return None
    usable = tuple(r for r in relays if isinstance(r, str) and r.startswith("wss://"))
    return OperatorState(authority, usable, confirmed_at) if usable else None


def read_operator_state(nsec: str, relays: list[str]) -> OperatorState | None:
    """This operator's record, or ``None`` when absent, withdrawn or too old."""
    state = _as_state(_latest_body(nsec, relays))
    if state is None:
        return None
    if time.time() - state.confirmed_at > MAX_GLIDE_SECONDS:
        logger.warning("Operator state is too old to glide on; waiting for the Oracle.")
        return None
    return state


def record_operator_state(nsec: str, authority_npub: str, relays: list[str]) -> None:
    """Record what the Oracle just confirmed, when it differs or has aged."""
    current = _as_state(_latest_body(nsec, relays))
    if (
        current is not None
        and current.authority_npub == authority_npub
        and list(current.relays) == relays
        and time.time() - current.confirmed_at < _REFRESH_SECONDS
    ):
        return
    accepted = _publish(
        nsec,
        {"authority_npub": authority_npub, "relays": relays, "confirmed_at": int(time.time())},
        relays,
    )
    logger.info("Operator state recorded on %d relay(s).", accepted)


def withdraw_operator_state(nsec: str, relays: list[str]) -> None:
    """Replace a usable record with an empty one: the Oracle said no Authority."""
    if _as_state(_latest_body(nsec, relays)) is None:
        return
    accepted = _publish(nsec, {"withdrawn_at": int(time.time())}, relays)
    logger.warning("Operator state withdrawn on %d relay(s).", accepted)
