"""Bootstrap config delivery and retrieval via Nostr relays.

The Authority publishes the operator's Neon URL as a NIP-33 parameterized-
replaceable event (kind 30078, NIP-04-encrypted content), scoped by a
per-operator ``d`` tag. Relays keep only the latest replaceable per
(author, kind, ``d``), but free public relays still drop events, so a weekly
audit (``audit_bootstrap_configs``, run by a GitHub Action) has the Authority
republish any config held too thinly or too long ago, and an operator spreads
its own config when it finds it thin. The operator reads it on
cold start using only its nsec — no OAuth, no MCP-to-MCP calls, no additional
env vars.

Send side (Authority):
    send_bootstrap_config(
        authority_nsec="nsec1...",
        operator_npub="npub1...",
        config={"neon_database_url": "postgres://..."},
    )

Receive side (Operator) — nsec only, Authority discovered from the event:
    read = receive_bootstrap_config(operator_nsec="nsec1...")
    neon_url = read.config.get("neon_database_url")

When ``relays`` is omitted, both sides draw the relay set from the DPYC
community registry (``relay_registry.get_relays``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from tollbooth.relay_fanout import fan_out

logger = logging.getLogger(__name__)

BOOTSTRAP_CONFIG_TAG = "dpyc-bootstrap-config"

# One budget for a relay's connect AND its read-to-EOSE, so an abandoned
# worker ends on its own within it. Healthy relays answer in well under a
# second; six seconds is patience for a slow one, not for a dead one.
_READ_BUDGET_SECONDS = 6.0
# After the first config arrives, how much longer to listen for a newer one.
_SETTLE_SECONDS = 1.5
# A config held by fewer relays than this is one relay outage away from an
# operator that cannot start (six were on nos.lol alone, 2026-09-28).
MIN_HOLDERS = 3


@dataclass(frozen=True, slots=True)
class ConfigRead:
    """What one read of the relays found.

    ``event`` is the Authority-signed event that supplied ``config``, and
    ``holders`` the relays that served that very event — so a reader can tell
    when its config is thinly held and re-broadcast it (:func:`broadcast_signed_event`).
    """

    config: dict[str, str] | None
    author_hex: str | None
    diag: str
    event: dict[str, Any] | None = None
    holders: tuple[str, ...] = ()

    @property
    def thin(self) -> bool:
        return self.event is not None and len(self.holders) < MIN_HOLDERS


def _config_d_tag(op_pubkey_hex: str) -> str:
    """The NIP-33 ``d`` tag scoping the Authority's config for one operator.

    Parameterized-replaceable identity is (author, kind, d); namespacing the
    ``d`` by operator pubkey gives the Authority exactly one replaceable config
    event per operator — re-publishing replaces it in place.
    """
    return f"{BOOTSTRAP_CONFIG_TAG}:{op_pubkey_hex}"


# One budget for a relay's connect AND its NIP-20 OK read. Healthy relays
# answer in well under a second; ten seconds is patience for a slow one.
_PUBLISH_BUDGET_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class PublishResult:
    """How many relays accepted vs refused a bootstrap config publish.

    Falsy when nothing accepted, so legacy ``if sent:`` call sites keep working.
    """

    accepted: int
    rejected: int

    def __bool__(self) -> bool:
        return self.accepted > 0

    def __iter__(self):
        # Unpack as ``accepted, rejected = send_bootstrap_config(...)``.
        yield self.accepted
        yield self.rejected


def send_bootstrap_config(
    *,
    authority_nsec: str,
    operator_npub: str,
    config: dict[str, str],
    relays: list[str] | None = None,
) -> PublishResult:
    """Publish bootstrap config for an operator as a NIP-33 replaceable event.

    Called by the Authority after provisioning a Neon schema. The config is
    published as a NIP-78 application-data event (kind 30078), which is a NIP-33
    parameterized-replaceable event: relays keep only the latest per
    (Authority, kind, ``d``-tag), so it does NOT age off the way a stream of
    kind-4 DMs does. Content is NIP-04-encrypted so only the operator can read
    it (infrastructure config, not a personal credential).

    Publishes via :func:`relay_fanout.fan_out` so the weekly audit finishes
    inside one request instead of a serial 10 s-per-relay walk.

    Returns a :class:`PublishResult` (``accepted``, ``rejected``). Falsy when
    nothing accepted, so ``if sent:`` call sites keep working.
    """
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]

    from tollbooth.nip04 import encrypt as nip04_encrypt
    from tollbooth.relay_registry import get_relays
    relay_urls = relays or get_relays()

    # Derive authority keys
    if authority_nsec.startswith("nsec1"):
        auth_pk = PrivateKey.from_nsec(authority_nsec)
    else:
        auth_pk = PrivateKey(bytes.fromhex(authority_nsec))

    # Resolve operator pubkey hex
    if operator_npub.startswith("npub1"):
        op_pubkey_hex = PublicKey.from_npub(operator_npub).hex()
    else:
        op_pubkey_hex = operator_npub

    # Build payload. The Authority stamps its own npub inside the encrypted
    # body so the operator learns — authoritatively — who signed its config
    # without a prior registry lookup. (The event's own ``pubkey`` says the same
    # thing in the clear; the operator cross-checks both against the Oracle.)
    payload = json.dumps({
        "type": BOOTSTRAP_CONFIG_TAG,
        "config": config,
        "authority_npub": auth_pk.public_key.bech32(),
        "ts": int(time.time()),
    })

    # NIP-04 encrypt
    ciphertext = nip04_encrypt(
        private_key_hex=auth_pk.hex(),
        public_key_hex=op_pubkey_hex,
        plaintext=payload,
    )

    # Build NIP-33 parameterized-replaceable event (kind 30078, NIP-78 app data).
    # The `d` tag scopes one replaceable config per operator; `p` lets the
    # operator also be located as recipient.
    event = Event(
        pubkey=auth_pk.public_key.hex(),
        kind=30078,
        content=ciphertext,
        created_at=int(time.time()),
        tags=[["d", _config_d_tag(op_pubkey_hex)], ["p", op_pubkey_hex]],
    )
    event.sign(auth_pk.hex())
    return broadcast_signed_event(event.to_dict(), relay_urls)


def broadcast_signed_event(event: dict[str, Any], relays: list[str]) -> PublishResult:
    """Publish an already-signed event, unchanged, to every relay at once.

    Anyone may relay a signed Nostr event — each relay verifies the author's
    signature — so an operator can spread the Authority's config event without
    holding any key but its own.
    """
    event_msg = json.dumps(["EVENT", event])
    outcomes = fan_out(
        relays,
        lambda url: _publish_one(url, event_msg),
        deadline=_PUBLISH_BUDGET_SECONDS,
    )

    accepted = 0
    rejected = 0
    for o in outcomes:
        if o.state == "ok" and o.value is True:
            accepted += 1
            logger.info("Bootstrap config event %s accepted by %s", event["id"][:12], o.relay)
        else:
            rejected += 1
            detail = o.error if o.state != "ok" else "rejected"
            # Abandoned / refused relays are weather; a true NIP-20 rejection is news.
            log = logger.warning if o.state == "ok" else logger.debug
            log(
                "Relay %s did not accept bootstrap config event %s: %s",
                o.relay, event["id"][:12], detail,
            )

    return PublishResult(accepted, rejected)


def _publish_one(relay_url: str, event_msg: str) -> bool:
    """Publish one EVENT to one relay; return True only on NIP-20 OK/true.

    Runs on a fan-out worker thread. Transport failures raise (the relay
    refused us); a parsed rejection returns False (the relay answered no).
    """
    import websocket  # type: ignore[import-untyped]

    ws = websocket.create_connection(relay_url, timeout=_PUBLISH_BUDGET_SECONDS)
    try:
        ws.send(event_msg)
        # Read OK response — NIP-20: ["OK", <event_id>, <true|false>, <message>].
        # Parse strictly: a rejection like ["OK", id, false, "rate-limited"]
        # must not count as published (substring matching on "ok" did,
        # silently dropping relays from the bootstrap config's coverage).
        resp = ws.recv()
        try:
            reply = json.loads(resp)
            return (
                isinstance(reply, list)
                and len(reply) >= 3
                and reply[0] == "OK"
                and reply[2] is True
            )
        except (json.JSONDecodeError, TypeError):
            return False
    finally:
        ws.close()


def receive_bootstrap_config(
    *,
    operator_nsec: str,
    relays: list[str] | None = None,
    expected_authority_hex: str | None = None,
) -> ConfigRead:
    """Read bootstrap config from Nostr relays using ONLY the operator nsec.

    Called by the operator on cold start. Polls relays for the config event
    (kind 30078) scoped to this operator's own ``d`` tag — the operator does
    NOT need to know its Authority in advance. Each candidate is decrypted with
    the **event's own author** as the NIP-04 counterparty, so the Authority npub
    is *discovered* from the event rather than supplied. No age window: a
    replaceable event is the current config however old it is.

    ``expected_authority_hex`` (optional): when the caller already knows the
    trusted Authority (e.g. verified via the Oracle), only events from that
    author are accepted — a spoofed event carrying this operator's ``d`` tag
    from any other author is ignored. When ``None``, newest-``ts`` wins across
    all authors and the caller is expected to verify the returned author
    out-of-band (Oracle cross-check) before trusting the config.

    Returns a :class:`ConfigRead`: the winning config, its author's hex pubkey
    (``None`` when no config was found), the diagnostics, and the signed event
    with the relays that served it.
    """
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]

    from tollbooth.relay_registry import get_relays
    relay_urls = relays or get_relays()

    # Derive operator keys
    if operator_nsec.startswith("nsec1"):
        op_pk = PrivateKey.from_nsec(operator_nsec)
    else:
        op_pk = PrivateKey(bytes.fromhex(operator_nsec))

    op_pubkey_hex = op_pk.public_key.hex()
    op_privkey_hex = op_pk.hex()

    # Sanity check — ensure hex strings are valid
    try:
        bytes.fromhex(op_privkey_hex)
        if expected_authority_hex is not None:
            bytes.fromhex(expected_authority_hex)
    except ValueError as e:
        logger.error("Bootstrap key hex invalid: priv=%s... err=%s",
                     op_privkey_hex[:8], e)
        return ConfigRead(None, None, f"key hex error: {e}")

    # Build subscription filter: the config event (kind 30078) scoped to THIS
    # operator's `d` tag. No `authors` clause — the `d` tag is already namespaced
    # by operator pubkey, so the operator finds its own config without knowing
    # who signed it. No `since` — a replaceable is the current config however old.
    sub_filter = {
        "kinds": [30078],
        "#d": [_config_d_tag(op_pubkey_hex)],
    }

    outcomes = fan_out(
        relay_urls,
        lambda url: _read_one(url, sub_filter, op_privkey_hex, expected_authority_hex),
        deadline=_READ_BUDGET_SECONDS,
        settle=_SETTLE_SECONDS,
        accept=lambda read: read.config is not None,
    )

    # Newest ``ts`` wins across every relay that answered. A re-published
    # replaceable propagates unevenly, so one relay may still serve an older
    # revision — and a stale config can carry a rotated-away role password,
    # which fails worse than no config at all. The settle window above is what
    # lets a slightly slower relay still cast its vote.
    best_config: dict[str, str] | None = None
    best_author: str | None = None
    best_event: dict[str, Any] | None = None
    best_ts = 0
    served: list[tuple[str, str]] = []  # (relay, id of the event it supplied)
    events_found = 0
    undecryptable = 0
    relay_errors: list[str] = []
    abandoned = 0
    for o in outcomes:
        if o.state == "abandoned":
            abandoned += 1
            continue
        if o.state == "error":
            relay_errors.append(f"{o.relay}: {o.error}")
            continue
        read: _RelayRead = o.value
        events_found += read.events
        undecryptable += read.undecryptable
        if read.event is not None:
            served.append((o.relay, read.event["id"]))
        if read.config is not None and read.author_hex and read.ts > best_ts:
            best_config, best_author, best_ts = read.config, read.author_hex, read.ts
            best_event = read.event
            logger.info(
                "Bootstrap config received from %s via %s (ts=%d)",
                read.author_hex[:16], o.relay, read.ts,
            )

    # ``errors=[…]`` names relays that REFUSED us; the Oracle re-measures each
    # one. A slow relay and a relay serving an undecryptable event both answered,
    # so they are counted before that bracket, never inside it.
    diag = f"relays={len(relay_urls)}, events={events_found}"
    if abandoned:
        diag += f", slow={abandoned}"
    if undecryptable:
        diag += f", undecryptable={undecryptable}"
    if relay_errors:
        diag += f", errors=[{'; '.join(relay_errors)}]"

    if best_config is None:
        logger.warning("Bootstrap relay poll failed: %s", diag)

    holders = tuple(r for r, eid in served if best_event and eid == best_event["id"])
    return ConfigRead(best_config, best_author, diag, best_event, holders)


def config_coverage(
    authority_hex: str,
    operator_hexes: list[str],
    relays: list[str],
) -> dict[str, tuple[str, ...]]:
    """Which relays hold this Authority's config event for each operator.

    One subscription per relay, all relays at once. The ``authors`` clause
    means a copy signed by anyone else never counts. Nothing is decrypted:
    this measures where the events are, not what they say.
    """
    wanted = {_config_d_tag(h): h for h in operator_hexes}
    sub_filter = {"kinds": [30078], "authors": [authority_hex], "#d": list(wanted)}
    outcomes = fan_out(
        relays,
        lambda url: _held_on(url, sub_filter),
        deadline=_READ_BUDGET_SECONDS,
    )
    holders: dict[str, list[str]] = {h: [] for h in operator_hexes}
    for o in outcomes:
        if o.state != "ok":
            continue
        for d_tag in o.value:
            if d_tag in wanted:
                holders[wanted[d_tag]].append(o.relay)
    return {h: tuple(rs) for h, rs in holders.items()}


def _held_on(relay_url: str, sub_filter: dict[str, Any]) -> set[str]:
    """The ``d`` tags one relay holds for this filter, read to EOSE."""
    import websocket  # type: ignore[import-untyped]

    t0 = time.monotonic()
    ws = websocket.create_connection(relay_url, timeout=_READ_BUDGET_SECONDS)
    found: set[str] = set()
    try:
        sub_id = f"coverage-{int(time.time())}"
        ws.send(json.dumps(["REQ", sub_id, sub_filter]))
        while (remaining := _READ_BUDGET_SECONDS - (time.monotonic() - t0)) > 0:
            ws.settimeout(remaining)
            msg = json.loads(ws.recv())
            if msg[0] == "EOSE":
                break
            if msg[0] == "EVENT" and len(msg) >= 3:
                found.update(t[1] for t in msg[2].get("tags", []) if t[:1] == ["d"] and len(t) > 1)
        ws.send(json.dumps(["CLOSE", sub_id]))
    finally:
        ws.close()
    return found


@dataclass(frozen=True, slots=True)
class _RelayRead:
    """One relay's answer: the newest decryptable config it served, plus counts."""

    config: dict[str, str] | None
    author_hex: str | None
    ts: int
    events: int
    undecryptable: int
    event: dict[str, Any] | None = None


def _read_one(
    relay_url: str,
    sub_filter: dict[str, Any],
    op_privkey_hex: str,
    expected_authority_hex: str | None,
) -> _RelayRead:
    """Subscribe to one relay and read until EOSE, inside one time budget.

    Runs on a fan-out worker thread. Transport failures raise (the relay
    refused us); an event that fails to decrypt is counted, not raised (the
    relay answered, it just served something that is not ours).
    """
    import websocket  # type: ignore[import-untyped]

    from tollbooth.nip04 import decrypt as nip04_decrypt

    t0 = time.monotonic()
    ws = websocket.create_connection(relay_url, timeout=_READ_BUDGET_SECONDS)
    config: dict[str, str] | None = None
    author: str | None = None
    signed: dict[str, Any] | None = None
    best_ts = 0
    events = 0
    undecryptable = 0
    try:
        sub_id = f"bootstrap-{int(time.time())}"
        ws.send(json.dumps(["REQ", sub_id, sub_filter]))
        while (remaining := _READ_BUDGET_SECONDS - (time.monotonic() - t0)) > 0:
            ws.settimeout(remaining)
            msg = json.loads(ws.recv())
            if msg[0] == "EOSE":
                break
            if msg[0] != "EVENT" or len(msg) < 3:
                continue
            events += 1
            event_data = msg[2]
            author_hex = event_data.get("pubkey", "")
            # When a trusted author is known, ignore anyone else's event bearing
            # our `d` tag (spoof guard).
            if expected_authority_hex and author_hex != expected_authority_hex:
                continue
            try:
                payload = json.loads(nip04_decrypt(
                    ciphertext_with_iv=event_data["content"],
                    private_key_hex=op_privkey_hex,
                    public_key_hex=author_hex,
                ))
            except Exception as exc:  # noqa: BLE001
                undecryptable += 1
                logger.warning("Bootstrap event on %s did not decrypt: %s", relay_url, exc)
                continue
            if payload.get("type") != BOOTSTRAP_CONFIG_TAG:
                continue
            ts = payload.get("ts", event_data.get("created_at", 0))
            if ts > best_ts:
                config, author, best_ts = payload.get("config", {}), author_hex, ts
                signed = event_data
        ws.send(json.dumps(["CLOSE", sub_id]))
    finally:
        ws.close()
    return _RelayRead(config, author, best_ts, events, undecryptable, signed)
