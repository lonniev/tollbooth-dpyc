"""Bootstrap config delivery and retrieval via Nostr relays.

The Authority publishes the operator's Neon URL as a NIP-33 parameterized-
replaceable event (kind 30078, NIP-04-encrypted content), scoped by a
per-operator ``d`` tag. Because relays keep only the latest replaceable per
(author, kind, ``d``), the config does NOT age off the way a stream of kind-4
DMs does — there is no heartbeat and no re-publish schedule to maintain. The
operator reads it on cold start using only its nsec — no OAuth, no MCP-to-MCP
calls, no additional env vars.

Send side (Authority):
    send_bootstrap_config(
        authority_nsec="nsec1...",
        operator_npub="npub1...",
        config={"neon_database_url": "postgres://..."},
        relays=["wss://nostr.wine", ...],
    )

Receive side (Operator):
    config = receive_bootstrap_config(
        operator_nsec="nsec1...",
        authority_pubkey_hex="abc123...",
        relays=["wss://nostr.wine", ...],
    )
    neon_url = config.get("neon_database_url")
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

BOOTSTRAP_CONFIG_TAG = "dpyc-bootstrap-config"

# Default relays for bootstrap config delivery
BOOTSTRAP_RELAYS = [
    "wss://relay.primal.net",
    "wss://nos.lol",
    "wss://relay.damus.io",
    "wss://relay.nostr.band",
]


def _config_d_tag(op_pubkey_hex: str) -> str:
    """The NIP-33 ``d`` tag scoping the Authority's config for one operator.

    Parameterized-replaceable identity is (author, kind, d); namespacing the
    ``d`` by operator pubkey gives the Authority exactly one replaceable config
    event per operator — re-publishing replaces it in place.
    """
    return f"{BOOTSTRAP_CONFIG_TAG}:{op_pubkey_hex}"


def send_bootstrap_config(
    *,
    authority_nsec: str,
    operator_npub: str,
    config: dict[str, str],
    relays: list[str] | None = None,
) -> bool:
    """Publish bootstrap config for an operator as a NIP-33 replaceable event.

    Called by the Authority after provisioning a Neon schema. The config is
    published as a NIP-78 application-data event (kind 30078), which is a NIP-33
    parameterized-replaceable event: relays keep only the latest per
    (Authority, kind, ``d``-tag), so it does NOT age off the way a stream of
    kind-4 DMs does. Content is NIP-04-encrypted so only the operator can read
    it (infrastructure config, not a personal credential).

    Returns True if published to at least one relay.
    """
    from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]
    from pynostr.event import Event  # type: ignore[import-untyped]
    from tollbooth.nip04 import encrypt as nip04_encrypt

    relay_urls = relays or BOOTSTRAP_RELAYS

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

    # Build payload
    payload = json.dumps({
        "type": BOOTSTRAP_CONFIG_TAG,
        "config": config,
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

    # Publish to relays
    import websocket  # type: ignore[import-untyped]

    published = 0
    for relay_url in relay_urls:
        try:
            ws = websocket.create_connection(relay_url, timeout=10)
            msg = json.dumps(["EVENT", event.to_dict()])
            ws.send(msg)
            # Read OK response — NIP-20: ["OK", <event_id>, <true|false>, <message>].
            # Parse strictly: a rejection like ["OK", id, false, "rate-limited"]
            # must not count as published (substring matching on "ok" did,
            # silently dropping relays from the bootstrap config's coverage).
            resp = ws.recv()
            ws.close()
            try:
                reply = json.loads(resp)
                accepted = (
                    isinstance(reply, list)
                    and len(reply) >= 3
                    and reply[0] == "OK"
                    and reply[2] is True
                )
            except (json.JSONDecodeError, TypeError):
                accepted = False
            if accepted:
                published += 1
                logger.info("Bootstrap config sent to %s via %s", operator_npub[:16], relay_url)
            else:
                logger.warning(
                    "Relay %s rejected bootstrap config for %s: %s",
                    relay_url, operator_npub[:16], resp[:200],
                )
        except Exception as exc:
            logger.debug("Failed to publish bootstrap config to %s: %s", relay_url, exc)

    return published > 0


def receive_bootstrap_config(
    *,
    operator_nsec: str,
    authority_pubkey_hex: str,
    relays: list[str] | None = None,
) -> tuple[dict[str, str] | None, str]:
    """Read bootstrap config from Nostr relays.

    Called by the operator on cold start. Polls relays for the Authority's
    NIP-33 parameterized-replaceable config event (kind 30078, scoped by the
    per-operator ``d`` tag), decrypts the NIP-04 content, and returns the config
    dict. No age window: a replaceable event is the current config regardless of
    how long ago it was published.

    Returns the config dict or None if not found.
    """
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]
    from tollbooth.nip04 import decrypt as nip04_decrypt

    relay_urls = relays or BOOTSTRAP_RELAYS

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
        bytes.fromhex(authority_pubkey_hex)
    except ValueError as e:
        logger.error("Bootstrap key hex invalid: priv=%s... pub=%s... err=%s",
                     op_privkey_hex[:8], authority_pubkey_hex[:8], e)
        return None, f"key hex error: {e}"

    # Build subscription filter: the Authority's NIP-33 replaceable config event
    # (kind 30078) scoped to this operator's `d` tag. No `since` — a replaceable
    # is the current config however old it is.
    sub_filter = {
        "kinds": [30078],
        "authors": [authority_pubkey_hex],
        "#d": [_config_d_tag(op_pubkey_hex)],
    }

    import websocket  # type: ignore[import-untyped]

    best_config: dict[str, str] | None = None
    best_ts = 0
    relay_errors: list[str] = []
    events_found = 0

    for relay_url in relay_urls:
        try:
            ws = websocket.create_connection(relay_url, timeout=10)
            sub_id = f"bootstrap-{int(time.time())}"
            ws.send(json.dumps(["REQ", sub_id, sub_filter]))

            # Read events until EOSE
            deadline = time.time() + 10
            while time.time() < deadline:
                raw = ws.recv()
                msg = json.loads(raw)

                if msg[0] == "EOSE":
                    break

                if msg[0] == "EVENT" and len(msg) >= 3:
                    events_found += 1
                    event_data = msg[2]
                    try:
                        plaintext = nip04_decrypt(
                            ciphertext_with_iv=event_data["content"],
                            private_key_hex=op_privkey_hex,
                            public_key_hex=authority_pubkey_hex,
                        )
                        payload = json.loads(plaintext)
                        if payload.get("type") == BOOTSTRAP_CONFIG_TAG:
                            ts = payload.get("ts", event_data.get("created_at", 0))
                            if ts > best_ts:
                                best_config = payload.get("config", {})
                                best_ts = ts
                                logger.info(
                                    "Bootstrap config received from %s via %s (ts=%d)",
                                    authority_pubkey_hex[:16], relay_url, ts,
                                )
                    except Exception as exc:
                        relay_errors.append(f"{relay_url}: decrypt err: {exc}")

            ws.send(json.dumps(["CLOSE", sub_id]))
            ws.close()

            # No early break: poll EVERY relay and let the newest ``ts``
            # win. Re-published replaceable events propagate unevenly, so one
            # relay may still serve an older revision — and a stale config can
            # carry a rotated-away role password, which fails worse than no
            # config at all. Newest-wins across all relays guards that.

        except Exception as exc:
            relay_errors.append(f"{relay_url}: {exc}")

    diag = f"relays={len(relay_urls)}, events={events_found}"
    if relay_errors:
        diag += f", errors=[{'; '.join(relay_errors)}]"

    if best_config is None:
        logger.warning("Bootstrap relay poll failed: %s", diag)

    return best_config, diag
