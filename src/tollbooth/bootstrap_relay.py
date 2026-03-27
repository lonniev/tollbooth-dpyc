"""Bootstrap config delivery and retrieval via Nostr relays.

The Authority sends the operator's Neon URL as a NIP-44 encrypted DM
at registration time. The operator reads it on cold start using only
its nsec — no OAuth, no MCP-to-MCP calls, no additional env vars.

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


def send_bootstrap_config(
    *,
    authority_nsec: str,
    operator_npub: str,
    config: dict[str, str],
    relays: list[str] | None = None,
) -> bool:
    """Send bootstrap config to an operator via NIP-04 encrypted DM.

    Called by the Authority after provisioning a Neon schema.
    The config is encrypted so only the operator can read it.

    Uses NIP-04 (simpler, widely supported) rather than NIP-44/NIP-17
    because this is infrastructure config, not a secret credential.

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

    # Build NIP-04 DM event (kind 4)
    event = Event(
        pubkey=auth_pk.public_key.hex(),
        kind=4,
        content=ciphertext,
        created_at=int(time.time()),
        tags=[["p", op_pubkey_hex]],
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
            # Read OK response
            resp = ws.recv()
            ws.close()
            if "true" in resp.lower() or "ok" in resp.lower():
                published += 1
                logger.info("Bootstrap config sent to %s via %s", operator_npub[:16], relay_url)
        except Exception as exc:
            logger.debug("Failed to publish bootstrap config to %s: %s", relay_url, exc)

    return published > 0


def receive_bootstrap_config(
    *,
    operator_nsec: str,
    authority_pubkey_hex: str,
    relays: list[str] | None = None,
    max_age_seconds: int = 30 * 24 * 3600,  # 30 days
) -> dict[str, str] | None:
    """Read bootstrap config from Nostr relays.

    Called by the operator on cold start. Polls relays for NIP-04
    encrypted DMs from the Authority, decrypts, and returns the
    config dict.

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
    since = int(time.time()) - max_age_seconds

    # Build subscription filter: NIP-04 DMs (kind 4) from authority to operator
    sub_filter = {
        "kinds": [4],
        "authors": [authority_pubkey_hex],
        "#p": [op_pubkey_hex],
        "since": since,
    }

    import websocket  # type: ignore[import-untyped]

    best_config: dict[str, str] | None = None
    best_ts = 0

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
                    event_data = msg[2]
                    try:
                        plaintext = nip04_decrypt(
                            private_key_hex=op_pk.hex(),
                            public_key_hex=authority_pubkey_hex,
                            ciphertext=event_data["content"],
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
                        logger.debug("Failed to decrypt bootstrap DM: %s", exc)

            ws.send(json.dumps(["CLOSE", sub_id]))
            ws.close()

            # If we found config, no need to check more relays
            if best_config is not None:
                break

        except Exception as exc:
            logger.debug("Failed to poll %s for bootstrap config: %s", relay_url, exc)

    return best_config
