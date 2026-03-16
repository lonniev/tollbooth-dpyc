#!/usr/bin/env python3
"""Diagnostic script to reproduce self-DM ephemeral agent flow locally.

Simulates the exact schwab-mcp operator onboarding scenario:
1. open_channel(service="schwab-operator", recipient_npub=operator_npub)  [self-DM]
2. User replies TO the ephemeral agent npub
3. receive(sender_npub=operator_npub, service="schwab-operator")

Runs against a local NostrCredentialExchange with no relay I/O.
"""

import asyncio
import json
import logging
import time
from unittest.mock import patch

from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_credentials import (
    NostrCredentialExchange,
    _parse_delimited_credentials,
)

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _schwab_templates() -> dict[str, CredentialTemplate]:
    """Replicate schwab-mcp's two templates."""
    return {
        "schwab-operator": CredentialTemplate(
            service="schwab-operator",
            version=1,
            fields={
                "app_key": FieldSpec(required=True, sensitive=True),
                "secret": FieldSpec(required=True, sensitive=True),
            },
            description="Schwab API app credentials (operator-provided)",
        ),
        "schwab": CredentialTemplate(
            service="schwab",
            version=1,
            fields={
                "token_json": FieldSpec(required=True, sensitive=True),
                "account_hash": FieldSpec(required=True, sensitive=True),
            },
            description="Schwab patron credentials",
        ),
    }


def _make_nip04_event(
    sender_privkey: PrivateKey,
    recipient_pubkey_hex: str,
    payload: dict | str,
) -> dict:
    """Build a kind 4 NIP-04 event addressed to recipient."""
    import base64
    import os

    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.padding import PKCS7

    from tollbooth.nip04 import _get_shared_secret

    raw = payload if isinstance(payload, str) else _to_delimited(payload)
    plaintext = raw.encode("utf-8")

    shared_secret = _get_shared_secret(sender_privkey.hex(), recipient_pubkey_hex)
    iv = os.urandom(16)

    padder = PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    ct_b64 = base64.b64encode(ciphertext).decode()
    iv_b64 = base64.b64encode(iv).decode()

    return {
        "id": f"event_diag_{int(time.time())}",
        "pubkey": sender_privkey.public_key.hex(),
        "kind": 4,
        "created_at": int(time.time()),
        "tags": [["p", recipient_pubkey_hex]],
        "content": f"{ct_b64}?iv={iv_b64}",
        "sig": "0" * 128,
    }


def _to_delimited(payload: dict) -> str:
    return "\n".join(f"{k} = @@@{v}@@@" for k, v in payload.items())


async def main():
    operator_key = PrivateKey()
    operator_npub = operator_key.public_key.bech32()
    operator_hex = operator_key.public_key.hex()

    logger.info("Operator npub: %s", operator_npub)
    logger.info("Operator hex:  %s", operator_hex)

    ex = NostrCredentialExchange(
        nsec=operator_key.nsec,
        relays=["wss://relay.damus.io"],
        templates=_schwab_templates(),
    )

    # Step 1: open_channel — self-DM
    logger.info("=" * 60)
    logger.info("STEP 1: open_channel(service='schwab-operator', recipient=self)")
    with patch.object(ex, "_start_subscription"), \
         patch.object(ex, "send_dm") as mock_send, \
         patch.object(ex, "_send_dm_as") as mock_send_as:
        result = await ex.open_channel(
            "schwab-operator",
            greeting="Schwab MCP operator credentials",
            recipient_npub=operator_npub,
        )

    logger.info("open_channel result: %s", json.dumps(result, indent=2))

    agent_npub = result.get("agent_npub")
    poison = result["poison"]
    logger.info("Ephemeral agent npub: %s", agent_npub)
    logger.info("Poison phrase: %s", poison)

    # Verify ephemeral agent was stored
    resolved_service = ex._resolve_service("schwab-operator")
    agent_key_stored = ex._ephemeral_agents.get((operator_npub, resolved_service))
    logger.info("Ephemeral agent key stored: %s", agent_key_stored is not None)
    if agent_key_stored:
        stored_npub = agent_key_stored.public_key.bech32()
        logger.info("Stored agent npub: %s", stored_npub)
        logger.info("Matches result?  %s", stored_npub == agent_npub)

    # Step 2: Simulate user reply TO the ephemeral agent
    logger.info("=" * 60)
    logger.info("STEP 2: Simulating user reply TO ephemeral agent")

    if agent_key_stored is None:
        logger.error("No ephemeral agent key — cannot proceed")
        return

    ephemeral_pubkey_hex = agent_key_stored.public_key.hex()
    logger.info("User will address DM to ephemeral agent hex: %s", ephemeral_pubkey_hex)

    # The user (who IS the operator) replies with credentials
    # The NIP-04 event is FROM operator, TO ephemeral agent
    credential_payload = {
        "app_key": "test-app-key-123",
        "secret": "test-secret-456",
        "poison": poison,
    }
    reply_event = _make_nip04_event(
        operator_key,            # sender = operator
        ephemeral_pubkey_hex,    # recipient = ephemeral agent
        credential_payload,
    )

    logger.info("Reply event pubkey (sender):      %s", reply_event["pubkey"])
    logger.info("Reply event p-tag (recipient):     %s", reply_event["tags"][0][1])
    logger.info("Reply event kind:                  %s", reply_event["kind"])

    # Inject into the exchange's event buffer
    with ex._lock:
        ex._received_events.append(reply_event)
    logger.info("Injected 1 reply event into buffer")

    # Step 3: receive
    logger.info("=" * 60)
    logger.info("STEP 3: receive(sender_npub=operator, service='schwab-operator')")

    try:
        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(operator_npub, service="schwab-operator")
        logger.info("receive result: %s", json.dumps(result, indent=2))
    except Exception as exc:
        logger.error("receive FAILED: %s", exc)

        # Debug: dump buffer state
        logger.info("--- DEBUG: buffer state ---")
        logger.info("received_events count: %d", len(ex._received_events))
        logger.info("consumed_ids: %s", ex._consumed_ids)
        logger.info("pending_poisons: %s", ex._pending_poisons)
        logger.info("ephemeral_agents: %s",
            {str(k): v.public_key.bech32()[:20] for k, v in ex._ephemeral_agents.items()})


if __name__ == "__main__":
    asyncio.run(main())
