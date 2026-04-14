"""Identity proof — Nostr kind-27235 event for proving npub ownership.

Used by both operators (RESTRICTED tool access) and patrons (high-value
tool authorization). The proof is a Schnorr-signed Nostr event with the
tool name in the ``u`` tag and a freshness window.

Proof format (kind 27235, NIP-98 style):
    {
        "pubkey": "<hex_pubkey>",
        "kind": 27235,
        "content": "",
        "created_at": <unix_timestamp>,
        "tags": [["u", "<tool_name>"]],
        "sig": "<schnorr_signature>"
    }

Dependencies: ``pynostr`` (available via ``tollbooth-dpyc[nostr]``).
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

PROOF_EVENT_KIND = 27235
"""NIP-98 HTTP Auth event kind, repurposed for MCP identity proofs."""

DEFAULT_WINDOW_SECONDS = 60
"""Maximum age (in seconds) of a valid proof event."""

OWNERSHIP_SENTINEL = "npub_ownership"
"""Sentinel tool name for npub ownership proofs (not tied to a specific tool)."""


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to a hex pubkey string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey.from_npub(npub).hex()


def create_proof(nsec: str, tool_name: str) -> str:
    """Create a signed kind-27235 identity proof.

    Args:
        nsec: Nostr private key (bech32 nsec1... or hex).
        tool_name: The MCP tool name to embed in the ``u`` tag.

    Returns:
        JSON string of the signed Nostr event.
    """
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]
    from pynostr.event import Event  # type: ignore[import-untyped]

    if nsec.startswith("nsec1"):
        pk = PrivateKey.from_nsec(nsec)
    else:
        pk = PrivateKey(bytes.fromhex(nsec))

    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()),
        tags=[["u", tool_name]],
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


def create_ownership_proof(nsec: str) -> str:
    """Create a kind-27235 proof for npub ownership (no specific tool).

    Args:
        nsec: Nostr private key (bech32 nsec1... or hex).

    Returns:
        JSON string of the signed Nostr event with ``u=npub_ownership``.
    """
    return create_proof(nsec, OWNERSHIP_SENTINEL)


def verify_proof(
    proof_json: str,
    expected_npub: str,
    tool_name: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> bool:
    """Verify a Nostr kind-27235 identity proof.

    Works for any npub holder — operator or patron.

    Args:
        proof_json: JSON string of the signed Nostr event.
        expected_npub: The npub (bech32) the proof must be signed by.
        tool_name: The MCP tool name that must appear in the ``u`` tag.
        window_seconds: Maximum age (in seconds) of the proof event.

    Returns:
        ``True`` if the proof is valid, ``False`` otherwise.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pynostr not installed — cannot verify identity proof")
        return False

    try:
        event_dict = json.loads(proof_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("identity_proof: invalid JSON")
        return False

    try:
        event = Event.from_dict(event_dict)
    except Exception:
        logger.debug("identity_proof: invalid Nostr event structure")
        return False

    try:
        if not event.verify():
            logger.debug("identity_proof: signature verification failed")
            return False
    except Exception:
        logger.debug("identity_proof: signature verification error")
        return False

    if event.kind != PROOF_EVENT_KIND:
        logger.debug("identity_proof: wrong kind %d (expected %d)", event.kind, PROOF_EVENT_KIND)
        return False

    try:
        expected_hex = _npub_to_hex(expected_npub)
    except Exception:
        logger.debug("identity_proof: invalid expected npub")
        return False

    if event.pubkey != expected_hex:
        logger.debug("identity_proof: pubkey mismatch")
        return False

    u_values = [tag[1] for tag in event.tags if len(tag) >= 2 and tag[0] == "u"]
    if tool_name not in u_values:
        logger.debug("identity_proof: tool_name %r not in u tags %r", tool_name, u_values)
        return False

    now = time.time()
    age = abs(now - event.created_at)
    if age > window_seconds:
        logger.debug("identity_proof: expired (age=%.1fs, window=%ds)", age, window_seconds)
        return False

    return True
