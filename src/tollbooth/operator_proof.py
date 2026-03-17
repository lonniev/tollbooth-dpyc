"""Operator proof verification — Nostr kind-27235 event validation.

Verifies that a caller can prove operator identity by presenting a signed
Nostr kind-27235 event (NIP-98 style).  The ``u`` tag carries the tool name
instead of a URL and ``created_at`` must be within 60 seconds of the current
server time.

Servers **only verify** — signing happens exclusively on the client where
the operator's nsec lives.

Dependencies: ``pynostr`` (already available via ``tollbooth-dpyc[nostr]``).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

OPERATOR_PROOF_KIND = 27235
"""NIP-98 HTTP Auth event kind, repurposed for MCP operator proofs."""

PROOF_WINDOW_SECONDS = 60
"""Maximum age (in seconds) of a valid operator proof event."""


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to a hex pubkey string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey.from_npub(npub).hex()


def verify_identity_proof(
    proof_json: str,
    expected_npub: str,
    tool_name: str,
    window_seconds: int = PROOF_WINDOW_SECONDS,
) -> bool:
    """Verify a Nostr kind-27235 identity proof for any npub.

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

    # 1. Parse JSON
    try:
        event_dict = json.loads(proof_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("identity_proof: invalid JSON")
        return False

    # 2. Construct Event and verify Schnorr signature
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

    # 3. Verify event kind
    if event.kind != OPERATOR_PROOF_KIND:
        logger.debug("identity_proof: wrong kind %d (expected %d)", event.kind, OPERATOR_PROOF_KIND)
        return False

    # 4. Verify pubkey matches expected npub
    try:
        expected_hex = _npub_to_hex(expected_npub)
    except Exception:
        logger.debug("identity_proof: invalid expected npub")
        return False

    if event.pubkey != expected_hex:
        logger.debug("identity_proof: pubkey mismatch")
        return False

    # 5. Check ``u`` tag contains the tool name
    u_values = [tag[1] for tag in event.tags if len(tag) >= 2 and tag[0] == "u"]
    if tool_name not in u_values:
        logger.debug("identity_proof: tool_name %r not in u tags %r", tool_name, u_values)
        return False

    # 6. Check created_at within window
    now = time.time()
    age = abs(now - event.created_at)
    if age > window_seconds:
        logger.debug("identity_proof: expired (age=%.1fs, window=%ds)", age, window_seconds)
        return False

    return True


def verify_operator_proof(
    proof_json: str,
    operator_npub: str,
    tool_name: str,
) -> bool:
    """Verify a Nostr kind-27235 operator proof.

    Thin wrapper around :func:`verify_identity_proof` that uses the
    operator's npub and the default proof window.

    Args:
        proof_json: JSON string of the signed Nostr event.
        operator_npub: The operator's npub (bech32) to verify against.
        tool_name: The MCP tool name that must appear in the ``u`` tag.

    Returns:
        ``True`` if the proof is valid, ``False`` otherwise.
    """
    return verify_identity_proof(proof_json, operator_npub, tool_name)
