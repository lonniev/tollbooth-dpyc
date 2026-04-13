"""ACL event verification — Nostr kind-30078 signed ACL events.

Operators sign kind-30078 (parameterized replaceable) events that list
npubs authorized to call specific restricted tools.  This module verifies
those signed events and provides the high-level ``check_acl_access``
orchestrator that combines store fetch + signature verification + npub
lookup + caller identity proof verification.

ACL event content format::

    {
        "version": 1,
        "tool_pattern": "set_pricing_model",
        "allowed_npubs": ["npub1abc...", "npub1def..."],
        "expires_at": 1742400000          # optional Unix timestamp
    }

Event tags: ``["d", "<tool_pattern>"]`` (parameterized replaceable d-tag).

``tool_pattern`` uses the **unprefixed** tool name (e.g.,
``set_pricing_model``, not ``brain_set_pricing_model``).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

ACL_EVENT_KIND = 30078
"""NIP-78 application-specific data, used for signed ACL definitions."""


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to a hex pubkey string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey.from_npub(npub).hex()


def verify_acl_event(
    signed_event_json: str,
    operator_npub: str,
) -> dict | None:
    """Verify a kind-30078 ACL event's Schnorr signature and structure.

    Returns the parsed content dict if valid, ``None`` if invalid.
    Checks: kind == 30078, pubkey matches operator_npub, valid sig,
    d-tag present, content is JSON with version/tool_pattern/allowed_npubs.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pynostr not installed — cannot verify ACL event")
        return None

    # 1. Parse JSON
    try:
        event_dict = json.loads(signed_event_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("acl_verify: invalid JSON")
        return None

    # 2. Construct Event and verify Schnorr signature
    try:
        event = Event.from_dict(event_dict)
    except Exception:
        logger.debug("acl_verify: invalid Nostr event structure")
        return None

    try:
        if not event.verify():
            logger.debug("acl_verify: signature verification failed")
            return None
    except Exception:
        logger.debug("acl_verify: signature verification error")
        return None

    # 3. Verify event kind
    if event.kind != ACL_EVENT_KIND:
        logger.debug("acl_verify: wrong kind %d (expected %d)", event.kind, ACL_EVENT_KIND)
        return None

    # 4. Verify pubkey matches operator npub
    try:
        operator_hex = _npub_to_hex(operator_npub)
    except Exception:
        logger.debug("acl_verify: invalid operator npub")
        return None

    if event.pubkey != operator_hex:
        logger.debug("acl_verify: pubkey mismatch")
        return None

    # 5. Check d-tag present
    d_values = [tag[1] for tag in event.tags if len(tag) >= 2 and tag[0] == "d"]
    if not d_values:
        logger.debug("acl_verify: missing d-tag")
        return None

    # 6. Parse content JSON and validate structure
    try:
        content = json.loads(event.content)
    except (json.JSONDecodeError, TypeError):
        logger.debug("acl_verify: invalid content JSON")
        return None

    if not isinstance(content, dict):
        logger.debug("acl_verify: content is not a dict")
        return None

    required_fields = ("version", "tool_pattern", "allowed_npubs")
    for field in required_fields:
        if field not in content:
            logger.debug("acl_verify: missing required field %r", field)
            return None

    if not isinstance(content["allowed_npubs"], list):
        logger.debug("acl_verify: allowed_npubs is not a list")
        return None

    # Cross-check: d-tag should match tool_pattern
    if content["tool_pattern"] not in d_values:
        logger.debug("acl_verify: d-tag %r does not match tool_pattern %r", d_values, content["tool_pattern"])
        return None

    return content


async def check_acl_access(
    acl_store: Any,
    operator_npub: str,
    caller_npub: str,
    tool_name: str,
    caller_identity_proof: str | None = None,
) -> bool:
    """Check if caller_npub is authorized via ACL for tool_name.

    Steps:
        1. Fetch ACL for exact tool_name, then for ``"*"`` wildcard
        2. Verify stored event's operator Schnorr signature
        3. Check caller_npub in allowed_npubs
        4. Check expires_at if present
        5. Verify caller_identity_proof (kind-27235) if provided

    Returns ``True`` if the caller is authorized, ``False`` otherwise.
    """
    from tollbooth.identity_proof import verify_proof as verify_identity_proof

    # 1. Fetch ACL — exact match first, then wildcard
    signed_event_json = await acl_store.fetch_acl(operator_npub, tool_name)
    if signed_event_json is None:
        signed_event_json = await acl_store.fetch_acl(operator_npub, "*")
    if signed_event_json is None:
        logger.debug("check_acl_access: no ACL found for %s / %s", operator_npub, tool_name)
        return False

    # 2. Verify operator signature
    content = verify_acl_event(signed_event_json, operator_npub)
    if content is None:
        logger.debug("check_acl_access: ACL event verification failed")
        return False

    # 3. Check caller in allowed_npubs
    if caller_npub not in content.get("allowed_npubs", []):
        logger.debug("check_acl_access: caller %s not in allowed_npubs", caller_npub)
        return False

    # 4. Check expiry
    expires_at = content.get("expires_at")
    if expires_at is not None:
        if time.time() > expires_at:
            logger.debug("check_acl_access: ACL expired at %s", expires_at)
            return False

    # 5. Verify caller identity proof if provided
    if caller_identity_proof is not None:
        if not verify_identity_proof(caller_identity_proof, caller_npub, tool_name):
            logger.debug("check_acl_access: caller identity proof invalid")
            return False

    return True
