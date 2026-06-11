"""Npub-ownership proof tool bodies (audit M2.1f Phase 2).

Extracted verbatim from the ``request_npub_proof`` / ``receive_npub_proof`` /
``check_proof_status`` closures in ``register_standard_tools``. These are
orchestrators over ``NostrCredentialExchange`` internals (the deterministic,
poison-scoped drain loop) plus the proven-npub cache — §2-sensitive identity
code, so the move is behavior-preserving: ``resolve_npub`` became
``rt.resolve_npub`` and the ``_PROOF_SERVICE`` closure local became the module
constant below; nothing else changed. The drain loop is pinned by
``tests/test_proof_tools_characterization.py``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# The credential-template service name the proof challenge/response rides on.
# Channel state is keyed by this raw service (matching open_channel and
# _resolve_pinned_record), never a resolved credential template.
PROOF_SERVICE = "npub_ownership"


async def request_npub_proof_tool(
    rt: Any,
    patron_npub: str,
    *,
    service_name: str,
) -> dict[str, Any]:
    """Send an npub-ownership challenge DM via the Secure Courier."""
    err = rt.npub_validation_error(patron_npub, param="patron_npub")
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

    # Purge stale DMs for this patron before sending a fresh challenge. The
    # fetch and NIP-09 deletes are blocking websocket I/O, so run them off the
    # event loop (same reasoning as the courier drains in 0.44.3 / M1.2).
    def _purge_stale() -> int:
        from tollbooth.nostr_credentials import _npub_to_hex
        exchange = courier._exchange
        patron_hex = _npub_to_hex(patron_npub)
        exchange._fetch_dms_from_relays()
        stale = exchange._find_dm_candidates(patron_hex)
        for candidate in stale:
            exchange._pop_event(candidate.get("id", ""))
        return len(stale)

    try:
        n_purged = await asyncio.to_thread(_purge_stale)
        if n_purged:
            logger.info("Purged %d stale DM(s) for %s", n_purged, patron_npub[:20])
    except Exception:
        logger.debug("best-effort stale-DM purge failed", exc_info=True)

    try:
        _greeting = rt._npub_proof_greeting or (
            f"Hi — {service_name or 'this service'} needs to verify "
            "you own this npub. Reply with any text to confirm. "
            "Your signed Nostr DM is the proof."
        )
        result = await courier.open_channel(
            PROOF_SERVICE,
            greeting=_greeting,
            recipient_npub=patron_npub,
        )
        if not result.get("success"):
            return result
    except Exception as e:
        return {"success": False, "error": f"Failed to send proof request: {e}"}

    # Extract the poison phrase and rendezvous relay from the channel result.
    poison = result.get("poison", "")
    rendezvous_relay = result.get("rendezvous_relay", "")

    response: dict[str, Any] = {
        "success": True,
        "proof_token": poison,
        "message": (
            "Proof request sent via Secure Courier. "
            "Call receive_npub_proof to complete."
        ),
    }
    if rendezvous_relay:
        response["rendezvous_relay"] = rendezvous_relay
        response["message"] = (
            f"Proof request sent via Secure Courier on {rendezvous_relay}. "
            f"The patron MUST reply on that relay (it's embedded in the DM). "
            f"Call receive_npub_proof after the patron confirms they replied."
        )
    return response


async def receive_npub_proof_tool(
    rt: Any,
    patron_npub: str,
    poison: str,
) -> dict[str, Any]:
    """Drain the pinned relay, verify the proof reply, and cache proven status."""
    err = rt.npub_validation_error(patron_npub, param="patron_npub")
    if err is not None:
        return err
    from tollbooth.constants import ErrorCode as _EC
    if not poison:
        return {
            "success": False,
            "error_code": _EC.POISON_MISSING,
            "error": (
                "poison is required — pass the proof_token returned by "
                "request_npub_proof."
            ),
        }
    resolved = rt.resolve_npub(patron_npub)

    courier = await rt.courier()
    if courier is None:
        return {
            "success": False,
            "error_code": _EC.SECURE_COURIER_UNAVAILABLE,
            "error": "Secure Courier not configured.",
        }

    exchange = courier._exchange
    from tollbooth.nostr_credentials import (
        _npub_to_hex, _parse_delimited_credentials,
        _NACK_TOKEN, _MAX_NACKS_PER_DRAIN, _courier_resolve_error,
    )

    patron_hex = _npub_to_hex(resolved)
    # Channel state is keyed by the RAW proof service (matching open_channel
    # and _resolve_pinned_record), never the resolved credential template.
    poison_key = (resolved, PROOF_SERVICE)

    # Resolve + verify the pinned rendezvous relay for this exact
    # (patron, proof-service, poison). Handles cold-start vault rehydration
    # and returns a structured error — popping nothing — when unresolved.
    pinned, error_code = await exchange._resolve_pinned_record(
        resolved, PROOF_SERVICE, poison,
    )
    if error_code is not None:
        return {
            "success": False,
            "error_code": error_code,
            "popped_dms": 0,
            "error": _courier_resolve_error(error_code, "request_npub_proof"),
        }
    expected_phrase = poison

    # Single drain of ONLY the pinned relay — human-gated, no retry loop.
    # Off the event loop — blocking websocket I/O (0.44.3 / M1.2 reasoning).
    await asyncio.to_thread(exchange._fetch_dms_from_relays, [pinned])
    candidates = [
        c for c in exchange._find_dm_candidates(patron_hex)
        if c.get("_relay") in (pinned, None)
    ]

    if not candidates:
        return {
            "success": False,
            "error_code": _EC.COURIER_NOT_FOUND,
            "popped_dms": 0,
            "error": (
                f"No reply found on the pinned relay ({pinned}). "
                f"Confirm the patron replied there, then try again."
            ),
        }

    # Drain loop, stop-at-match. The poison phrase is the sole scoping
    # mechanism — a reply carrying the current one-time poison is by definition
    # the right reply, regardless of wall-clock timing (no timestamp gate, which
    # raced clock skew + human-paced replies and silently dropped valid proofs).
    # Mismatched DMs are NACK'd up to the cap. For self-DM proofs the reply is
    # encrypted to the ephemeral agent npub, so decrypt with the agent key
    # (restored by the resolver) and fall back to the operator nsec.
    matched_payload = None
    last_failure = None
    popped = 0
    nacks_sent = 0
    agent_key = exchange._ephemeral_agents.get(poison_key)
    decrypt_key = agent_key.hex() if agent_key else exchange._privkey_hex

    for candidate in candidates:
        event_id = candidate.get("id", "")

        nack_reason: str | None = None
        payload = None
        try:
            plaintext = exchange._decrypt_dm(
                candidate, patron_hex, decrypt_privkey_hex=decrypt_key,
            )
        except Exception:
            plaintext = None
            nack_reason = _NACK_TOKEN
            last_failure = "undecryptable DM"

        if plaintext:
            payload = _parse_delimited_credentials(plaintext)
            if payload is None:
                nack_reason = _NACK_TOKEN
                last_failure = "no @@@ fields"
            elif payload.get("poison", "") != expected_phrase:
                nack_reason = _NACK_TOKEN
                last_failure = "wrong token"
        elif nack_reason is None:
            nack_reason = _NACK_TOKEN
            last_failure = "empty DM"

        if nack_reason is None and payload is not None:
            matched_payload = payload
            exchange._pop_event(event_id)
            popped += 1
            break  # stop-at-match

        if nacks_sent < _MAX_NACKS_PER_DRAIN:
            exchange._pop_event(event_id, resolved, nack_reason, target_relay=pinned)
            nacks_sent += 1
        else:
            exchange._pop_event(event_id)
        popped += 1

    # Clean up poison state and rendezvous pin (one-time use)
    exchange._pending_poisons.pop(poison_key, None)
    exchange._pinned_relays.pop(poison_key, None)
    exchange._ephemeral_agents.pop(poison_key, None)

    # One summary DM to patron
    if matched_payload is not None:
        # Store to vault (for cold-start recovery of proven status)
        if exchange._credential_vault is not None:
            try:
                await exchange._vault_store(
                    PROOF_SERVICE, resolved, matched_payload,
                )
            except Exception:
                logger.debug(
                    "best-effort proof persistence to vault failed", exc_info=True,
                )

        # Operator callback (e.g., taxsort stores passphrase hash)
        if rt._on_npub_proven is not None:
            try:
                await rt._on_npub_proven(resolved, matched_payload)
            except Exception as exc:
                logger.warning("on_npub_proven callback failed: %s", exc)

        cache = await rt.proven_npub_cache()

        # Compute poison hash — the caller-supplied proof token for future paid
        # calls. The raw poison is returned to the caller but never stored.
        import hashlib as _hashlib
        poison_hash = _hashlib.sha256(
            expected_phrase.encode(),
        ).hexdigest()

        # Parse patron's chosen cache duration (default: 2h)
        raw_duration = (matched_payload or {}).get("cache_duration", "").strip()
        ttl_seconds: int | None = None
        if raw_duration:
            try:
                from tollbooth.proven_npub import parse_duration
                ttl_seconds = parse_duration(raw_duration)
            except ValueError:
                pass  # unparseable → use cache default

        from tollbooth.proven_npub import UNSET
        record = await cache.mark_proven(poison_hash, resolved, ttl_override=ttl_seconds if raw_duration else UNSET)

        ttl_display = int(record.expires_at - record.verified_at)
        hours = ttl_display / 3600
        if hours >= 1:
            duration_human = f"{hours:.0f} hour{'s' if hours != 1 else ''}"
        else:
            duration_human = f"{ttl_display // 60} minute{'s' if ttl_display >= 120 else ''}"

        from datetime import datetime, timezone
        expires_dt = datetime.fromtimestamp(record.expires_at, tz=timezone.utc)
        expires_str = expires_dt.strftime("%Y-%m-%d %H:%M UTC")

        npub_short = resolved[:16] + "..." if len(resolved) > 20 else resolved
        op_name = rt._service_name

        confirmation_msg = (
            f"Your ownership of {npub_short} is confirmed "
            f"for {op_name}. "
            f"Proof remains valid until {expires_str} "
            f"({duration_human} from now). "
            f"Cleaned {popped} DM(s) from relay."
        )

        # Send enriched confirmation DM to patron
        try:
            exchange.send_dm(resolved, confirmation_msg)
        except Exception:
            logger.debug("proof confirmation DM send failed", exc_info=True)

        return {
            "success": True,
            "proven_npub": resolved,
            "proof_token": expected_phrase,
            "popped_dms": popped,
            "expires_in_seconds": ttl_display,
            "expires_at": expires_str,
            "message": confirmation_msg,
        }
    else:
        # NACKs already went to each mismatched sender during the drain;
        # the returned error never reveals the expected phrase.
        summary = (
            f"Drained the pinned relay ({pinned}); cleaned {popped} DM(s) "
            f"but none carried the expected proof phrase. The queue is now "
            f"empty of the sought reply — confirm the patron replied on "
            f"{pinned}, or call request_npub_proof for a fresh exchange."
        )
        if last_failure:
            summary += f" (last issue: {last_failure})"
        return {
            "success": False,
            "error_code": _EC.COURIER_NOT_FOUND,
            "popped_dms": popped,
            "error": summary,
        }


async def check_proof_status_tool(
    rt: Any,
    patron_npub: str,
    proof_token: str,
) -> dict[str, Any]:
    """Report whether a cached proof_token is still valid (no side effects)."""
    err = rt.npub_validation_error(patron_npub, param="patron_npub")
    if err is not None:
        return err
    err = rt.proof_validation_error(proof_token, param="proof_token")
    if err is not None:
        return err
    resolved = rt.resolve_npub(patron_npub)

    import hashlib as _hashlib
    poison_hash = _hashlib.sha256(proof_token.encode()).hexdigest()
    cache = await rt.proven_npub_cache()
    info = await cache.proof_status(poison_hash, resolved)

    status = info["status"]
    if status == "valid":
        message = (
            "Proof is valid. Pass proof_token as the proof parameter "
            "on paid tool calls."
        )
    elif status == "expired":
        message = (
            "Proof has expired. Call request_npub_proof and "
            "receive_npub_proof to refresh."
        )
    else:
        message = (
            "No proof record found for this (patron_npub, proof_token). "
            "Call request_npub_proof and receive_npub_proof first."
        )
    return {
        "success": True,
        "status": status,
        "expires_in_seconds": info["expires_in_seconds"],
        "message": message,
    }
