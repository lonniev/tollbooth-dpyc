"""Npub-ownership proof tool bodies (audit M2.1f Phase 2).

Extracted verbatim from the ``request_npub_proof`` / ``receive_npub_proof`` /
``check_proof_status`` closures in ``register_standard_tools``. These are
orchestrators over ``NostrCredentialExchange`` internals (the deterministic,
dpop_token-scoped drain loop) plus the proven-npub cache — §2-sensitive identity
code, so the move is behavior-preserving: ``resolve_npub`` became
``rt.resolve_npub`` and the ``_PROOF_SERVICE`` closure local became the module
constant below; nothing else changed. The drain loop is pinned by
``tests/test_proof_tools_characterization.py``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC
from typing import Any

from tollbooth.constants import ErrorCode as _EC_MODULE
from tollbooth.nostr_credentials import CourierUnreachableError

_EC_UNREACHABLE = _EC_MODULE.COURIER_RELAY_UNREACHABLE

logger = logging.getLogger(__name__)

# The credential-template service name the proof challenge/response rides on.
# Channel state is keyed by this raw service (matching open_channel and
# _resolve_pinned_record), never a resolved credential template.
PROOF_SERVICE = "npub_ownership"


# Real-client-IP headers, most-trustworthy first. Different fronts use different
# names; we take the first that yields a *globally routable* address.
_IP_HEADERS = (
    "true-client-ip", "cf-connecting-ip", "fly-client-ip", "fastly-client-ip",
    "x-real-ip", "x-client-ip", "x-forwarded-for", "x-envoy-external-address",
)
_GEO_COUNTRY_HEADERS = (
    "cf-ipcountry", "x-vercel-ip-country", "x-country-code",
    "fastly-geo-country-code", "x-geo-country",
)
_GEO_CITY_HEADERS = ("cf-ipcity", "x-vercel-ip-city", "x-geo-city")


def _coarsen_ip(ip: str) -> str:
    """Blunt an IP for provenance display — the recipient judges *region*, not a
    pinpoint address. IPv4 → drop the last octet; IPv6 → keep the /48 prefix."""
    ip = ip.strip().strip("[]")
    if ":" in ip:  # IPv6
        return ":".join(ip.split(":")[:3]) + "::/48"
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3]) + ".0/24"
    return ip


def _first_public_ip(headers: dict[str, str], req: object) -> str:
    """The first globally-routable client IP the transport reveals, or "".

    A private / loopback / link-local address (127.*, 10.*, 192.168.*, etc.) is
    the *internal proxy*, NOT the client — on Horizon the app sees
    localhost — so it is deliberately discarded: a bogus "could be anywhere"
    address is worse than none.
    """
    import ipaddress

    def _global(cand: str) -> str:
        cand = cand.split(",")[0].strip().strip("[]")
        # strip a :port on IPv4
        if cand.count(":") == 1 and "." in cand:
            cand = cand.split(":")[0]
        try:
            return cand if ipaddress.ip_address(cand).is_global else ""
        except ValueError:
            return ""

    for h in _IP_HEADERS:
        got = _global(headers.get(h, ""))
        if got:
            return got
    try:
        host = getattr(getattr(req, "client", None), "host", "") or ""
        return _global(host)
    except Exception:  # noqa: BLE001
        return ""


def _assemble_origin(headers: dict[str, str], req: object) -> str | None:
    """Compose an origin string from what the transport reveals — but ONLY when
    an *observed* signal survives (a public client IP or an edge geo). A
    self-reported ``User-Agent`` alone is NOT enough: on a platform that hides
    the client IP (Horizon → the app sees localhost) that is all that
    remains, and asserting a "trust me" origin from a self-reported string is
    exactly what we must not do. Returns ``None`` in that case so the attestation
    omits the tag rather than showing a weak, misleading hint.
    """
    observed: list[str] = []

    # Geo — present only if a front injects it.
    country = next((headers[h].strip() for h in _GEO_COUNTRY_HEADERS
                    if headers.get(h, "").strip()), "")
    city = next((headers[h].strip() for h in _GEO_CITY_HEADERS
                 if headers.get(h, "").strip()), "")
    loc = ", ".join(x for x in (city, country) if x and x.upper() != "XX")
    if loc:
        observed.append(loc)

    # A *public* client IP, coarsened. A loopback/private address is discarded.
    ip = _first_public_ip(headers, req)
    if ip:
        observed.append(_coarsen_ip(ip))

    # Nothing the operator actually OBSERVED → omit. The self-reported client
    # agent is only added as context ALONGSIDE an observed signal, never alone.
    if not observed:
        return None
    ua = (headers.get("user-agent") or "").strip()
    if ua:
        observed.append(ua[:48])
    return " · ".join(observed)


def harvest_request_origin() -> str | None:
    """Best-effort, operator-OBSERVED provenance of the client that triggered
    this tool call. Pulls the transport headers server-side and delegates to
    :func:`_assemble_origin`, which returns ``None`` unless an observed signal
    (public IP / geo) survives — so a platform that hides the client IP simply
    yields no ``origin`` tag rather than a self-reported one.
    """
    try:
        from fastmcp.server.dependencies import get_http_headers, get_http_request
        headers = {k.lower(): v for k, v in (get_http_headers() or {}).items()}
        try:
            req: object = get_http_request()
        except Exception:  # noqa: BLE001
            req = None
    except Exception:  # noqa: BLE001
        return None
    return _assemble_origin(headers, req)


async def request_npub_proof_tool(
    rt: Any,
    patron_npub: str,
    *,
    service_name: str,
    reason: str | None = None,
    verify_at: str | None = None,
) -> dict[str, Any]:
    """Send an npub-ownership challenge DM via the Secure Courier.

    ``reason`` is an optional human-readable purpose the Operator states for
    the request ("I'm working on your request XYZ and need the Operator to do
    ABC"). It is signed into the provenance attestation (tamper-evident) and
    shown in the DM body, so the recipient sees *why* they are being asked —
    especially valuable for an unknown-signer request, where the stated
    purpose is what lets a human judge a stranger's ask.
    """
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
        # NOTE: `reason` is carried ONCE, in the signed attestation tag (below,
        # via open_channel → create_provenance_attestation). It is deliberately
        # NOT also spliced into the greeting or a provenance line — the payload
        # stays lean and single-sourced, and the recipient reads the reason from
        # the tamper-evident tag, not a duplicated free-text copy.
        # Stamp the request time into the preamble so the patron can see
        # when the challenge was raised. Kept to a single terse line — proof
        # DMs are succinct notifications, not documents.
        from datetime import datetime
        requested_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        _greeting = f"{_greeting}\nRequested: {requested_at}"
        # Best-effort operator-observed provenance of the triggering client —
        # signed into the attestation so the human can judge an unsolicited ask.
        origin = harvest_request_origin()
        result = await courier.open_channel(
            PROOF_SERVICE,
            greeting=_greeting,
            recipient_npub=patron_npub,
            reason=reason,
            origin=origin,
            verify_at=verify_at,
        )
        if not result.get("success"):
            return result
    except CourierUnreachableError as e:
        # Every candidate relay refused the challenge. That is a relay
        # situation, not a caller mistake, and it carries a code so an agent
        # can branch instead of parsing prose.
        return {
            "success": False,
            "error_code": _EC_UNREACHABLE,
            "error": (
                f"No relay would accept the proof challenge, so there is no "
                f"rendezvous for the patron to reply on. This is a relay "
                f"outage, not something the patron can fix. Try again shortly. "
                f"({e})"
            ),
        }
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": f"Failed to send proof request: {e}"}

    # Extract the dpop_token phrase and rendezvous relay from the channel result.
    dpop_token = result.get("dpop_token", "")
    rendezvous_relay = result.get("rendezvous_relay", "")

    response: dict[str, Any] = {
        "success": True,
        "dpop_token": dpop_token,
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
    dpop_token: str,
) -> dict[str, Any]:
    """Drain the pinned relay, verify the proof reply, and cache proven status."""
    err = rt.npub_validation_error(patron_npub, param="patron_npub")
    if err is not None:
        return err
    from tollbooth.constants import ErrorCode as _EC
    if not dpop_token:
        return {
            "success": False,
            "error_code": _EC.DPOP_TOKEN_MISSING,
            "error": (
                "dpop_token is required — pass the dpop_token returned by "
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
        _MAX_NACKS_PER_DRAIN,
        _NACK_TOKEN,
        _courier_resolve_error,
        _npub_to_hex,
        _parse_delimited_credentials,
    )

    patron_hex = _npub_to_hex(resolved)
    # Channel state is keyed by the RAW proof service (matching open_channel
    # and _resolve_pinned_record), never the resolved credential template.
    dpop_token_key = (resolved, PROOF_SERVICE)

    # Resolve + verify the pinned rendezvous relay for this exact
    # (patron, dpop_token-service, dpop_token). Handles cold-start vault rehydration
    # and returns a structured error — popping nothing — when unresolved.
    pinned, error_code = await exchange._resolve_pinned_record(
        resolved, PROOF_SERVICE, dpop_token,
    )
    if error_code is not None:
        return {
            "success": False,
            "error_code": error_code,
            "popped_dms": 0,
            "error": _courier_resolve_error(error_code, "request_npub_proof"),
        }
    expected_phrase = dpop_token

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

    # Drain loop, stop-at-match. The dpop_token phrase is the sole scoping
    # mechanism — a reply carrying the current one-time dpop_token is by definition
    # the right reply, regardless of wall-clock timing (no timestamp gate, which
    # raced clock skew + human-paced replies and silently dropped valid proofs).
    # Mismatched DMs are NACK'd up to the cap. For self-DM proofs the reply is
    # encrypted to the ephemeral agent npub, so decrypt with the agent key
    # (restored by the resolver) and fall back to the operator nsec.
    matched_payload = None
    last_failure = None
    popped = 0
    nacks_sent = 0
    agent_key = exchange._ephemeral_agents.get(dpop_token_key)
    decrypt_key = agent_key.hex() if agent_key else exchange._privkey_hex

    for candidate in candidates:
        event_id = candidate.get("id", "")

        nack_reason: str | None = None
        payload = None
        try:
            plaintext = exchange._decrypt_dm(
                candidate, patron_hex, decrypt_privkey_hex=decrypt_key,
            )
        except Exception:  # noqa: BLE001
            plaintext = None
            nack_reason = _NACK_TOKEN
            last_failure = "undecryptable DM"

        if plaintext:
            payload = _parse_delimited_credentials(plaintext)
            if payload is None:
                nack_reason = _NACK_TOKEN
                last_failure = "no @@@ fields"
            elif payload.get("dpop_token", "") != expected_phrase:
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

    # Clean up dpop_token state and rendezvous pin (one-time use)
    exchange._pending_dpop_tokens.pop(dpop_token_key, None)
    exchange._pinned_relays.pop(dpop_token_key, None)
    exchange._ephemeral_agents.pop(dpop_token_key, None)

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
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_npub_proven callback failed: %s", exc)

        cache = await rt.proven_npub_cache()

        # Compute dpop_token hash — the caller-supplied proof token for future paid
        # calls. The raw dpop_token is returned to the caller but never stored.
        import hashlib as _hashlib
        dpop_token_hash = _hashlib.sha256(
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
        record = await cache.mark_proven(dpop_token_hash, resolved, ttl_override=ttl_seconds if raw_duration else UNSET)

        ttl_display = int(record.expires_at - record.verified_at)
        hours = ttl_display / 3600
        if hours >= 1:
            duration_human = f"{hours:.0f} hour{'s' if hours != 1 else ''}"
        else:
            duration_human = f"{ttl_display // 60} minute{'s' if ttl_display >= 120 else ''}"

        from datetime import datetime
        expires_dt = datetime.fromtimestamp(record.expires_at, tz=UTC)
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
            "dpop_token": expected_phrase,
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
    dpop_token: str,
) -> dict[str, Any]:
    """Report whether a cached dpop_token is still valid (no side effects)."""
    err = rt.npub_validation_error(patron_npub, param="patron_npub")
    if err is not None:
        return err
    err = rt.proof_validation_error(dpop_token, param="dpop_token")
    if err is not None:
        return err
    resolved = rt.resolve_npub(patron_npub)

    import hashlib as _hashlib
    dpop_token_hash = _hashlib.sha256(dpop_token.encode()).hexdigest()
    cache = await rt.proven_npub_cache()
    info = await cache.proof_status(dpop_token_hash, resolved)

    status = info["status"]
    if status == "valid":
        message = (
            "Proof is valid. Pass dpop_token as the dpop_token parameter "
            "on paid tool calls."
        )
    elif status == "expired":
        message = (
            "Proof has expired. Call request_npub_proof and "
            "receive_npub_proof to refresh."
        )
    else:
        message = (
            "No proof record found for this (patron_npub, dpop_token). "
            "Call request_npub_proof and receive_npub_proof first."
        )
    return {
        "success": True,
        "status": status,
        "expires_in_seconds": info["expires_in_seconds"],
        "message": message,
    }
