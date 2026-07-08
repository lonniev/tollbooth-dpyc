"""Nostr Relay Health Diagnostics — courier_health and courier_ping.

Provides two diagnostic functions for troubleshooting relay connectivity
and DM delivery issues in the Secure Courier Service.

``courier_health()`` performs a full diagnostic:
  - Per-relay WebSocket connectivity test
  - NIP-11 relay information document parsing
  - NIP capability detection (NIP-04, NIP-17, NIP-44)
  - Self-DM round-trip test (publish kind:4 to self, verify via subscription)
  - Round-trip latency measurement
  - Subscription filter test for kind:4 and kind:1059

``courier_ping(npub)`` performs a lightweight outbound DM test:
  - Sends a test NIP-04 DM to the specified npub
  - Returns event_id, relay confirmations, and timestamp

Both functions are free (0 api_sats) diagnostic utilities.

Dependencies are optional — install with ``pip install tollbooth-dpyc[nostr]``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation
try:
    from pynostr.event import Event  # type: ignore[import-untyped]
    from pynostr.key import PrivateKey, PublicKey  # type: ignore[import-untyped]

    _HAS_PYNOSTR = True
except ImportError:
    _HAS_PYNOSTR = False

try:
    from websocket import create_connection  # type: ignore[import-untyped]

    _HAS_WEBSOCKET = True
except ImportError:
    _HAS_WEBSOCKET = False

try:
    from tollbooth.nip04 import encrypt as _nip04_encrypt

    _HAS_NIP04 = True
except ImportError:
    _HAS_NIP04 = False


# Nostr event kinds (same as nostr_credentials.py)
_KIND_ENCRYPTED_DM = 4
_KIND_GIFT_WRAP = 1059

# Timeouts
_WS_CONNECT_TIMEOUT = 10  # seconds
_WS_RECV_TIMEOUT = 10  # seconds
_NIP11_TIMEOUT = 10  # seconds
_SELF_DM_SUBSCRIBE_TIMEOUT = 15  # seconds


class DiagnosticError(Exception):
    """Base error for diagnostic operations."""


def _npub_to_hex(npub: str) -> str:
    """Convert npub bech32 to 32-byte hex pubkey."""
    if not _HAS_PYNOSTR:
        raise DiagnosticError("pynostr required for npub conversion")
    return PublicKey.from_npub(npub).hex()


async def _fetch_nip11(relay_url: str) -> dict[str, Any]:
    """Fetch and parse a relay's NIP-11 information document.

    Converts the WebSocket URL to HTTP(S) and sends a GET request with
    ``Accept: application/nostr+json``.

    Returns:
        Parsed NIP-11 JSON dict, or an error dict with ``"error"`` key.
    """
    # Convert wss:// to https:// and ws:// to http://
    http_url = relay_url.replace("wss://", "https://").replace("ws://", "http://")

    try:
        async with httpx.AsyncClient(timeout=_NIP11_TIMEOUT) as client:
            resp = await client.get(
                http_url,
                headers={"Accept": "application/nostr+json"},
            )
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code}"}
            return resp.json()
    except httpx.TimeoutException:
        return {"error": "timeout"}
    except httpx.ConnectError as exc:
        return {"error": f"connect_error: {exc}"}
    except json.JSONDecodeError:
        return {"error": "invalid_json"}
    except Exception as exc:
        return {"error": str(exc)}


def _extract_nip_support(nip11: dict[str, Any]) -> dict[str, bool]:
    """Extract NIP capability flags from a NIP-11 document.

    Checks for NIP-04 (legacy encrypted DMs), NIP-17 (private DMs via
    gift wrap), and NIP-44 (versioned encryption) support.

    Returns:
        Dict with ``nip04``, ``nip17``, ``nip44`` boolean flags.
    """
    supported = nip11.get("supported_nips", [])
    # supported_nips may be a list of ints or strings
    nip_set: set[int] = set()
    for item in supported:
        try:
            nip_set.add(int(item))
        except (ValueError, TypeError):
            pass

    return {
        "nip04": 4 in nip_set,
        "nip17": 17 in nip_set,
        "nip44": 44 in nip_set,
    }


def test_ws_connectivity(
    relay_url: str,
    *,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Test WebSocket connectivity to a single relay.

    Args:
        relay_url: WebSocket URL of the relay (e.g. ``wss://nostr.wine``).
        timeout: Connection timeout in seconds. Defaults to module-level
            ``_WS_CONNECT_TIMEOUT`` (10s).

    Returns:
        Dict with ``relay`` (str), ``connected`` (bool),
        ``latency_ms`` (float or None), and ``error`` (str or None).
    """
    if not _HAS_WEBSOCKET:
        return {
            "relay": relay_url,
            "connected": False,
            "latency_ms": None,
            "error": "websocket-client not installed",
        }

    effective_timeout = timeout if timeout is not None else _WS_CONNECT_TIMEOUT
    t0 = time.monotonic()
    try:
        ws = create_connection(relay_url, timeout=effective_timeout)
        latency = (time.monotonic() - t0) * 1000
        try:
            ws.close()
        except Exception:
            logger.debug("best-effort websocket close failed", exc_info=True)
        return {
            "relay": relay_url,
            "connected": True,
            "latency_ms": round(latency, 1),
            "error": None,
        }
    except Exception as exc:
        latency = (time.monotonic() - t0) * 1000
        return {
            "relay": relay_url,
            "connected": False,
            "latency_ms": round(latency, 1),
            "error": str(exc),
        }


# Backward-compat alias (used internally in this module)
_test_ws_connectivity = test_ws_connectivity


def probe_relay_liveness(
    relay_urls: list[str],
    *,
    timeout: int = 5,
) -> list[dict[str, Any]]:
    """Probe WebSocket connectivity for a list of relays.

    Returns a list of dicts: ``{relay, connected, latency_ms, error}``.
    Sorted: connected relays first (by latency), then disconnected.

    Args:
        relay_urls: Relay WebSocket URLs to probe.
        timeout: Per-relay connection timeout in seconds (default 5).
    """
    results = [test_ws_connectivity(url, timeout=timeout) for url in relay_urls]
    # Sort: connected first (by latency), then disconnected
    results.sort(key=lambda r: (not r["connected"], r["latency_ms"] or float("inf")))
    return results


# ── Relay resolution ──────────────────────────────────────────────────


def resolve_relays(*, timeout: int = 5) -> list[str]:
    """Resolve a live, ordered set of Nostr relay URLs.

    The relay set is governed solely by the DPYC community registry
    (``dpyc-community/relays.json``, via ``relay_registry.get_relays``). This
    function sources that curated, primary-first set, probes each relay for
    liveness, and returns the live relays **in registry order** (so the
    ``primary`` relay stays first for the courier's rendezvous). If none
    respond, it returns the full registry set unprobed, hoping for recovery.

    Args:
        timeout: Per-relay probe timeout in seconds (default 5).

    Returns:
        A non-empty list of relay WebSocket URLs.

    Raises:
        RelayRegistryError: on a cold cache with an unreachable registry.
    """
    from tollbooth.relay_registry import get_relays

    relays = get_relays()

    results = probe_relay_liveness(relays, timeout=timeout)
    live_set = {r["relay"] for r in results if r["connected"]}
    # Preserve registry (primary-first) order among the live relays.
    live = [url for url in relays if url in live_set]

    if live:
        logger.info("Relay probe: %d/%d registry relays live", len(live), len(relays))
        return live

    logger.warning("No registry relays responded — using full set, hoping for recovery")
    return relays


def _test_subscription_filter(
    relay_url: str,
    pubkey_hex: str,
    kind: int,
) -> dict[str, Any]:
    """Test whether a relay accepts a subscription filter for a given kind.

    Sends a REQ with the specified filter and checks for EOSE (success)
    or CLOSED/NOTICE (rejection).

    Returns:
        Dict with ``accepted`` (bool), ``event_count`` (int), and
        ``error`` (str or None).
    """
    if not _HAS_WEBSOCKET:
        return {"accepted": False, "event_count": 0, "error": "websocket-client not installed"}

    sub_id = f"diag-{kind}-{uuid.uuid4().hex[:8]}"
    filt: dict[str, Any] = {
        "kinds": [kind],
        "#p": [pubkey_hex],
        "since": int(time.time()) - 300,
        "limit": 5,
    }

    try:
        ws = create_connection(relay_url, timeout=_WS_CONNECT_TIMEOUT)
    except Exception as exc:
        return {"accepted": False, "event_count": 0, "error": str(exc)}

    try:
        ws.send(json.dumps(["REQ", sub_id, filt]))
        ws.settimeout(_WS_RECV_TIMEOUT)

        event_count = 0
        accepted = False

        while True:
            try:
                raw = ws.recv()
            except Exception:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(msg, list) or len(msg) < 2:
                continue

            msg_type = msg[0]
            if msg_type == "EVENT":
                event_count += 1
            elif msg_type == "EOSE":
                accepted = True
                break
            elif msg_type in ("CLOSED", "NOTICE"):
                break

        # Clean up subscription
        try:
            ws.send(json.dumps(["CLOSE", sub_id]))
        except Exception:
            logger.debug("best-effort CLOSE frame send failed", exc_info=True)

        return {
            "accepted": accepted,
            "event_count": event_count,
            "error": None,
        }
    except Exception as exc:
        return {"accepted": False, "event_count": 0, "error": str(exc)}
    finally:
        try:
            ws.close()
        except Exception:
            logger.debug("best-effort websocket close failed", exc_info=True)


def _self_dm_roundtrip(
    relay_url: str,
    privkey_hex: str,
    pubkey_hex: str,
) -> dict[str, Any]:
    """Publish a kind:4 DM to self and verify receipt via subscription.

    Returns:
        Dict with ``success`` (bool), ``roundtrip_ms`` (float or None),
        ``event_id`` (str or None), and ``error`` (str or None).
    """
    if not _HAS_PYNOSTR:
        return {"success": False, "roundtrip_ms": None, "event_id": None, "error": "pynostr not installed"}
    if not _HAS_WEBSOCKET:
        return {"success": False, "roundtrip_ms": None, "event_id": None, "error": "websocket-client not installed"}
    if not _HAS_NIP04:
        return {"success": False, "roundtrip_ms": None, "event_id": None, "error": "nip04 not available"}

    nonce = uuid.uuid4().hex
    ping_text = f"courier-health-ping-{nonce}"

    # Encrypt to self
    encrypted = _nip04_encrypt(ping_text, privkey_hex, pubkey_hex)

    # Build kind:4 event
    event = Event(
        kind=_KIND_ENCRYPTED_DM,
        content=encrypted,
        tags=[["p", pubkey_hex]],
        pubkey=pubkey_hex,
        created_at=int(time.time()),
    )
    event.sign(privkey_hex)
    event_id = event.id
    publish_msg = event.to_message()

    t0 = time.monotonic()

    # Publish
    try:
        ws = create_connection(relay_url, timeout=_WS_CONNECT_TIMEOUT)
    except Exception as exc:
        return {"success": False, "roundtrip_ms": None, "event_id": None, "error": f"connect: {exc}"}

    try:
        ws.send(publish_msg)
        ws.settimeout(_WS_RECV_TIMEOUT)

        # Read the OK response
        try:
            raw = ws.recv()
            ok_msg = json.loads(raw)
            if isinstance(ok_msg, list) and ok_msg[0] == "OK" and not ok_msg[2]:
                return {
                    "success": False,
                    "roundtrip_ms": None,
                    "event_id": event_id,
                    "error": f"relay rejected event: {ok_msg[3] if len(ok_msg) > 3 else 'unknown'}",
                }
        except Exception:
            logger.debug("error awaiting relay OK confirmation", exc_info=True)
    finally:
        try:
            ws.close()
        except Exception:
            logger.debug("best-effort websocket close failed", exc_info=True)

    # Subscribe to verify receipt
    sub_id = f"diag-selfping-{uuid.uuid4().hex[:8]}"
    filt: dict[str, Any] = {
        "kinds": [_KIND_ENCRYPTED_DM],
        "#p": [pubkey_hex],
        "ids": [event_id],
        "limit": 1,
    }

    try:
        ws2 = create_connection(relay_url, timeout=_WS_CONNECT_TIMEOUT)
    except Exception as exc:
        return {
            "success": False,
            "roundtrip_ms": None,
            "event_id": event_id,
            "error": f"subscribe connect: {exc}",
        }

    try:
        ws2.send(json.dumps(["REQ", sub_id, filt]))
        ws2.settimeout(_SELF_DM_SUBSCRIBE_TIMEOUT)

        found = False
        while True:
            try:
                raw = ws2.recv()
            except Exception:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(msg, list) or len(msg) < 2:
                continue

            if msg[0] == "EVENT" and len(msg) >= 3:
                evt = msg[2]
                if isinstance(evt, dict) and evt.get("id") == event_id:
                    found = True
                    break
            elif msg[0] == "EOSE":
                break
            elif msg[0] in ("CLOSED", "NOTICE"):
                break

        roundtrip = (time.monotonic() - t0) * 1000

        try:
            ws2.send(json.dumps(["CLOSE", sub_id]))
        except Exception:
            logger.debug("best-effort CLOSE frame send failed", exc_info=True)

        return {
            "success": found,
            "roundtrip_ms": round(roundtrip, 1) if found else None,
            "event_id": event_id,
            "error": None if found else "event not found after publish",
        }
    except Exception as exc:
        return {
            "success": False,
            "roundtrip_ms": None,
            "event_id": event_id,
            "error": str(exc),
        }
    finally:
        try:
            ws2.close()
        except Exception:
            logger.debug("best-effort websocket close failed", exc_info=True)


async def courier_health(
    relays: list[str],
    operator_nsec: str,
) -> dict[str, Any]:
    """Full relay connectivity and capability diagnostic.

    For each configured relay, reports:
    - WebSocket connectivity (connected / unreachable / timeout)
    - NIP-11 information document (name, description, supported NIPs)
    - NIP capability detection (NIP-04, NIP-17, NIP-44)
    - Subscription filter acceptance for kind:4 and kind:1059
    - Self-DM round-trip test with latency measurement

    Args:
        relays: List of relay WebSocket URLs.
        operator_nsec: Operator's Nostr nsec (bech32).

    Returns:
        Dict with per-relay results under ``"relays"`` key, plus a
        top-level ``"summary"`` with counts and overall health.
    """
    if not _HAS_PYNOSTR:
        return {
            "success": False,
            "error": "pynostr not installed. Install with: pip install tollbooth-dpyc[nostr]",
        }

    # Derive keys
    try:
        pk = PrivateKey.from_nsec(operator_nsec)
        privkey_hex = pk.hex()
        pubkey_hex = pk.public_key.hex()
        npub = pk.public_key.bech32()
    except Exception as exc:
        return {
            "success": False,
            "error": f"Invalid operator nsec: {exc}",
        }

    relays = [r.strip() for r in relays if r.strip()]
    if not relays:
        return {
            "success": False,
            "error": "No relays configured.",
        }

    relay_results: list[dict[str, Any]] = []
    connected_count = 0
    nip04_capable_count = 0
    nip17_capable_count = 0
    roundtrip_ok_count = 0

    for relay_url in relays:
        result: dict[str, Any] = {"relay": relay_url}

        # 1. WebSocket connectivity
        ws_result = _test_ws_connectivity(relay_url)
        result["connectivity"] = ws_result

        if not ws_result["connected"]:
            result["nip11"] = {"error": "skipped (not connected)"}
            result["nip_support"] = {"nip04": False, "nip17": False, "nip44": False}
            result["subscription_kind4"] = {"accepted": False, "event_count": 0, "error": "skipped"}
            result["subscription_kind1059"] = {"accepted": False, "event_count": 0, "error": "skipped"}
            result["self_dm_roundtrip"] = {"success": False, "roundtrip_ms": None, "event_id": None, "error": "skipped"}
            relay_results.append(result)
            continue

        connected_count += 1

        # 2. NIP-11 info document
        nip11 = await _fetch_nip11(relay_url)
        result["nip11"] = nip11

        # 3. NIP capability detection
        if "error" not in nip11:
            nip_support = _extract_nip_support(nip11)
        else:
            nip_support = {"nip04": False, "nip17": False, "nip44": False}
        result["nip_support"] = nip_support

        if nip_support["nip04"]:
            nip04_capable_count += 1
        if nip_support["nip17"]:
            nip17_capable_count += 1

        # 4. Subscription filter tests
        kind4_sub = _test_subscription_filter(relay_url, pubkey_hex, _KIND_ENCRYPTED_DM)
        result["subscription_kind4"] = kind4_sub

        kind1059_sub = _test_subscription_filter(relay_url, pubkey_hex, _KIND_GIFT_WRAP)
        result["subscription_kind1059"] = kind1059_sub

        # 5. Self-DM round-trip
        selfping = _self_dm_roundtrip(relay_url, privkey_hex, pubkey_hex)
        result["self_dm_roundtrip"] = selfping
        if selfping["success"]:
            roundtrip_ok_count += 1

        relay_results.append(result)

    return {
        "success": True,
        "operator_npub": npub,
        "relay_count": len(relays),
        "summary": {
            "connected": connected_count,
            "unreachable": len(relays) - connected_count,
            "nip04_capable": nip04_capable_count,
            "nip17_capable": nip17_capable_count,
            "self_dm_roundtrip_ok": roundtrip_ok_count,
        },
        "relays": relay_results,
    }


async def courier_ping(
    npub: str,
    relays: list[str],
    operator_nsec: str,
) -> dict[str, Any]:
    """Lightweight outbound DM test to a specified npub.

    Sends a plaintext test message as a NIP-04 encrypted DM to the
    target npub via all configured relays.  The patron checks their
    Nostr client to verify receipt.

    Args:
        npub: Target recipient's npub (bech32).
        relays: List of relay WebSocket URLs.
        operator_nsec: Operator's Nostr nsec (bech32).

    Returns:
        Dict with ``event_id``, per-relay confirmation results, and
        timestamp.
    """
    if not _HAS_PYNOSTR:
        return {
            "success": False,
            "error": "pynostr not installed. Install with: pip install tollbooth-dpyc[nostr]",
        }
    if not _HAS_WEBSOCKET:
        return {
            "success": False,
            "error": "websocket-client not installed. Install with: pip install tollbooth-dpyc[nostr]",
        }
    if not _HAS_NIP04:
        return {
            "success": False,
            "error": "NIP-04 module not available. Install cryptography.",
        }

    # Derive operator keys
    try:
        pk = PrivateKey.from_nsec(operator_nsec)
        privkey_hex = pk.hex()
        pubkey_hex = pk.public_key.hex()
        operator_npub = pk.public_key.bech32()
    except Exception as exc:
        return {
            "success": False,
            "error": f"Invalid operator nsec: {exc}",
        }

    # Resolve recipient
    try:
        recipient_hex = _npub_to_hex(npub)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Invalid recipient npub: {exc}",
        }

    relays = [r.strip() for r in relays if r.strip()]
    if not relays:
        return {
            "success": False,
            "error": "No relays configured.",
        }

    # Build ping message
    nonce = uuid.uuid4().hex[:12]
    timestamp = int(time.time())
    ping_text = (
        f"Courier ping from {operator_npub} at "
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(timestamp))}. "
        f"Nonce: {nonce}. "
        f"If you see this message, your Nostr client can receive DMs from "
        f"this Tollbooth operator."
    )

    # Encrypt
    encrypted = _nip04_encrypt(ping_text, privkey_hex, recipient_hex)

    # Build event
    event = Event(
        kind=_KIND_ENCRYPTED_DM,
        content=encrypted,
        tags=[["p", recipient_hex]],
        pubkey=pubkey_hex,
        created_at=timestamp,
    )
    event.sign(privkey_hex)
    event_id = event.id
    publish_msg = event.to_message()

    # Publish to each relay
    relay_confirmations: list[dict[str, Any]] = []
    confirmed_count = 0

    for relay_url in relays:
        try:
            ws = create_connection(relay_url, timeout=_WS_CONNECT_TIMEOUT)
        except Exception as exc:
            relay_confirmations.append({
                "relay": relay_url,
                "confirmed": False,
                "error": str(exc),
            })
            continue

        try:
            ws.send(publish_msg)
            ws.settimeout(_WS_RECV_TIMEOUT)

            try:
                raw = ws.recv()
                msg = json.loads(raw)
                if isinstance(msg, list) and msg[0] == "OK":
                    ok = msg[2] if len(msg) > 2 else False
                    reason = msg[3] if len(msg) > 3 else ""
                    relay_confirmations.append({
                        "relay": relay_url,
                        "confirmed": bool(ok),
                        "error": reason if not ok else None,
                    })
                    if ok:
                        confirmed_count += 1
                else:
                    relay_confirmations.append({
                        "relay": relay_url,
                        "confirmed": False,
                        "error": f"unexpected response: {msg[0] if isinstance(msg, list) else 'unknown'}",
                    })
            except Exception as exc:
                relay_confirmations.append({
                    "relay": relay_url,
                    "confirmed": False,
                    "error": f"recv: {exc}",
                })
        finally:
            try:
                ws.close()
            except Exception:
                logger.debug("best-effort websocket close failed", exc_info=True)

    return {
        "success": confirmed_count > 0,
        "event_id": event_id,
        "recipient_npub": npub,
        "operator_npub": operator_npub,
        "timestamp": timestamp,
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
        "nonce": nonce,
        "relays_attempted": len(relays),
        "relays_confirmed": confirmed_count,
        "relay_results": relay_confirmations,
        "message": (
            f"Ping sent to {npub} via {confirmed_count}/{len(relays)} relays. "
            f"Ask the recipient to check their Nostr client for a DM from {operator_npub}."
            if confirmed_count > 0
            else "Ping failed — no relay confirmed delivery. Check relay connectivity."
        ),
    }
