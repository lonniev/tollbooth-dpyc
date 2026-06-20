"""Nostr kind-0 profile read / publish over relays (NIP-01).

Patron profiles are PUBLIC, self-sovereign kind-0 metadata events signed by
the patron's OWN key. The wheel never holds a patron nsec — bad opsec:

  - READ: fetch any npub's latest kind-0 from relays (it's public; no proof).
  - WRITE: only RELAY a client-SIGNED kind-0 after verifying its signature
    matches the claimed npub. The signature IS the authorization — no proof
    token, no key custody. A frontend signs with the patron's session key or a
    NIP-07 extension and hands the signed event here.

Mirrors the self-contained raw-websocket approach in ``bootstrap_relay.py``.
``picture``/``banner`` are URLs — image bytes live off-relay.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

# Broad public relay set for profile discovery/publish. Patron profiles live on
# public relays, not necessarily the operator's, so default wide.
PROFILE_RELAYS = [
    "wss://relay.primal.net",
    "wss://nos.lol",
    "wss://relay.damus.io",
    "wss://relay.nostr.band",
]

_KIND_METADATA = 0
_TIMEOUT = 10
# Recognized kind-0 fields (NIP-01 + common extensions). Others are dropped.
_PROFILE_FIELDS = {
    "name", "display_name", "about", "picture", "banner", "nip05", "website", "lud16",
}


def _npub_to_hex(npub: str) -> str:
    from pynostr.key import PublicKey  # type: ignore[import-untyped]
    return PublicKey.from_npub(npub).hex()


def fetch_profile(npub: str, relays: list[str] | None = None) -> dict[str, str] | None:
    """Return the latest kind-0 metadata for ``npub`` (newest across relays).

    Returns the recognized profile fields as a dict, or None if no profile is
    found / npub is malformed / relays unreachable.
    """
    relay_urls = relays or PROFILE_RELAYS
    try:
        hex_pk = _npub_to_hex(npub)
    except Exception:
        return None

    import websocket  # type: ignore[import-untyped]

    best: dict | None = None
    best_ts = -1
    sub_filter = {"kinds": [_KIND_METADATA], "authors": [hex_pk], "limit": 1}

    for relay_url in relay_urls:
        try:
            ws = websocket.create_connection(relay_url, timeout=_TIMEOUT)
            sub = f"prof-{int(time.time())}"
            try:
                ws.send(json.dumps(["REQ", sub, sub_filter]))
                deadline = time.time() + _TIMEOUT
                while time.time() < deadline:
                    msg = json.loads(ws.recv())
                    if msg[0] == "EOSE":
                        break
                    if msg[0] == "EVENT" and len(msg) >= 3:
                        ev = msg[2]
                        ts = int(ev.get("created_at", 0) or 0)
                        if ev.get("kind") == _KIND_METADATA and ts > best_ts:
                            try:
                                content = json.loads(ev.get("content", "{}"))
                            except (json.JSONDecodeError, TypeError):
                                continue
                            if isinstance(content, dict):
                                best, best_ts = content, ts
                ws.send(json.dumps(["CLOSE", sub]))
            finally:
                ws.close()
        except Exception as exc:
            logger.debug("Profile fetch from %s failed: %s", relay_url, exc)

    if best is None:
        return None
    return {k: v for k, v in best.items() if k in _PROFILE_FIELDS and isinstance(v, str)}


def publish_profile_event(
    signed_event: dict | str, npub: str, relays: list[str] | None = None,
) -> dict:
    """Relay a CLIENT-SIGNED kind-0 event after verifying it belongs to ``npub``.

    The wheel does NOT sign — it verifies (kind 0, pubkey == npub, valid
    signature) and fans the event out to relays. Returns
    ``{success, ok, total, errors}``.
    """
    if isinstance(signed_event, str):
        try:
            signed_event = json.loads(signed_event)
        except (json.JSONDecodeError, TypeError):
            return {"success": False, "error": "signed_event is not valid JSON."}
    if not isinstance(signed_event, dict):
        return {"success": False, "error": "signed_event must be a JSON object."}

    if int(signed_event.get("kind", -1)) != _KIND_METADATA:
        return {"success": False, "error": "Event must be kind 0 (profile metadata)."}

    try:
        expected = _npub_to_hex(npub)
    except Exception:
        return {"success": False, "error": f"Invalid npub: {npub!r}"}
    if signed_event.get("pubkey") != expected:
        return {
            "success": False,
            "error": "Event pubkey does not match the claimed npub — refusing to relay.",
        }

    # Verify the Schnorr signature — the event must be genuinely signed by npub.
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
        ev = Event.from_dict(signed_event)
        if not ev.verify():
            return {"success": False, "error": "Event signature is invalid."}
    except Exception as exc:
        return {"success": False, "error": f"Could not verify event: {exc}"}

    message = json.dumps(["EVENT", signed_event])
    import websocket  # type: ignore[import-untyped]

    relay_urls = relays or PROFILE_RELAYS
    ok = 0
    errors: list[str] = []
    for relay_url in relay_urls:
        try:
            ws = websocket.create_connection(relay_url, timeout=_TIMEOUT)
            try:
                ws.send(message)
                raw = ws.recv()
                try:
                    ack = json.loads(raw)
                    if isinstance(ack, list) and len(ack) >= 3 and ack[0] == "OK" and ack[2] is True:
                        ok += 1
                    else:
                        errors.append(f"{relay_url}: {str(raw)[:120]}")
                except (json.JSONDecodeError, IndexError):
                    errors.append(f"{relay_url}: unparseable ack {str(raw)[:80]}")
            finally:
                ws.close()
        except Exception as exc:
            errors.append(f"{relay_url}: {exc}")

    return {"success": ok > 0, "ok": ok, "total": len(relay_urls), "errors": errors}
