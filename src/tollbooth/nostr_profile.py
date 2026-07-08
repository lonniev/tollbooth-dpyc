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
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_KIND_METADATA = 0
# Per-relay socket timeout. Relays are queried in PARALLEL (one thread each),
# so total wall-clock is bounded by the single slowest relay (~_TIMEOUT), not
# the sum across the set. Kept short: a profile read/publish should feel
# instant, and a slow or dead relay must never hold the whole call hostage.
_TIMEOUT = 5
# Recognized kind-0 fields (NIP-01 + common extensions). Others are dropped.
_PROFILE_FIELDS = {
    "name", "display_name", "about", "picture", "banner", "nip05", "website", "lud16",
}


def _npub_to_hex(npub: str) -> str:
    from pynostr.key import PublicKey  # type: ignore[import-untyped]
    return PublicKey.from_npub(npub).hex()


def _fetch_one(relay_url: str, sub_filter: dict) -> tuple[int, dict] | None:
    """Fetch a single relay's latest kind-0 for the filter. (ts, content) or None.

    Runs in its own thread; raises nothing — failures return None so a dead
    relay never breaks the fan-out.
    """
    import websocket  # type: ignore[import-untyped]

    best: dict | None = None
    best_ts = -1
    try:
        ws = websocket.create_connection(relay_url, timeout=_TIMEOUT)
        sub = f"prof-{int(time.time() * 1000)}"
        try:
            ws.settimeout(_TIMEOUT)
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
        return None
    return (best_ts, best) if best is not None else None


def fetch_profile(npub: str, relays: list[str] | None = None) -> dict[str, str] | None:
    """Return the latest kind-0 metadata for ``npub`` (newest across relays).

    Queries all relays in parallel and keeps the newest result; wall-clock is
    bounded by the slowest single relay (~_TIMEOUT), not their sum. Returns the
    recognized profile fields as a dict, or None if no profile is found / npub
    is malformed / relays unreachable.
    """
    from tollbooth.relay_registry import get_relays
    relay_urls = relays or get_relays()
    try:
        hex_pk = _npub_to_hex(npub)
    except Exception:
        return None

    sub_filter = {"kinds": [_KIND_METADATA], "authors": [hex_pk], "limit": 1}

    best: dict | None = None
    best_ts = -1
    with ThreadPoolExecutor(max_workers=len(relay_urls)) as pool:
        futures = {pool.submit(_fetch_one, url, sub_filter): url for url in relay_urls}
        for future in as_completed(futures, timeout=_TIMEOUT + 2):
            try:
                result = future.result()
            except Exception:
                continue
            if result is not None and result[0] > best_ts:
                best_ts, best = result[0], result[1]

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
    from tollbooth.relay_registry import get_relays
    relay_urls = relays or get_relays()

    ok = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=len(relay_urls)) as pool:
        futures = {pool.submit(_publish_one, url, message): url for url in relay_urls}
        for future in as_completed(futures, timeout=_TIMEOUT + 2):
            url = futures[future]
            try:
                accepted, err = future.result()
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                continue
            if accepted:
                ok += 1
            elif err:
                errors.append(err)

    return {"success": ok > 0, "ok": ok, "total": len(relay_urls), "errors": errors}


def _publish_one(relay_url: str, message: str) -> tuple[bool, str | None]:
    """Relay one EVENT message to a single relay. (accepted, error) — never raises."""
    import websocket  # type: ignore[import-untyped]

    try:
        ws = websocket.create_connection(relay_url, timeout=_TIMEOUT)
        try:
            ws.settimeout(_TIMEOUT)
            ws.send(message)
            raw = ws.recv()
            try:
                ack = json.loads(raw)
                if isinstance(ack, list) and len(ack) >= 3 and ack[0] == "OK" and ack[2] is True:
                    return (True, None)
                return (False, f"{relay_url}: {str(raw)[:120]}")
            except (json.JSONDecodeError, IndexError):
                return (False, f"{relay_url}: unparseable ack {str(raw)[:80]}")
        finally:
            ws.close()
    except Exception as exc:
        return (False, f"{relay_url}: {exc}")
