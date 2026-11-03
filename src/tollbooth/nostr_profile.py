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

from tollbooth.relay_fanout import fan_out

logger = logging.getLogger(__name__)

_KIND_METADATA = 0
# Per-relay socket timeout. Relays are asked together (``fan_out``), so total
# wall-clock is bounded by the single slowest relay (~_TIMEOUT), not the sum
# across the set. Kept short: a profile read/publish should feel instant, and
# a slow or dead relay must never hold the whole call hostage.
_TIMEOUT = 5
# Recognized kind-0 fields (NIP-01 + common extensions). Others are dropped.
_PROFILE_FIELDS = {
    "name", "display_name", "about", "picture", "banner", "nip05", "website", "lud16",
}
# A relay's clock may run a little ahead; a profile dated further ahead than
# this is refused, or one relay could pin a forgery above every real update.
_MAX_FUTURE_SECONDS = 600


def _npub_to_hex(npub: str) -> str:
    from pynostr.key import PublicKey  # type: ignore[import-untyped]
    return PublicKey.from_npub(npub).hex()


def _authentic_profile(ev: object, hex_pk: str, now: float) -> tuple[int, dict] | None:
    """(created_at, content) of a kind-0 the npub itself signed, else None.

    A relay can return anything, so a profile counts only when it is kind 0,
    its author is the npub asked for, its Schnorr signature verifies over the
    recomputed id, and it is not dated in the future beyond clock skew.
    Otherwise one relay could serve a forged Lightning address as the newest.
    """
    if not isinstance(ev, dict) or ev.get("kind") != _KIND_METADATA or ev.get("pubkey") != hex_pk:
        return None
    try:
        ts = int(ev.get("created_at", 0) or 0)
        if ts > now + _MAX_FUTURE_SECONDS:
            return None
        from pynostr.event import Event  # type: ignore[import-untyped]

        if not Event.from_dict(ev).verify():
            return None
        content = json.loads(ev.get("content", "{}"))
    except Exception:  # noqa: BLE001 — malformed is not authentic
        return None
    return (ts, content) if isinstance(content, dict) else None


def _fetch_one(relay_url: str, sub_filter: dict) -> tuple[int, dict] | None:
    """Fetch a single relay's latest kind-0 for the filter. (ts, content) or None.

    Runs in its own thread; raises nothing — failures return None so a dead
    relay never breaks the fan-out.
    """
    import websocket  # type: ignore[import-untyped]

    best: dict | None = None
    best_ts = -1
    hex_pk = (sub_filter.get("authors") or [""])[0]
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
                    found = _authentic_profile(msg[2], hex_pk, time.time())
                    if found is not None and found[0] > best_ts:
                        best_ts, best = found
            ws.send(json.dumps(["CLOSE", sub]))
        finally:
            ws.close()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Profile fetch from %s failed: %s", relay_url, exc)
        return None
    return (best_ts, best) if best is not None else None


def fetch_profile(npub: str, relays: list[str] | None = None) -> dict[str, str] | None:
    """Return the latest kind-0 metadata for ``npub`` (newest across relays).

    Asks all relays together and keeps the newest result; wall-clock is
    bounded by the slowest single relay (~_TIMEOUT), not their sum. Returns the
    recognized profile fields as a dict, or None if no profile is found / npub
    is malformed / relays unreachable.
    """
    from tollbooth.relay_registry import get_relays
    relay_urls = relays or get_relays()
    try:
        hex_pk = _npub_to_hex(npub)
    except Exception:  # noqa: BLE001
        return None

    sub_filter = {"kinds": [_KIND_METADATA], "authors": [hex_pk], "limit": 1}

    best: dict | None = None
    best_ts = -1
    for o in fan_out(relay_urls, lambda url: _fetch_one(url, sub_filter), deadline=_TIMEOUT + 2):
        if o.ok and o.value is not None and o.value[0] > best_ts:
            best_ts, best = o.value

    if best is None:
        return None
    return {k: v for k, v in best.items() if k in _PROFILE_FIELDS and isinstance(v, str)}


def publish_event(signed_event: dict | str, relays: list[str] | None = None) -> dict:
    """Relay an ALREADY-SIGNED event and report per-relay results.

    Kind-agnostic transport. Does NOT verify signer identity — the caller owns
    authorship policy, because legitimate callers exist where the signer is
    deliberately NOT the claimed subject (e.g. an ephemeral scribe key
    annotating on a proven patron's behalf). Fans the event out to all relays
    together (wall-clock ~ the single slowest relay) and
    parses each ack strictly per NIP-20 — only ``["OK", <id>, true, …]`` counts
    as accepted.

    Returns ``{success, event_id, accepted, attempted, relays}`` where
    ``relays`` is a list of ``{relay, accepted, error}`` — one entry per relay.
    On a malformed ``signed_event`` returns ``{success: False, error}`` instead.
    """
    if isinstance(signed_event, str):
        try:
            signed_event = json.loads(signed_event)
        except (json.JSONDecodeError, TypeError):
            return {"success": False, "error": "signed_event is not valid JSON."}
    if not isinstance(signed_event, dict):
        return {"success": False, "error": "signed_event must be a JSON object."}

    message = json.dumps(["EVENT", signed_event])
    from tollbooth.relay_registry import get_relays
    relay_urls = relays or get_relays()

    results: list[dict] = []
    accepted = 0
    for o in fan_out(relay_urls, lambda url: _publish_one(url, message), deadline=_TIMEOUT + 2):
        ok, err = o.value if o.ok else (False, f"{o.relay}: {o.error}")
        accepted += ok
        results.append({"relay": o.relay, "accepted": ok, "error": err})

    return {
        "success": accepted > 0,
        "event_id": signed_event.get("id"),
        "accepted": accepted,
        "attempted": len(relay_urls),
        "relays": results,
    }


def publish_profile_event(
    signed_event: dict | str, npub: str, relays: list[str] | None = None,
) -> dict:
    """Relay a CLIENT-SIGNED kind-0 event after verifying it belongs to ``npub``.

    The wheel does NOT sign — it verifies (kind 0, pubkey == npub, valid
    signature) and then delegates the actual relay fan-out to the kind-agnostic
    ``publish_event`` transport. Authorship policy lives HERE; transport lives
    there. Returns ``{success, ok, total, errors}``.
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
    except Exception:  # noqa: BLE001
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
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Could not verify event: {exc}"}

    result = publish_event(signed_event, relays)
    errors = [r["error"] for r in result["relays"] if r["error"]]
    return {
        "success": result["success"],
        "ok": result["accepted"],
        "total": result["attempted"],
        "errors": errors,
    }


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
    except Exception as exc:  # noqa: BLE001
        return (False, f"{relay_url}: {exc}")
