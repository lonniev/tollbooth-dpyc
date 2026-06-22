"""Tests for nostr_profile — kind-0 read/publish helpers.

Covers the publish-side validation (the security-critical part: the wheel
relays a client-signed event only when kind/pubkey/signature check out). The
relay I/O itself (fetch_profile, the publish fan-out) is integration-only.
"""

import json
from unittest.mock import patch

from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth import nostr_profile
from tollbooth.nostr_profile import fetch_profile, publish_profile_event

# A real npub so _npub_to_hex succeeds; the per-relay I/O is mocked.
_NPUB = PrivateKey().public_key.bech32()


def _signed_kind(sk: PrivateKey, kind: int, content: dict) -> dict:
    ev = Event(
        kind=kind,
        content=json.dumps(content),
        tags=[],
        pubkey=sk.public_key.hex(),
        created_at=1_700_000_000,
    )
    ev.sign(sk.hex())
    return {
        "id": ev.id,
        "pubkey": ev.pubkey,
        "created_at": ev.created_at,
        "kind": ev.kind,
        "tags": ev.tags,
        "content": ev.content,
        "sig": ev.sig,
    }


def test_rejects_bad_json():
    r = publish_profile_event("{not json", "npub1whatever")
    assert r["success"] is False
    assert "JSON" in r["error"]


def test_rejects_wrong_kind():
    sk = PrivateKey()
    ev = _signed_kind(sk, 1, {"hello": "world"})  # kind 1, not 0
    r = publish_profile_event(ev, sk.public_key.bech32())
    assert r["success"] is False
    assert "kind 0" in r["error"]


def test_rejects_pubkey_mismatch():
    signer = PrivateKey()
    other = PrivateKey()
    ev = _signed_kind(signer, 0, {"name": "imposter"})
    # Claim a DIFFERENT npub than the one that signed it.
    r = publish_profile_event(ev, other.public_key.bech32())
    assert r["success"] is False
    assert "does not match" in r["error"]


def test_rejects_tampered_signature():
    sk = PrivateKey()
    ev = _signed_kind(sk, 0, {"name": "real"})
    ev["sig"] = "00" * 64  # tamper the signature; pubkey still matches
    r = publish_profile_event(ev, sk.public_key.bech32())
    assert r["success"] is False
    assert "signature is invalid" in r["error"].lower()


# --- Parallel fetch selection (per-relay I/O mocked) ---

def test_fetch_picks_newest_across_relays():
    """The newest created_at wins regardless of which relay returns it."""
    def fake(relay_url, sub_filter):
        return {
            "wss://relay.primal.net": (100, {"name": "old", "about": "stale"}),
            "wss://nos.lol": (300, {"name": "new", "about": "fresh"}),
            "wss://relay.damus.io": (200, {"name": "mid"}),
            "wss://relay.nostr.band": None,  # dead relay — tolerated
        }[relay_url]

    with patch.object(nostr_profile, "_fetch_one", side_effect=fake):
        profile = fetch_profile(_NPUB)
    assert profile == {"name": "new", "about": "fresh"}


def test_fetch_returns_none_when_all_relays_dead():
    with patch.object(nostr_profile, "_fetch_one", return_value=None):
        assert fetch_profile(_NPUB) is None


def test_fetch_drops_unrecognized_fields():
    def fake(relay_url, sub_filter):
        return (100, {"name": "ok", "evil": "x", "lud16": "a@b.com"}) \
            if relay_url == "wss://relay.primal.net" else None

    with patch.object(nostr_profile, "_fetch_one", side_effect=fake):
        profile = fetch_profile(_NPUB)
    assert profile == {"name": "ok", "lud16": "a@b.com"}  # "evil" dropped


def test_fetch_malformed_npub_is_none():
    assert fetch_profile("not-an-npub") is None
