"""Tests for nostr_profile — kind-0 read/publish helpers.

Covers the publish-side validation (the security-critical part: the wheel
relays a client-signed event only when kind/pubkey/signature check out). The
relay I/O itself (fetch_profile, the publish fan-out) is integration-only.
"""

import json
from unittest.mock import patch

import pytest
from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth import nostr_profile
from tollbooth.nostr_profile import fetch_profile, publish_event, publish_profile_event

# A real npub so _npub_to_hex succeeds; the per-relay I/O is mocked.
_NPUB = PrivateKey().public_key.bech32()

# The relay set the fan-out iterates. Stubbed so these unit tests never touch
# the network (the real relay set comes from the DPYC community registry).
_TEST_RELAYS = [
    "wss://relay.primal.net",
    "wss://nos.lol",
    "wss://relay.damus.io",
    "wss://relay.nostr.band",
]


@pytest.fixture(autouse=True)
def _stub_relay_registry():
    with patch("tollbooth.relay_registry.get_relays", return_value=list(_TEST_RELAYS)):
        yield


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


# --- publish_event: kind-agnostic transport (per-relay I/O mocked) ---

def test_publish_event_rejects_bad_json():
    r = publish_event("{not json")
    assert r["success"] is False
    assert "JSON" in r["error"]


def test_publish_event_rejects_non_object():
    r = publish_event(42)  # type: ignore[arg-type]
    assert r["success"] is False
    assert "JSON object" in r["error"]


def test_publish_event_is_kind_agnostic_and_signer_agnostic():
    """A kind-1 event whose signer is NOT the subject relays with no identity gate."""
    scribe = PrivateKey()  # ephemeral signer, deliberately not the "author"
    ev = _signed_kind(scribe, 1, {"note": "annotation on someone else's behalf"})

    with patch.object(nostr_profile, "_publish_one", return_value=(True, None)):
        r = publish_event(ev)

    assert r["success"] is True
    assert r["event_id"] == ev["id"]
    assert r["accepted"] == len(_TEST_RELAYS)
    assert r["attempted"] == len(_TEST_RELAYS)
    assert {row["relay"] for row in r["relays"]} == set(_TEST_RELAYS)
    assert all(row["accepted"] and row["error"] is None for row in r["relays"])


def test_publish_event_reports_per_relay_failures():
    ev = _signed_kind(PrivateKey(), 1, {"note": "mixed relay outcomes"})

    def fake(relay_url, message):
        if relay_url == "wss://nos.lol":
            return (True, None)
        return (False, f"{relay_url}: rejected")

    with patch.object(nostr_profile, "_publish_one", side_effect=fake):
        r = publish_event(ev)

    assert r["success"] is True  # at least one relay accepted
    assert r["accepted"] == 1
    assert r["attempted"] == len(_TEST_RELAYS)
    failed = [row for row in r["relays"] if not row["accepted"]]
    assert len(failed) == len(_TEST_RELAYS) - 1
    assert all("rejected" in row["error"] for row in failed)


def test_publish_event_all_relays_reject_is_unsuccessful():
    ev = _signed_kind(PrivateKey(), 1, {"note": "nobody wants it"})
    with patch.object(nostr_profile, "_publish_one", return_value=(False, "wss://x: no")):
        r = publish_event(ev)
    assert r["success"] is False
    assert r["accepted"] == 0


# --- publish_profile_event delegates its fan-out to publish_event ---

def test_publish_profile_event_delegates_to_publish_event():
    """After its kind-0/signer validation, the fan-out is publish_event's."""
    sk = PrivateKey()
    ev = _signed_kind(sk, 0, {"name": "real"})

    sentinel = {
        "success": True,
        "event_id": ev["id"],
        "accepted": 3,
        "attempted": 4,
        "relays": [
            {"relay": "wss://a", "accepted": True, "error": None},
            {"relay": "wss://b", "accepted": True, "error": None},
            {"relay": "wss://c", "accepted": True, "error": None},
            {"relay": "wss://d", "accepted": False, "error": "wss://d: nope"},
        ],
    }
    with patch.object(nostr_profile, "publish_event", return_value=sentinel) as delegated:
        r = publish_profile_event(ev, sk.public_key.bech32())

    delegated.assert_called_once()
    # It relays the validated event, not something else.
    assert delegated.call_args.args[0] == ev
    # Legacy {success, ok, total, errors} shape preserved for existing consumers.
    assert r == {"success": True, "ok": 3, "total": 4, "errors": ["wss://d: nope"]}


def test_publish_profile_event_does_not_delegate_when_validation_fails():
    """A wrong-kind event is rejected BEFORE any transport happens."""
    sk = PrivateKey()
    ev = _signed_kind(sk, 1, {"hello": "world"})  # kind 1, not 0
    with patch.object(nostr_profile, "publish_event") as delegated:
        r = publish_profile_event(ev, sk.public_key.bech32())
    assert r["success"] is False
    delegated.assert_not_called()
