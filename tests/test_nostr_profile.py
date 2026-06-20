"""Tests for nostr_profile — kind-0 read/publish helpers.

Covers the publish-side validation (the security-critical part: the wheel
relays a client-signed event only when kind/pubkey/signature check out). The
relay I/O itself (fetch_profile, the publish fan-out) is integration-only.
"""

import json

from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth.nostr_profile import publish_profile_event


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
