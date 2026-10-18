"""sign_request mints a unique event per call (nonce), so rapid identical requests
never collide on the replay guard — while each still verifies and stays bound to
its own method, path and body."""

import base64
import hashlib
import json

from pynostr.key import PrivateKey

from tollbooth.identity_proof import sign_request, verify_request_signature

URL = "https://op.example/mcp"


def _event(header: str) -> dict:
    return json.loads(base64.b64decode(header.removeprefix("Nostr ")))


def test_two_identical_requests_have_distinct_event_ids():
    nsec = PrivateKey().hex()
    a = _event(sign_request(nsec, method="POST", url=URL, body=b"{}"))
    b = _event(sign_request(nsec, method="POST", url=URL, body=b"{}"))
    assert a["id"] != b["id"]


def test_signed_request_verifies_and_is_body_bound():
    pk = PrivateKey()
    header = sign_request(pk.hex(), method="POST", url=URL, body=b'{"a":1}')
    npub, reason = verify_request_signature(
        header.removeprefix("Nostr "), method="POST", path="/mcp",
        body_sha256=hashlib.sha256(b'{"a":1}').hexdigest(),
    )
    assert (npub, reason) == (pk.public_key.bech32(), None)

    other = sign_request(pk.hex(), method="POST", url=URL, body=b'{"a":1}')
    npub, reason = verify_request_signature(
        other.removeprefix("Nostr "), method="POST", path="/mcp",
        body_sha256=hashlib.sha256(b'{"a":2}').hexdigest(),
    )
    assert npub is None and reason == "request_mismatch"


def test_nonce_tag_present_and_request_tags_intact():
    ev = _event(sign_request(PrivateKey().hex(), method="get", url=URL, body=b""))
    tags = {tag[0]: tag[1] for tag in ev["tags"] if len(tag) >= 2}
    assert tags["u"] == URL
    assert tags["method"] == "GET"
    assert tags["payload"] == hashlib.sha256(b"").hexdigest()
    assert len(tags.get("nonce", "")) >= 16
