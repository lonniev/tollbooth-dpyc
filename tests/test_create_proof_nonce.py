"""create_proof mints a unique event per call (nonce), so rapid same-tool callers
never collide on the replay guard — while still verifying and staying tool-bound."""

import json

from pynostr.key import PrivateKey

from tollbooth.identity_proof import create_proof, verify_proof


def _npub():
    pk = PrivateKey()
    return pk.hex(), pk.public_key.bech32()


def test_two_proofs_for_same_tool_have_distinct_event_ids():
    nsec, _ = _npub()
    a = json.loads(create_proof(nsec, "cypher_create_query"))
    b = json.loads(create_proof(nsec, "cypher_create_query"))
    # Same wall-clock second is likely here; the nonce must still make them differ.
    assert a["id"] != b["id"]
    assert a["created_at"] == b["created_at"] or True  # (may or may not share the second)


def test_nonce_proof_still_verifies_and_is_tool_bound():
    nsec, npub = _npub()
    token = create_proof(nsec, "cypher_assert_rationale")
    assert verify_proof(token, npub, "cypher_assert_rationale") is True
    assert verify_proof(token, npub, "cypher_record_triage") is False


def test_nonce_tag_present_and_u_tag_intact():
    nsec, _ = _npub()
    ev = json.loads(create_proof(nsec, "t"))
    tags = {tag[0]: tag[1] for tag in ev["tags"] if len(tag) >= 2}
    assert tags.get("u") == "t"
    assert len(tags.get("nonce", "")) >= 16
