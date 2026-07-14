"""Tests for tollbooth.agent_keyring.signed_arguments — the per-call proof injection.

The FastMCP proxy/middleware plumbing is FastMCP's; the logic we own is: inject the npub
and a fresh, tool-bound, verifiable kind-27235 proof into each call's arguments. That is
what these tests pin — no FastMCP import required.
"""

from pynostr.key import PrivateKey

from tollbooth.agent_keyring import signed_arguments
from tollbooth.identity_proof import verify_proof


def _agent():
    pk = PrivateKey()
    return pk.hex(), pk.public_key.bech32()  # (nsec-hex, npub)


def test_injects_npub_and_verifiable_proof():
    nsec, npub = _agent()
    out = signed_arguments("cypher_record_triage", {"repo_name": "x", "issue_number": 1}, npub, nsec)
    # original args preserved
    assert out["repo_name"] == "x" and out["issue_number"] == 1
    # identity injected
    assert out["npub"] == npub
    # the proof verifies for this npub AND is bound to this exact tool
    assert verify_proof(out["dpop_token"], npub, "cypher_record_triage") is True


def test_proof_is_bound_to_the_tool_name():
    nsec, npub = _agent()
    out = signed_arguments("cypher_assert_rationale", None, npub, nsec)
    assert verify_proof(out["dpop_token"], npub, "cypher_assert_rationale") is True
    # a proof for one tool must not pass for another (replay across the surface)
    assert verify_proof(out["dpop_token"], npub, "cypher_record_triage") is False


def test_does_not_override_an_explicit_npub_but_always_signs():
    nsec, npub = _agent()
    # caller-supplied npub is respected; dpop_token is always (re)minted fresh
    out = signed_arguments("cypher_note_rejection", {"npub": npub, "dpop_token": "stale"}, npub, nsec)
    assert out["npub"] == npub
    assert out["dpop_token"] != "stale"
    assert verify_proof(out["dpop_token"], npub, "cypher_note_rejection") is True


def test_two_calls_mint_distinct_proofs():
    nsec, npub = _agent()
    a = signed_arguments("cypher_record_triage", {}, npub, nsec)["dpop_token"]
    b = signed_arguments("cypher_record_triage", {}, npub, nsec)["dpop_token"]
    # each call is its own in-memory proof event (distinct signatures / ids)
    assert a != b
