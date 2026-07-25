"""Tests for the single patron-signing home, PatronSigner."""

from pynostr.key import PrivateKey

from tollbooth.identity_proof import verify_proof
from tollbooth.patron_signer import PatronSigner


def _agent():
    pk = PrivateKey()
    return pk.hex(), pk.public_key.bech32()  # (nsec-hex, npub)


def test_authenticate_injects_npub_and_verifiable_proof():
    nsec, npub = _agent()
    s = PatronSigner(npub, nsec)
    args = s.authenticate("cypher_record_triage", {"repo_name": "x"})
    assert args["repo_name"] == "x"
    assert args["npub"] == npub
    assert verify_proof(args["dpop_token"], npub, "cypher_record_triage") is True


def test_proof_is_bound_to_the_tool():
    nsec, npub = _agent()
    s = PatronSigner(npub, nsec)
    token = s.proof("authority_certify_credits")
    assert verify_proof(token, npub, "authority_certify_credits") is True
    assert verify_proof(token, npub, "something_else") is False


def test_each_call_mints_a_distinct_proof():
    nsec, npub = _agent()
    s = PatronSigner(npub, nsec)
    assert s.proof("t") != s.proof("t")


def test_explicit_npub_is_respected_and_token_is_always_fresh():
    nsec, npub = _agent()
    s = PatronSigner(npub, nsec)
    args = s.authenticate("t", {"npub": npub, "dpop_token": "stale"})
    assert args["npub"] == npub
    assert args["dpop_token"] != "stale"


def test_empty_nsec_yields_empty_proof_not_an_error():
    # A caller that legitimately presents no proof (parity with the old AuthorityCertifier
    # behaviour): the proof is "", never a crash.
    _, npub = _agent()
    s = PatronSigner(npub, "")
    assert s.proof("t") == ""
    assert s.authenticate("t")["dpop_token"] == ""
