"""Tests for patron-signed delegation grants ("secretary" authorization).

A patron who has proven ownership of an npub may want an ephemeral "secretary"
key to publish Nostr notes on their behalf — without ever handing over the
patron nsec. A bare "this key acts for me" assertion is unverifiable, so the
delegation is anchored in a grant the PATRON signs (with the asset an impostor
does not hold — the patron nsec) that names the ephemeral secretary pubkey, the
scope it is allowed, an optional binding to the ``request_npub_proof``
challenge, and an optional expiry. Any verifier can then confirm the secretary
is genuinely authorized rather than taking its word for it.

Mirrors ``test_provenance_attestation.py``: the grant is the reverse-direction
counterpart — there the *Operator* attests a request; here the *patron*
authorizes a secretary.
"""

import json
import time

import pytest
from pynostr.key import PrivateKey

from tollbooth.identity_proof import (
    DELEGATION_GRANT_TOOL,
    create_delegation_grant,
    create_ownership_proof,
    verify_delegation_grant,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def patron():
    pk = PrivateKey()
    return pk, pk.public_key.hex(), pk.public_key.bech32()


@pytest.fixture()
def secretary():
    pk = PrivateKey()
    return pk, pk.public_key.hex()


def _grant(patron_pk, *, secretary_hex, scope="nostr_publish", challenge=None, expires_at=None):
    return create_delegation_grant(
        patron_pk.nsec,
        secretary_pubkey_hex=secretary_hex,
        scope=scope,
        challenge=challenge,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# Primitive: create / verify roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_valid(patron, secretary):
    patron_pk, patron_hex, patron_npub = patron
    _, sec_hex = secretary

    grant = _grant(patron_pk, secretary_hex=sec_hex)
    res = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
    )

    assert res["valid"] is True
    assert res["reason"] == "ok"
    # The recovered signer is the patron's own key — the whole basis for trust.
    assert res["patron_pubkey_hex"] == patron_hex
    assert res["secretary_pubkey_hex"] == sec_hex
    assert res["scope"] == "nostr_publish"


def test_scope_signed_in_and_enforced(patron, secretary):
    patron_pk, _, patron_npub = patron
    _, sec_hex = secretary
    grant = _grant(patron_pk, secretary_hex=sec_hex, scope="nostr_publish")

    tags = {t[0]: t[1] for t in json.loads(grant)["tags"] if len(t) >= 2}
    assert tags.get("scope") == "nostr_publish"

    # A verifier that demands a DIFFERENT scope must refuse.
    res = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
        expected_scope="credential_write",
    )
    assert res["valid"] is False
    assert res["reason"] == "scope_mismatch"

    # The matching scope passes.
    res_ok = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
        expected_scope="nostr_publish",
    )
    assert res_ok["valid"] is True


def test_challenge_binds_grant_to_proof_exchange(patron, secretary):
    """Folding the grant into the request_npub_proof round-trip: the grant may
    carry the one-time dpop_token so it cannot be replayed against another
    exchange."""
    patron_pk, _, patron_npub = patron
    _, sec_hex = secretary
    grant = _grant(patron_pk, secretary_hex=sec_hex, challenge="bold-hawk-42")

    # Verifier bound to a DIFFERENT exchange refuses.
    res = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
        expected_challenge="calm-wolf-99",
    )
    assert res["valid"] is False
    assert res["reason"] == "challenge_mismatch"

    # The right exchange passes.
    res_ok = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
        expected_challenge="bold-hawk-42",
    )
    assert res_ok["valid"] is True


def test_challenge_absent_when_omitted(patron, secretary):
    patron_pk, _, _ = patron
    _, sec_hex = secretary
    grant = _grant(patron_pk, secretary_hex=sec_hex)
    assert "challenge" not in [t[0] for t in json.loads(grant)["tags"]]


def test_expiry_signed_in_and_enforced(patron, secretary):
    """A time-bounded grant: a secretary key must not be able to act forever."""
    patron_pk, _, patron_npub = patron
    _, sec_hex = secretary

    # Already-expired grant is refused.
    expired = _grant(patron_pk, secretary_hex=sec_hex, expires_at=int(time.time()) - 5)
    res = verify_delegation_grant(
        expired,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
    )
    assert res["valid"] is False
    assert res["reason"] == "expired"

    # A grant valid for another hour passes and surfaces its expiry.
    future = int(time.time()) + 3600
    live = _grant(patron_pk, secretary_hex=sec_hex, expires_at=future)
    res_ok = verify_delegation_grant(
        live,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
    )
    assert res_ok["valid"] is True
    assert res_ok["expires_at"] == future


def test_expiry_absent_means_no_expiry(patron, secretary):
    patron_pk, _, patron_npub = patron
    _, sec_hex = secretary
    grant = _grant(patron_pk, secretary_hex=sec_hex)  # no expires_at
    assert "grant_expires" not in [t[0] for t in json.loads(grant)["tags"]]
    res = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
    )
    assert res["valid"] is True
    assert res["expires_at"] is None


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_secretary_mismatch_rejected(patron, secretary):
    """A grant naming one secretary cannot authorize a different ephemeral key."""
    patron_pk, patron_hex, patron_npub = patron
    _, real_sec_hex = secretary
    other_sec_hex = PrivateKey().public_key.hex()

    grant = _grant(patron_pk, secretary_hex=real_sec_hex)
    res = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=other_sec_hex,
    )
    assert res["valid"] is False
    assert res["reason"] == "secretary_mismatch"
    # Signature verified, so the true patron signer is still recoverable.
    assert res["patron_pubkey_hex"] == patron_hex


def test_wrong_patron_rejected(patron, secretary):
    """A grant genuinely signed by someone else must not pass as this patron's."""
    _, _, patron_npub = patron
    impostor = PrivateKey()
    _, sec_hex = secretary

    grant = _grant(impostor, secretary_hex=sec_hex)
    res = verify_delegation_grant(
        grant,
        expected_patron_npub=patron_npub,  # NOT the signer
        expected_secretary_pubkey_hex=sec_hex,
    )
    assert res["valid"] is False
    assert res["reason"] == "patron_mismatch"
    # The verify surfaces the true (impostor) signer for the caller to judge.
    assert res["patron_pubkey_hex"] == impostor.public_key.hex()


def test_tampered_secretary_fails_signature(patron, secretary):
    """Swapping the named secretary after signing breaks the signature — an
    attacker cannot lift a patron's grant onto their own ephemeral key."""
    patron_pk, _, patron_npub = patron
    _, sec_hex = secretary
    attacker_hex = PrivateKey().public_key.hex()

    grant = _grant(patron_pk, secretary_hex=sec_hex)
    event = json.loads(grant)
    for tag in event["tags"]:
        if tag and tag[0] == "secretary":
            tag[1] = attacker_hex
    tampered = json.dumps(event)

    res = verify_delegation_grant(
        tampered,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=attacker_hex,
    )
    assert res["valid"] is False
    assert res["reason"] == "bad_signature"


def test_malformed_json_rejected():
    res = verify_delegation_grant(
        "not json at all",
        expected_patron_npub="npub1x",
        expected_secretary_pubkey_hex="y",
    )
    assert res["valid"] is False
    assert res["reason"] == "malformed"


def test_ownership_proof_is_not_a_grant(patron, secretary):
    """A patron's kind-27235 ownership proof must never pass as a grant — the
    ``u`` sentinel keeps the two disjoint so an ownership proof can't be
    re-read as an open-ended delegation."""
    patron_pk, _, patron_npub = patron
    _, sec_hex = secretary
    ownership = create_ownership_proof(patron_pk.nsec)
    res = verify_delegation_grant(
        ownership,
        expected_patron_npub=patron_npub,
        expected_secretary_pubkey_hex=sec_hex,
    )
    assert res["valid"] is False
    assert res["reason"] == "not_grant"


def test_grant_sentinel_is_distinct():
    """The grant sentinel must differ from the ownership + attestation ones so
    the three kind-27235 roles can never be confused."""
    from tollbooth.identity_proof import (
        OWNERSHIP_SENTINEL,
        PROVENANCE_ATTESTATION_TOOL,
    )

    assert DELEGATION_GRANT_TOOL not in (OWNERSHIP_SENTINEL, PROVENANCE_ATTESTATION_TOOL)
