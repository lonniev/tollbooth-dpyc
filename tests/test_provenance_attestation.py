"""Tests for Operator provenance attestations on Secure-Courier request DMs.

A proof/credential request DM is delivered from a key other than the
Operator's own npub in the self-addressed case (relays drop self-addressed
DMs). The Operator therefore signs an attestation — with its *registered*
identity key — that is embedded in the (encrypted) DM body and binds the
delivery key + subject + one-time challenge, so a human can verify the
request traces to the registered Operator and is not an impostor.
"""

import json

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.identity_proof import (
    create_ownership_proof,
    create_provenance_attestation,
    verify_provenance_attestation,
)
from tollbooth.nostr_credentials import NostrCredentialExchange


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def operator():
    pk = PrivateKey()
    return pk, pk.public_key.hex(), pk.public_key.bech32()


def _attest(operator_pk, *, sender_hex, subject_npub, service="x", challenge="bold-hawk-42"):
    return create_provenance_attestation(
        operator_pk.nsec,
        sender_pubkey_hex=sender_hex,
        subject_npub=subject_npub,
        service=service,
        challenge=challenge,
    )


def _template() -> dict[str, CredentialTemplate]:
    return {
        "x": CredentialTemplate(
            service="x",
            version=1,
            fields={"api_key": FieldSpec(required=True, sensitive=True)},
            description="Test X API credentials",
        ),
    }


# ---------------------------------------------------------------------------
# Primitive: create / verify roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_valid(operator):
    op_pk, op_hex, _ = operator
    subject = PrivateKey().public_key.bech32()
    ephemeral = PrivateKey().public_key.hex()

    att = _attest(op_pk, sender_hex=ephemeral, subject_npub=subject)
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=ephemeral,
        expected_subject_npub=subject,
        expected_challenge="bold-hawk-42",
    )

    assert res["valid"] is True
    assert res["reason"] == "ok"
    # The recovered signer is the Operator's registered key — the seam the
    # caller resolves against the DPYC registry for the trust state.
    assert res["operator_pubkey_hex"] == op_hex


def test_sender_mismatch_rejected(operator):
    """An attestation lifted onto a DM delivered by a different key fails."""
    op_pk, op_hex, _ = operator
    subject = PrivateKey().public_key.bech32()
    real_ephemeral = PrivateKey().public_key.hex()
    other_ephemeral = PrivateKey().public_key.hex()

    att = _attest(op_pk, sender_hex=real_ephemeral, subject_npub=subject)
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=other_ephemeral,
        expected_subject_npub=subject,
        expected_challenge="bold-hawk-42",
    )

    assert res["valid"] is False
    assert res["reason"] == "sender_mismatch"
    # Signature verified, so the signer is still recoverable.
    assert res["operator_pubkey_hex"] == op_hex


def test_subject_mismatch_rejected(operator):
    op_pk, _, _ = operator
    ephemeral = PrivateKey().public_key.hex()
    att = _attest(op_pk, sender_hex=ephemeral, subject_npub=PrivateKey().public_key.bech32())
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=ephemeral,
        expected_subject_npub=PrivateKey().public_key.bech32(),
        expected_challenge="bold-hawk-42",
    )
    assert res["valid"] is False
    assert res["reason"] == "subject_mismatch"


def test_challenge_mismatch_rejected(operator):
    """A captured attestation cannot be replayed against another exchange."""
    op_pk, _, _ = operator
    subject = PrivateKey().public_key.bech32()
    ephemeral = PrivateKey().public_key.hex()
    att = _attest(op_pk, sender_hex=ephemeral, subject_npub=subject, challenge="bold-hawk-42")
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=ephemeral,
        expected_subject_npub=subject,
        expected_challenge="calm-wolf-99",  # different exchange
    )
    assert res["valid"] is False
    assert res["reason"] == "challenge_mismatch"


def test_impostor_signer_is_surfaced_not_hidden(operator):
    """An impostor with no Operator nsec signs with its OWN key.

    Cryptographically the event is valid, so verify returns valid=True with the
    *impostor's* pubkey — which the caller then fails to resolve in the DPYC
    registry and renders red. The defense lives in registry resolution, and
    this test pins that verify surfaces the true signer to enable it.
    """
    _, op_hex, _ = operator
    impostor = PrivateKey()
    subject = PrivateKey().public_key.bech32()
    ephemeral = PrivateKey().public_key.hex()

    att = _attest(impostor, sender_hex=ephemeral, subject_npub=subject)
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=ephemeral,
        expected_subject_npub=subject,
        expected_challenge="bold-hawk-42",
    )

    assert res["valid"] is True  # validly signed — by the impostor
    assert res["operator_pubkey_hex"] == impostor.public_key.hex()
    assert res["operator_pubkey_hex"] != op_hex  # NOT the registered Operator


def test_tampered_body_fails_signature(operator):
    op_pk, _, _ = operator
    subject = PrivateKey().public_key.bech32()
    ephemeral = PrivateKey().public_key.hex()
    att = _attest(op_pk, sender_hex=ephemeral, subject_npub=subject)

    event = json.loads(att)
    # Flip the bound challenge tag after signing — signature must no longer verify.
    for tag in event["tags"]:
        if tag and tag[0] == "challenge":
            tag[1] = "tampered-tag-1"
    tampered = json.dumps(event)

    res = verify_provenance_attestation(
        tampered,
        expected_sender_pubkey_hex=ephemeral,
        expected_subject_npub=subject,
        expected_challenge="tampered-tag-1",
    )
    assert res["valid"] is False
    assert res["reason"] == "bad_signature"


def test_malformed_json_rejected(operator):
    res = verify_provenance_attestation(
        "not json at all",
        expected_sender_pubkey_hex="x",
        expected_subject_npub="y",
        expected_challenge="z",
    )
    assert res["valid"] is False
    assert res["reason"] == "malformed"


def test_ownership_proof_is_not_an_attestation(operator):
    """A caller's kind-27235 ownership proof must never pass as an attestation."""
    op_pk, _, _ = operator
    ownership = create_ownership_proof(op_pk.nsec)
    res = verify_provenance_attestation(
        ownership,
        expected_sender_pubkey_hex="x",
        expected_subject_npub="y",
        expected_challenge="z",
    )
    assert res["valid"] is False
    assert res["reason"] == "not_attestation"


# ---------------------------------------------------------------------------
# Integration: open_channel embeds a verifiable attestation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_dm_body_attributes_to_registered_npub_and_verifies():
    """The incident case: operator proving its own npub (self-DM).

    The body must attribute the request to the REGISTERED operator npub (not
    the ephemeral delivery key), label the delivery key, and carry an
    attestation that verifies against the ephemeral sender + challenge.
    """
    from unittest.mock import patch

    op = PrivateKey()
    ex = NostrCredentialExchange(
        nsec=op.nsec, relays=["wss://relay.test.com"], templates=_template(),
    )

    with patch.object(ex, "_start_subscription"), \
         patch.object(ex, "_send_dm_as") as mock_send_as:
        result = await ex.open_channel(
            "x", greeting="Hi", recipient_npub=op.public_key.bech32(),
        )

    mock_send_as.assert_called_once()
    welcome = mock_send_as.call_args[0][2]
    ephemeral_hex = ex._ephemeral_agents[(op.public_key.bech32(), "x")].public_key.hex()

    # Registered identity is the Operator line; ephemeral is labeled delivery-only.
    assert f"Operator: {op.public_key.bech32()}" in welcome
    assert "Delivery key:" in welcome
    assert "--- Operator Attestation ---" in welcome

    # Extract and verify the embedded attestation.
    marker = "attestation = @@@"
    start = welcome.index(marker) + len(marker)
    end = welcome.index("@@@", start)
    att = welcome[start:end]
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=ephemeral_hex,
        expected_subject_npub=op.public_key.bech32(),
        expected_challenge=result["dpop_token"],
    )
    assert res["valid"] is True
    assert res["operator_pubkey_hex"] == op.public_key.hex()


@pytest.mark.asyncio
async def test_patron_dm_attestation_binds_operator_own_key():
    """Non-self-DM: no ephemeral, so the attestation binds the Operator key."""
    from unittest.mock import patch

    op = PrivateKey()
    patron = PrivateKey()
    ex = NostrCredentialExchange(
        nsec=op.nsec, relays=["wss://relay.test.com"], templates=_template(),
    )

    with patch.object(ex, "_start_subscription"), \
         patch.object(ex, "send_dm") as mock_send:
        result = await ex.open_channel(
            "x", greeting="Hi", recipient_npub=patron.public_key.bech32(),
        )

    welcome = mock_send.call_args[0][1]
    assert f"Operator: {op.public_key.bech32()}" in welcome
    assert "Delivery key:" not in welcome  # no ephemeral on a patron DM

    marker = "attestation = @@@"
    att = welcome[welcome.index(marker) + len(marker):welcome.index("@@@", welcome.index(marker) + len(marker))]
    res = verify_provenance_attestation(
        att,
        expected_sender_pubkey_hex=op.public_key.hex(),
        expected_subject_npub=patron.public_key.bech32(),
        expected_challenge=result["dpop_token"],
    )
    assert res["valid"] is True
