"""Tests for DPYC identity credential — kind 30080 sign/verify round-trip."""

import json
import time
from unittest.mock import AsyncMock

import pytest
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.identity_credential import (
    IDENTITY_CREDENTIAL_KIND,
    IDENTITY_CREDENTIAL_TAG,
    IDENTITY_CREDENTIAL_LABEL,
    IdentityCredentialError,
    sign_identity_credential,
    verify_identity_credential,
    verify_credential_chain,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def operator_keys():
    """Generate an operator Nostr keypair."""
    pk = PrivateKey()
    return pk, pk.public_key.bech32(), pk.public_key.hex()


@pytest.fixture()
def citizen_keys():
    """Generate a citizen Nostr keypair."""
    pk = PrivateKey()
    return pk, pk.public_key.bech32(), pk.public_key.hex()


# ---------------------------------------------------------------------------
# sign + verify round-trip
# ---------------------------------------------------------------------------


class TestSignAndVerify:
    def test_round_trip(self, operator_keys, citizen_keys):
        """Sign and immediately verify — happy path."""
        _, operator_nsec_bech32, _ = operator_keys
        # Need nsec for signing
        op_pk = operator_keys[0]
        op_nsec = op_pk.bech32()  # nsec bech32

        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_nsec,
            ttl_seconds=3600,
        )

        claims = verify_identity_credential(credential_json, citizen_npub)
        assert claims["citizen_npub"] == citizen_npub
        assert claims["operator_npub"] == operator_keys[1]  # operator npub
        assert claims["jti"]  # non-empty
        assert claims["expiration"] > time.time()
        assert claims["issued_at"]  # non-empty

    def test_event_structure(self, operator_keys, citizen_keys):
        """Verify the raw event has correct kind, tags, and content."""
        op_pk = operator_keys[0]
        _, citizen_npub, citizen_hex = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        event_dict = json.loads(credential_json)
        assert event_dict["kind"] == IDENTITY_CREDENTIAL_KIND
        assert event_dict["pubkey"] == operator_keys[2]  # operator hex

        # Check tags
        tags = event_dict["tags"]
        tag_keys = [t[0] for t in tags]
        assert "d" in tag_keys
        assert "p" in tag_keys
        assert "t" in tag_keys
        assert "L" in tag_keys
        assert "expiration" in tag_keys

        # p-tag should be citizen hex
        p_tags = [t for t in tags if t[0] == "p"]
        assert p_tags[0][1] == citizen_hex

        # t-tag
        t_tags = [t for t in tags if t[0] == "t"]
        assert t_tags[0][1] == IDENTITY_CREDENTIAL_TAG

        # L-tag
        l_tags = [t for t in tags if t[0] == "L"]
        assert l_tags[0][1] == IDENTITY_CREDENTIAL_LABEL

        # Content is valid JSON with expected fields
        content = json.loads(event_dict["content"])
        assert content["citizen_npub"] == citizen_npub
        assert content["operator_npub"] == operator_keys[1]

    def test_verify_without_expected_citizen(self, operator_keys, citizen_keys):
        """Verify works without specifying expected citizen."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        claims = verify_identity_credential(credential_json)
        assert claims["citizen_npub"] == citizen_npub


# ---------------------------------------------------------------------------
# Verification failures
# ---------------------------------------------------------------------------


class TestVerifyFailures:
    def test_expired_credential(self, operator_keys, citizen_keys):
        """Credential with TTL in the past is rejected."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
            ttl_seconds=-10,  # already expired
        )

        with pytest.raises(IdentityCredentialError, match="expired"):
            verify_identity_credential(credential_json, citizen_npub)

    def test_wrong_citizen_npub(self, operator_keys, citizen_keys):
        """Credential for wrong citizen is rejected."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys
        other_pk = PrivateKey()
        other_npub = other_pk.public_key.bech32()

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        with pytest.raises(IdentityCredentialError, match="does not match"):
            verify_identity_credential(credential_json, other_npub)

    def test_tampered_content(self, operator_keys, citizen_keys):
        """Tampered event content fails signature check."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        event_dict = json.loads(credential_json)
        event_dict["content"] = '{"citizen_npub":"npub1fake","operator_npub":"npub1fake","issued_at":"2026-01-01"}'

        with pytest.raises(IdentityCredentialError, match="invalid|tampering"):
            verify_identity_credential(json.dumps(event_dict), citizen_npub)

    def test_wrong_kind(self, operator_keys, citizen_keys):
        """Event with wrong kind is rejected."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        event_dict = json.loads(credential_json)
        event_dict["kind"] = 30079  # wrong kind
        # Re-sign needed for valid sig, but the kind check happens after sig
        # So this should still fail on kind mismatch (if sig passes) or sig (if sig fails)
        with pytest.raises(IdentityCredentialError):
            verify_identity_credential(json.dumps(event_dict), citizen_npub)

    def test_invalid_json(self):
        """Non-JSON input is rejected."""
        with pytest.raises(IdentityCredentialError, match="not valid JSON"):
            verify_identity_credential("not json at all")

    def test_missing_expiration_tag(self, operator_keys, citizen_keys):
        """Event without expiration tag is rejected."""
        from pynostr.event import Event  # type: ignore[import-untyped]

        op_pk = operator_keys[0]
        _, citizen_npub, citizen_hex = citizen_keys

        # Build event manually without expiration
        event = Event(
            kind=IDENTITY_CREDENTIAL_KIND,
            content=json.dumps({"citizen_npub": citizen_npub, "operator_npub": operator_keys[1], "issued_at": "now"}),
            tags=[["d", "test-jti"], ["p", citizen_hex], ["t", IDENTITY_CREDENTIAL_TAG]],
            pubkey=operator_keys[2],
            created_at=int(time.time()),
        )
        event.sign(op_pk.hex())

        with pytest.raises(IdentityCredentialError, match="expiration"):
            verify_identity_credential(json.dumps(event.to_dict()), citizen_npub)


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------


class TestVerifyCredentialChain:
    @pytest.mark.asyncio
    async def test_registered_operator_passes(self, operator_keys, citizen_keys):
        """Credential signed by registered operator passes chain verification."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        registry = AsyncMock()
        registry.check_membership = AsyncMock(return_value={
            "npub": operator_keys[1],
            "role": "operator",
            "status": "active",
        })

        claims = await verify_credential_chain(credential_json, citizen_npub, registry)
        assert claims["citizen_npub"] == citizen_npub
        registry.check_membership.assert_called_once_with(operator_keys[1])

    @pytest.mark.asyncio
    async def test_unregistered_operator_fails(self, operator_keys, citizen_keys):
        """Credential signed by unregistered operator fails chain verification."""
        op_pk = operator_keys[0]
        _, citizen_npub, _ = citizen_keys

        credential_json = sign_identity_credential(
            citizen_npub=citizen_npub,
            operator_nsec=op_pk.bech32(),
        )

        registry = AsyncMock()
        registry.check_membership = AsyncMock(
            side_effect=Exception("Not found in registry")
        )

        with pytest.raises(IdentityCredentialError, match="not a registered"):
            await verify_credential_chain(credential_json, citizen_npub, registry)


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class TestIdentityCredentialError:
    def test_is_exception(self):
        assert issubclass(IdentityCredentialError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(IdentityCredentialError, match="test"):
            raise IdentityCredentialError("test")
