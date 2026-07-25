"""Tests for operator proof verification (kind-27235 Nostr events)."""

import json
import time

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.identity_proof import PROOF_EVENT_KIND as OPERATOR_PROOF_KIND
from tollbooth.identity_proof import verify_proof as verify_identity_proof
from tollbooth.identity_proof import verify_proof as verify_operator_proof

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def operator_keypair():
    """Generate an operator Nostr keypair for testing."""
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


@pytest.fixture()
def other_keypair():
    """Generate a different keypair (non-operator)."""
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


def _make_proof(
    private_key: PrivateKey,
    tool_name: str = "set_pricing_model",
    kind: int = OPERATOR_PROOF_KIND,
    created_at: int | None = None,
) -> str:
    """Create a signed kind-27235 proof event."""
    event = Event(
        kind=kind,
        content="",
        tags=[["u", tool_name]],
        pubkey=private_key.public_key.hex(),
    )
    if created_at is not None:
        event.created_at = created_at
    else:
        event.created_at = int(time.time())
    event.sign(private_key.hex())
    return json.dumps(event.to_dict())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVerifyOperatorProof:
    """verify_operator_proof() tests."""

    def test_valid_proof(self, operator_keypair):
        """A correctly signed proof for the right operator and tool passes."""
        private_key, npub = operator_keypair
        dpop_token = _make_proof(private_key, tool_name="set_pricing_model")
        assert verify_operator_proof(dpop_token, npub, "set_pricing_model") is True

    def test_wrong_operator_npub(self, operator_keypair, other_keypair):
        """Proof signed by a different key is rejected."""
        private_key, _ = operator_keypair
        _, other_npub = other_keypair
        dpop_token = _make_proof(private_key, tool_name="set_pricing_model")
        assert verify_operator_proof(dpop_token, other_npub, "set_pricing_model") is False

    def test_wrong_tool_name(self, operator_keypair):
        """Proof for a different tool is rejected."""
        private_key, npub = operator_keypair
        dpop_token = _make_proof(private_key, tool_name="set_pricing_model")
        assert verify_operator_proof(dpop_token, npub, "certify_credits") is False

    def test_expired_proof(self, operator_keypair):
        """Proof older than 60 seconds is rejected."""
        private_key, npub = operator_keypair
        dpop_token = _make_proof(
            private_key,
            tool_name="set_pricing_model",
            created_at=int(time.time()) - 120,
        )
        assert verify_operator_proof(dpop_token, npub, "set_pricing_model") is False

    def test_future_proof_within_window(self, operator_keypair):
        """Proof slightly in the future (clock skew) is accepted."""
        private_key, npub = operator_keypair
        dpop_token = _make_proof(
            private_key,
            tool_name="set_pricing_model",
            created_at=int(time.time()) + 30,
        )
        assert verify_operator_proof(dpop_token, npub, "set_pricing_model") is True

    def test_future_proof_outside_window(self, operator_keypair):
        """Proof far in the future is rejected."""
        private_key, npub = operator_keypair
        dpop_token = _make_proof(
            private_key,
            tool_name="set_pricing_model",
            created_at=int(time.time()) + 120,
        )
        assert verify_operator_proof(dpop_token, npub, "set_pricing_model") is False

    def test_wrong_kind(self, operator_keypair):
        """Event with a non-27235 kind is rejected."""
        private_key, npub = operator_keypair
        dpop_token = _make_proof(private_key, tool_name="set_pricing_model", kind=1)
        assert verify_operator_proof(dpop_token, npub, "set_pricing_model") is False

    def test_invalid_json(self, operator_keypair):
        """Non-JSON input returns False."""
        _, npub = operator_keypair
        assert verify_operator_proof("not json", npub, "set_pricing_model") is False

    def test_empty_proof(self, operator_keypair):
        """Empty string returns False."""
        _, npub = operator_keypair
        assert verify_operator_proof("", npub, "set_pricing_model") is False

    def test_tampered_signature(self, operator_keypair):
        """Modifying the event after signing causes rejection."""
        private_key, npub = operator_keypair
        proof_json = _make_proof(private_key, tool_name="set_pricing_model")
        event_dict = json.loads(proof_json)
        # Tamper with content
        event_dict["content"] = "tampered"
        tampered = json.dumps(event_dict)
        assert verify_operator_proof(tampered, npub, "set_pricing_model") is False


# ---------------------------------------------------------------------------
# verify_identity_proof (general function)
# ---------------------------------------------------------------------------


class TestVerifyIdentityProof:
    """verify_identity_proof() tests with non-operator keypairs."""

    def test_valid_proof(self, other_keypair):
        """A correctly signed proof for an arbitrary npub passes."""
        private_key, npub = other_keypair
        dpop_token = _make_proof(private_key, tool_name="search")
        assert verify_identity_proof(dpop_token, npub, "search") is True

    def test_wrong_npub(self, operator_keypair, other_keypair):
        """Proof signed by one key is rejected when verified against another."""
        private_key, _ = operator_keypair
        _, other_npub = other_keypair
        dpop_token = _make_proof(private_key, tool_name="search")
        assert verify_identity_proof(dpop_token, other_npub, "search") is False

    def test_wrong_tool(self, other_keypair):
        """Proof for a different tool is rejected."""
        private_key, npub = other_keypair
        dpop_token = _make_proof(private_key, tool_name="search")
        assert verify_identity_proof(dpop_token, npub, "other_tool") is False

    def test_expired(self, other_keypair):
        """Proof older than the window is rejected."""
        private_key, npub = other_keypair
        dpop_token = _make_proof(private_key, tool_name="search", created_at=int(time.time()) - 120)
        assert verify_identity_proof(dpop_token, npub, "search") is False

    def test_custom_window(self, other_keypair):
        """A custom window_seconds is respected."""
        private_key, npub = other_keypair
        dpop_token = _make_proof(private_key, tool_name="search", created_at=int(time.time()) - 90)
        # Default 60s window rejects
        assert verify_identity_proof(dpop_token, npub, "search", window_seconds=60) is False
        # 120s window accepts
        assert verify_identity_proof(dpop_token, npub, "search", window_seconds=120) is True

    def test_invalid_json(self):
        """Non-JSON input returns False."""
        assert verify_identity_proof("not json", "npub1fake", "search") is False

    def test_tampered(self, other_keypair):
        """Tampered event is rejected."""
        private_key, npub = other_keypair
        proof_json = _make_proof(private_key, tool_name="search")
        event_dict = json.loads(proof_json)
        event_dict["content"] = "tampered"
        assert verify_identity_proof(json.dumps(event_dict), npub, "search") is False
