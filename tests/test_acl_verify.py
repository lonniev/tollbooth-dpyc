"""Tests for ACL event verification and access checks (kind-30078 Nostr events)."""

import json
import time
from unittest.mock import AsyncMock

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.acl_verify import ACL_EVENT_KIND, verify_acl_event, check_acl_access
from tollbooth.operator_proof import OPERATOR_PROOF_KIND


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
def caller_keypair():
    """Generate a caller (non-operator) keypair."""
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


@pytest.fixture()
def other_keypair():
    """Generate a third-party keypair."""
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


def _make_acl_event(
    operator_key: PrivateKey,
    tool_pattern: str = "set_pricing_model",
    allowed_npubs: list[str] | None = None,
    expires_at: int | None = None,
    kind: int = ACL_EVENT_KIND,
    d_tag: str | None = None,
    content_override: str | None = None,
    version: int = 1,
) -> str:
    """Create a signed kind-30078 ACL event."""
    if allowed_npubs is None:
        allowed_npubs = []

    content_dict: dict = {
        "version": version,
        "tool_pattern": tool_pattern,
        "allowed_npubs": allowed_npubs,
    }
    if expires_at is not None:
        content_dict["expires_at"] = expires_at

    content = content_override if content_override is not None else json.dumps(content_dict)
    event = Event(
        kind=kind,
        content=content,
        tags=[["d", d_tag if d_tag is not None else tool_pattern]],
        pubkey=operator_key.public_key.hex(),
    )
    event.created_at = int(time.time())
    event.sign(operator_key.hex())
    return json.dumps(event.to_dict())


def _make_identity_proof(
    private_key: PrivateKey,
    tool_name: str = "set_pricing_model",
) -> str:
    """Create a signed kind-27235 identity proof event."""
    event = Event(
        kind=OPERATOR_PROOF_KIND,
        content="",
        tags=[["u", tool_name]],
        pubkey=private_key.public_key.hex(),
    )
    event.created_at = int(time.time())
    event.sign(private_key.hex())
    return json.dumps(event.to_dict())


# ---------------------------------------------------------------------------
# verify_acl_event
# ---------------------------------------------------------------------------


class TestVerifyAclEvent:
    """verify_acl_event() tests."""

    def test_valid_event(self, operator_keypair, caller_keypair):
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        event_json = _make_acl_event(op_key, allowed_npubs=[caller_npub])
        content = verify_acl_event(event_json, op_npub)
        assert content is not None
        assert content["tool_pattern"] == "set_pricing_model"
        assert caller_npub in content["allowed_npubs"]

    def test_wrong_kind(self, operator_keypair):
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(op_key, kind=1)
        assert verify_acl_event(event_json, op_npub) is None

    def test_wrong_signer(self, operator_keypair, other_keypair):
        op_key, _ = operator_keypair
        _, other_npub = other_keypair
        event_json = _make_acl_event(op_key)
        assert verify_acl_event(event_json, other_npub) is None

    def test_missing_d_tag(self, operator_keypair):
        op_key, op_npub = operator_keypair
        content = json.dumps({
            "version": 1,
            "tool_pattern": "set_pricing_model",
            "allowed_npubs": [],
        })
        event = Event(
            kind=ACL_EVENT_KIND,
            content=content,
            tags=[],  # no d-tag
            pubkey=op_key.public_key.hex(),
        )
        event.created_at = int(time.time())
        event.sign(op_key.hex())
        assert verify_acl_event(json.dumps(event.to_dict()), op_npub) is None

    def test_bad_json_content(self, operator_keypair):
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(op_key, content_override="not json")
        assert verify_acl_event(event_json, op_npub) is None

    def test_missing_required_fields(self, operator_keypair):
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(
            op_key,
            content_override=json.dumps({"version": 1}),
        )
        assert verify_acl_event(event_json, op_npub) is None

    def test_tampered_signature(self, operator_keypair):
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(op_key)
        event_dict = json.loads(event_json)
        event_dict["content"] = json.dumps({"version": 1, "tool_pattern": "hacked", "allowed_npubs": []})
        assert verify_acl_event(json.dumps(event_dict), op_npub) is None

    def test_expired_acl_still_valid_event(self, operator_keypair):
        """An expired ACL event is structurally valid — expiry is checked at access time."""
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(op_key, expires_at=int(time.time()) - 3600)
        content = verify_acl_event(event_json, op_npub)
        assert content is not None  # verify_acl_event doesn't check expiry
        assert content["expires_at"] < time.time()

    def test_invalid_json_input(self, operator_keypair):
        _, op_npub = operator_keypair
        assert verify_acl_event("not json at all", op_npub) is None

    def test_empty_input(self, operator_keypair):
        _, op_npub = operator_keypair
        assert verify_acl_event("", op_npub) is None

    def test_d_tag_mismatch(self, operator_keypair):
        """d-tag must match tool_pattern in content."""
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(op_key, tool_pattern="set_pricing_model", d_tag="wrong_tool")
        assert verify_acl_event(event_json, op_npub) is None


# ---------------------------------------------------------------------------
# check_acl_access
# ---------------------------------------------------------------------------


class TestCheckAclAccess:
    """check_acl_access() orchestration tests."""

    @pytest.mark.asyncio
    async def test_npub_allowed(self, operator_keypair, caller_keypair):
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        event_json = _make_acl_event(op_key, allowed_npubs=[caller_npub])

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=event_json)

        result = await check_acl_access(acl_store, op_npub, caller_npub, "set_pricing_model")
        assert result is True

    @pytest.mark.asyncio
    async def test_npub_denied(self, operator_keypair, caller_keypair, other_keypair):
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        _, other_npub = other_keypair
        # ACL allows other_npub but not caller_npub
        event_json = _make_acl_event(op_key, allowed_npubs=[other_npub])

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=event_json)

        result = await check_acl_access(acl_store, op_npub, caller_npub, "set_pricing_model")
        assert result is False

    @pytest.mark.asyncio
    async def test_wildcard_match(self, operator_keypair, caller_keypair):
        """Wildcard ACL grants access when no exact match exists."""
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        wildcard_event = _make_acl_event(
            op_key, tool_pattern="*", allowed_npubs=[caller_npub],
        )

        call_count = 0

        async def mock_fetch(operator, pattern):
            nonlocal call_count
            call_count += 1
            if pattern == "*":
                return wildcard_event
            return None

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(side_effect=mock_fetch)

        result = await check_acl_access(acl_store, op_npub, caller_npub, "set_pricing_model")
        assert result is True
        assert call_count == 2  # exact miss, then wildcard hit

    @pytest.mark.asyncio
    async def test_expired_acl(self, operator_keypair, caller_keypair):
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        event_json = _make_acl_event(
            op_key, allowed_npubs=[caller_npub],
            expires_at=int(time.time()) - 3600,
        )

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=event_json)

        result = await check_acl_access(acl_store, op_npub, caller_npub, "set_pricing_model")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_acl_rows(self, operator_keypair, caller_keypair):
        _, op_npub = operator_keypair
        _, caller_npub = caller_keypair

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=None)

        result = await check_acl_access(acl_store, op_npub, caller_npub, "set_pricing_model")
        assert result is False

    @pytest.mark.asyncio
    async def test_identity_proof_valid(self, operator_keypair, caller_keypair):
        op_key, op_npub = operator_keypair
        caller_key, caller_npub = caller_keypair
        event_json = _make_acl_event(op_key, allowed_npubs=[caller_npub])
        proof = _make_identity_proof(caller_key, tool_name="set_pricing_model")

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=event_json)

        result = await check_acl_access(
            acl_store, op_npub, caller_npub, "set_pricing_model",
            caller_identity_proof=proof,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_identity_proof_invalid(self, operator_keypair, caller_keypair, other_keypair):
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        other_key, _ = other_keypair
        event_json = _make_acl_event(op_key, allowed_npubs=[caller_npub])
        # Proof signed by other_key, not caller
        bad_proof = _make_identity_proof(other_key, tool_name="set_pricing_model")

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=event_json)

        result = await check_acl_access(
            acl_store, op_npub, caller_npub, "set_pricing_model",
            caller_identity_proof=bad_proof,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_identity_proof_missing_still_passes(self, operator_keypair, caller_keypair):
        """When no identity proof is provided but npub is in ACL, access is granted."""
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        event_json = _make_acl_event(op_key, allowed_npubs=[caller_npub])

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=event_json)

        result = await check_acl_access(
            acl_store, op_npub, caller_npub, "set_pricing_model",
            caller_identity_proof=None,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_exact_match_takes_priority(self, operator_keypair, caller_keypair):
        """Exact tool match is tried first; wildcard is not reached."""
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        exact_event = _make_acl_event(op_key, allowed_npubs=[caller_npub])

        acl_store = AsyncMock()
        acl_store.fetch_acl = AsyncMock(return_value=exact_event)

        result = await check_acl_access(acl_store, op_npub, caller_npub, "set_pricing_model")
        assert result is True
        # Only one call — exact match found, wildcard not queried
        acl_store.fetch_acl.assert_called_once_with(op_npub, "set_pricing_model")
