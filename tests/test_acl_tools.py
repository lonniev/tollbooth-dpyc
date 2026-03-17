"""Tests for ACL tool functions."""

import json
import time
from unittest.mock import AsyncMock

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.acl_verify import ACL_EVENT_KIND
from tollbooth.tools.acl import set_tool_acl_tool, get_tool_acls_tool, delete_tool_acl_tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def operator_keypair():
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


@pytest.fixture()
def caller_keypair():
    private_key = PrivateKey()
    npub = private_key.public_key.bech32()
    return private_key, npub


def _make_acl_event(
    operator_key: PrivateKey,
    tool_pattern: str = "set_pricing_model",
    allowed_npubs: list[str] | None = None,
) -> str:
    """Create a signed kind-30078 ACL event."""
    content = json.dumps({
        "version": 1,
        "tool_pattern": tool_pattern,
        "allowed_npubs": allowed_npubs or [],
    })
    event = Event(
        kind=ACL_EVENT_KIND,
        content=content,
        tags=[["d", tool_pattern]],
        pubkey=operator_key.public_key.hex(),
    )
    event.created_at = int(time.time())
    event.sign(operator_key.hex())
    return json.dumps(event.to_dict())


# ---------------------------------------------------------------------------
# set_tool_acl_tool
# ---------------------------------------------------------------------------


class TestSetToolAclTool:
    @pytest.mark.asyncio
    async def test_valid_acl(self, operator_keypair, caller_keypair):
        op_key, op_npub = operator_keypair
        _, caller_npub = caller_keypair
        event_json = _make_acl_event(op_key, allowed_npubs=[caller_npub])

        acl_store = AsyncMock()
        acl_store.ensure_schema = AsyncMock()
        acl_store.store_acl = AsyncMock(return_value="acl-uuid-1")

        result = await set_tool_acl_tool(acl_store, op_npub, event_json)
        assert result["status"] == "ok"
        assert result["tool_pattern"] == "set_pricing_model"
        assert result["allowed_count"] == 1
        assert result["acl_id"] == "acl-uuid-1"

    @pytest.mark.asyncio
    async def test_invalid_event(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()

        result = await set_tool_acl_tool(acl_store, op_npub, "bad json")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_wrong_signer(self, operator_keypair, other_keypair=None):
        """Event signed by a different key is rejected."""
        _, op_npub = operator_keypair
        other_key = PrivateKey()
        event_json = _make_acl_event(other_key)

        acl_store = AsyncMock()
        result = await set_tool_acl_tool(acl_store, op_npub, event_json)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_store_error(self, operator_keypair):
        op_key, op_npub = operator_keypair
        event_json = _make_acl_event(op_key)

        acl_store = AsyncMock()
        acl_store.ensure_schema = AsyncMock()
        acl_store.store_acl = AsyncMock(side_effect=RuntimeError("db error"))

        result = await set_tool_acl_tool(acl_store, op_npub, event_json)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# get_tool_acls_tool
# ---------------------------------------------------------------------------


class TestGetToolAclsTool:
    @pytest.mark.asyncio
    async def test_returns_acls(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()
        acl_store.fetch_all_acls = AsyncMock(return_value=[
            {"tool_pattern": "set_pricing_model", "signed_event_json": "{}"},
        ])

        result = await get_tool_acls_tool(acl_store, op_npub)
        assert result["status"] == "ok"
        assert len(result["acls"]) == 1

    @pytest.mark.asyncio
    async def test_returns_empty(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()
        acl_store.fetch_all_acls = AsyncMock(return_value=[])

        result = await get_tool_acls_tool(acl_store, op_npub)
        assert result["status"] == "ok"
        assert result["acls"] == []

    @pytest.mark.asyncio
    async def test_store_error(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()
        acl_store.fetch_all_acls = AsyncMock(side_effect=RuntimeError("db error"))

        result = await get_tool_acls_tool(acl_store, op_npub)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# delete_tool_acl_tool
# ---------------------------------------------------------------------------


class TestDeleteToolAclTool:
    @pytest.mark.asyncio
    async def test_deletes_existing(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()
        acl_store.delete_acl = AsyncMock(return_value=True)

        result = await delete_tool_acl_tool(acl_store, op_npub, "set_pricing_model")
        assert result["status"] == "ok"
        assert result["deleted"] is True

    @pytest.mark.asyncio
    async def test_not_found(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()
        acl_store.delete_acl = AsyncMock(return_value=False)

        result = await delete_tool_acl_tool(acl_store, op_npub, "nonexistent")
        assert result["status"] == "ok"
        assert result["deleted"] is False

    @pytest.mark.asyncio
    async def test_store_error(self, operator_keypair):
        _, op_npub = operator_keypair
        acl_store = AsyncMock()
        acl_store.delete_acl = AsyncMock(side_effect=RuntimeError("db error"))

        result = await delete_tool_acl_tool(acl_store, op_npub, "test")
        assert result["status"] == "error"
