"""Tests for tollbooth.oracle_client — OracleClient."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.oracle_client import OracleClient, OracleClientError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_block(data: dict) -> MagicMock:
    """Create a mock TextContent block with .text = JSON string."""
    block = MagicMock()
    block.text = json.dumps(data)
    return block


# ---------------------------------------------------------------------------
# call_tool() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_success():
    """call_tool() returns parsed dict on success."""
    oracle_response = {
        "success": True,
        "rate_percent": 2,
        "min_sats": 10,
    }

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(oracle_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("get_tax_rate")

    assert result["rate_percent"] == 2
    assert result["min_sats"] == 10
    mock_client.call_tool.assert_awaited_once_with("get_tax_rate", {})


@pytest.mark.asyncio
async def test_call_tool_with_arguments():
    """call_tool() passes arguments to the Oracle tool."""
    oracle_response = {
        "npub": "npub1abc...",
        "role": "citizen",
        "status": "active",
    }

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(oracle_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("lookup_member", {"npub": "npub1abc..."})

    assert result["role"] == "citizen"
    mock_client.call_tool.assert_awaited_once_with(
        "lookup_member", {"npub": "npub1abc..."}
    )


# ---------------------------------------------------------------------------
# call_tool() — connection failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_connection_failure():
    """call_tool() wraps connection errors in OracleClientError."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        with pytest.raises(OracleClientError, match="Failed to connect"):
            await client.call_tool("get_tax_rate")


# ---------------------------------------------------------------------------
# call_tool() — CallToolResult unwrapping (structured data path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_calltoolresult_data():
    """call_tool() unwraps CallToolResult-like objects with .data dict."""
    oracle_response = {
        "rate_percent": 2,
        "min_sats": 10,
    }

    call_tool_result = MagicMock()
    call_tool_result.data = oracle_response
    call_tool_result.content = None  # data takes priority

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=call_tool_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("get_tax_rate")

    assert result["rate_percent"] == 2
    assert result["min_sats"] == 10


@pytest.mark.asyncio
async def test_call_tool_calltoolresult_content():
    """call_tool() unwraps CallToolResult-like objects with .content list."""
    oracle_response = {
        "rate_percent": 2,
        "min_sats": 10,
    }

    call_tool_result = MagicMock(spec=[])  # no .data attribute
    call_tool_result.content = [_text_block(oracle_response)]

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=call_tool_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("get_tax_rate")

    assert result["rate_percent"] == 2


# ---------------------------------------------------------------------------
# call_tool() — plain text responses (non-JSON)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_plain_text():
    """call_tool() wraps plain-text responses in {"text": ...}."""
    markdown = "## How to Join\n\nStep 1: Generate a Nostr keypair..."

    text_block = MagicMock()
    text_block.text = markdown

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[text_block])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("how_to_join")

    assert result == {"text": markdown}


@pytest.mark.asyncio
async def test_call_tool_plain_text_calltoolresult():
    """call_tool() handles TextContent inside CallToolResult for plain text."""
    markdown = "# DPYC\n\nDon't Pester Your Customer..."

    text_block = MagicMock()
    text_block.text = markdown

    call_tool_result = MagicMock(spec=[])  # no .data attribute
    call_tool_result.content = [text_block]

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=call_tool_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("about")

    assert result == {"text": markdown}


@pytest.mark.asyncio
async def test_call_tool_json_non_dict():
    """call_tool() wraps JSON arrays/scalars as {"text": ...}."""
    text_block = MagicMock()
    text_block.text = '["item1", "item2"]'

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[text_block])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        result = await client.call_tool("some_tool")

    assert result == {"text": '["item1", "item2"]'}


# ---------------------------------------------------------------------------
# call_tool() — unexpected response format (no text blocks at all)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_unexpected_format():
    """call_tool() raises on empty list with no text blocks."""
    empty_block = MagicMock(spec=[])  # no .text attribute

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[empty_block])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.oracle_client.Client", return_value=mock_client):
        client = OracleClient(oracle_url="https://oracle.example.com/mcp")
        with pytest.raises(OracleClientError, match="unexpected response"):
            await client.call_tool("get_tax_rate")


# ---------------------------------------------------------------------------
# resolve_oracle_service — registry integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_oracle_service_success():
    """resolve_oracle_service walks authority chain to Prime Authority Oracle."""
    from unittest.mock import MagicMock as SyncMock

    import httpx

    from tollbooth.registry import DPYCRegistry

    members = [
        {
            "npub": "npub1operator",
            "role": "operator",
            "status": "active",
            "upstream_authority_npub": "npub1authority",
        },
        {
            "npub": "npub1authority",
            "role": "authority",
            "status": "active",
            "upstream_authority_npub": "npub1prime",
            "services": [
                {
                    "name": "tollbooth-authority",
                    "url": "https://authority.fastmcp.app/mcp",
                }
            ],
        },
        {
            "npub": "npub1prime",
            "role": "prime_authority",
            "status": "active",
            "upstream_authority_npub": None,
            "services": [
                {
                    "name": "dpyc-oracle",
                    "url": "https://dpyc-oracle.fastmcp.app/mcp",
                    "description": "DPYC Oracle",
                }
            ],
        },
    ]

    resp = SyncMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"members": members}
    resp.raise_for_status = SyncMock()

    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(return_value=resp)

    result = await registry.resolve_oracle_service("npub1operator")
    assert result["npub"] == "npub1prime"
    assert result["url"] == "https://dpyc-oracle.fastmcp.app/mcp"
    assert result["name"] == "dpyc-oracle"
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_oracle_service_no_oracle():
    """resolve_oracle_service raises when Prime Authority has no Oracle service."""
    from unittest.mock import MagicMock as SyncMock

    import httpx

    from tollbooth.registry import DPYCRegistry, RegistryError

    members = [
        {
            "npub": "npub1operator",
            "role": "operator",
            "status": "active",
            "upstream_authority_npub": "npub1prime",
        },
        {
            "npub": "npub1prime",
            "role": "prime_authority",
            "status": "active",
            "upstream_authority_npub": None,
            "services": [],
        },
    ]

    resp = SyncMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"members": members}
    resp.raise_for_status = SyncMock()

    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(return_value=resp)

    with pytest.raises(RegistryError, match="no dpyc-oracle service"):
        await registry.resolve_oracle_service("npub1operator")
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_oracle_service_convenience():
    """Module-level resolve_oracle_service() creates a one-shot registry."""
    from unittest.mock import MagicMock as SyncMock

    import httpx

    from tollbooth.registry import resolve_oracle_service

    members = [
        {
            "npub": "npub1operator",
            "role": "operator",
            "status": "active",
            "upstream_authority_npub": "npub1prime",
        },
        {
            "npub": "npub1prime",
            "role": "prime_authority",
            "status": "active",
            "upstream_authority_npub": None,
            "services": [
                {
                    "name": "dpyc-oracle",
                    "url": "https://dpyc-oracle.fastmcp.app/mcp",
                }
            ],
        },
    ]

    resp = SyncMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {"members": members}
    resp.raise_for_status = SyncMock()

    with patch("tollbooth.registry.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=resp)
        MockClient.return_value = instance

        result = await resolve_oracle_service(
            "npub1operator",
            registry_url="https://example.com/members.json",
        )
        assert result["url"] == "https://dpyc-oracle.fastmcp.app/mcp"
        instance.aclose.assert_awaited_once()
