"""Tests for tollbooth.authority_client — AuthorityCertifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.authority_client import AuthorityCertifier, AuthorityCertifyError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_block(data: dict) -> MagicMock:
    """Create a mock TextContent block with .text = JSON string."""
    block = MagicMock()
    block.text = json.dumps(data)
    return block


# ---------------------------------------------------------------------------
# certify() — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_certify_success():
    """certify() returns parsed certificate dict on success."""
    cert_response = {
        "success": True,
        "certificate": "eyJhbGciOi...",
        "jti": "abc-123",
        "amount_sats": 100,
        "fee_sats": 10,
        "net_sats": 90,
        "expires_at": "2026-03-03T12:00:00Z",
    }

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(cert_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        result = await certifier.certify_credits(100)

    assert result["certificate"] == "eyJhbGciOi..."
    assert result["net_sats"] == 90
    mock_client.call_tool.assert_awaited_once_with(
        "authority_certify_credits",
        {"operator_id": "npub1operator", "amount_sats": 100},
    )


# ---------------------------------------------------------------------------
# certify() — Authority refuses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_certify_authority_refuses():
    """certify() raises AuthorityCertifyError when Authority returns error."""
    error_response = {
        "success": False,
        "error": "Insufficient credit balance",
    }

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(error_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        with pytest.raises(AuthorityCertifyError, match="Insufficient credit balance"):
            await certifier.certify_credits(100)


# ---------------------------------------------------------------------------
# certify() — connection failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_certify_connection_failure():
    """certify() wraps connection errors in AuthorityCertifyError."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        with pytest.raises(AuthorityCertifyError, match="Failed to connect"):
            await certifier.certify_credits(100)


# ---------------------------------------------------------------------------
# certify() — custom tool name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_certify_custom_tool_name():
    """certify() uses the custom tool name when provided."""
    cert_response = {
        "success": True,
        "certificate": "jwt...",
        "jti": "x",
        "amount_sats": 50,
        "fee_sats": 5,
        "net_sats": 45,
        "expires_at": "2026-03-03T12:00:00Z",
    }

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(cert_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
            certify_tool_name="custom_certify",
        )
        await certifier.certify_credits(50)

    mock_client.call_tool.assert_awaited_once_with(
        "custom_certify",
        {"operator_id": "npub1operator", "amount_sats": 50},
    )


# ---------------------------------------------------------------------------
# certify() — CallToolResult unwrapping (structured data path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_certify_calltoolresult_data():
    """certify() unwraps CallToolResult-like objects with .data dict."""
    cert_response = {
        "success": True,
        "certificate": "eyJhbGciOi...",
        "jti": "ctr-456",
        "amount_sats": 100,
        "fee_sats": 10,
        "net_sats": 90,
        "expires_at": "2026-03-03T12:00:00Z",
    }

    # Simulate CallToolResult with .data attribute (structured output)
    call_tool_result = MagicMock()
    call_tool_result.data = cert_response
    call_tool_result.content = None  # data takes priority

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=call_tool_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        result = await certifier.certify_credits(100)

    assert result["certificate"] == "eyJhbGciOi..."
    assert result["jti"] == "ctr-456"
    assert result["net_sats"] == 90


@pytest.mark.asyncio
async def test_certify_calltoolresult_content():
    """certify() unwraps CallToolResult-like objects with .content list."""
    cert_response = {
        "success": True,
        "certificate": "eyJhbGciOi...",
        "jti": "ctr-789",
        "amount_sats": 50,
        "fee_sats": 5,
        "net_sats": 45,
        "expires_at": "2026-03-03T12:00:00Z",
    }

    # Simulate CallToolResult with .content attribute (text blocks)
    call_tool_result = MagicMock(spec=[])  # no .data attribute
    call_tool_result.content = [_text_block(cert_response)]

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=call_tool_result)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        result = await certifier.certify_credits(50)

    assert result["certificate"] == "eyJhbGciOi..."
    assert result["jti"] == "ctr-789"
    assert result["net_sats"] == 45


# ---------------------------------------------------------------------------
# certify() — unexpected response format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_certify_unexpected_format():
    """certify() raises on unexpected response (no certificate key)."""
    bad_response = {"success": True, "message": "ok but no cert"}

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(bad_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        with pytest.raises(AuthorityCertifyError, match="unexpected response"):
            await certifier.certify_credits(100)


# ---------------------------------------------------------------------------
# resolve_authority_service — registry integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_authority_service_success():
    """resolve_authority_service returns authority URL from registry."""
    from unittest.mock import MagicMock as SyncMock

    import httpx

    from tollbooth.registry import DPYCRegistry, RegistryError

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
            "services": [
                {
                    "name": "tollbooth-authority",
                    "url": "https://authority.fastmcp.app/mcp",
                    "description": "Authority service",
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

    result = await registry.resolve_authority_service("npub1operator")
    assert result["npub"] == "npub1authority"
    assert result["url"] == "https://authority.fastmcp.app/mcp"
    assert result["name"] == "tollbooth-authority"
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_authority_service_no_services():
    """resolve_authority_service raises when authority has no services."""
    from unittest.mock import MagicMock as SyncMock

    import httpx

    from tollbooth.registry import DPYCRegistry, RegistryError

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

    with pytest.raises(RegistryError, match="no services"):
        await registry.resolve_authority_service("npub1operator")
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_authority_service_convenience():
    """Module-level resolve_authority_service() creates a one-shot registry."""
    from unittest.mock import MagicMock as SyncMock

    import httpx

    from tollbooth.registry import resolve_authority_service

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
            "services": [
                {
                    "name": "tollbooth-authority",
                    "url": "https://authority.fastmcp.app/mcp",
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

        result = await resolve_authority_service(
            "npub1operator",
            registry_url="https://example.com/members.json",
        )
        assert result["url"] == "https://authority.fastmcp.app/mcp"
        instance.aclose.assert_awaited_once()
