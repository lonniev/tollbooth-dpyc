"""Tests for tollbooth.registry — DPYC community registry resolution."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tollbooth.registry import (
    DEFAULT_REGISTRY_URL,
    DPYCRegistry,
    RegistryError,
    resolve_authority_npub,
    resolve_service_by_name,
)

SAMPLE_MEMBERS = [
    {
        "npub": "npub1prime",
        "role": "prime_authority",
        "status": "active",
        "upstream_authority_npub": None,
        "services": [
            {"name": "dpyc-oracle", "url": "https://oracle.example.com/mcp", "description": "Oracle"},
        ],
    },
    {
        "npub": "npub1authority",
        "role": "authority",
        "status": "active",
        "upstream_authority_npub": "npub1prime",
        "services": [
            {"name": "tollbooth-authority", "url": "https://authority.example.com/mcp", "description": "Authority"},
        ],
    },
    {
        "npub": "npub1operator",
        "role": "operator",
        "status": "active",
        "upstream_authority_npub": "npub1authority",
        "services": [
            {"name": "my-mcp", "url": "https://my-mcp.example.com/mcp", "description": "My MCP"},
        ],
    },
    {
        "npub": "npub1banned",
        "role": "operator",
        "status": "banned",
        "upstream_authority_npub": "npub1authority",
        "services": [
            {"name": "banned-svc", "url": "https://banned.example.com", "description": "Banned"},
        ],
    },
    {
        "npub": "npub1advocate",
        "role": "advocate",
        "status": "active",
        "upstream_authority_npub": "npub1prime",
        "services": [
            {
                "name": "tollbooth-oauth2-collector",
                "url": "https://collector.example.com",
                "description": "OAuth2 callback mailbox",
            },
        ],
    },
]


def _mock_response(data, status_code=200):
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# --- resolve_authority_npub (instance method) ---


@pytest.mark.asyncio
async def test_resolve_authority_npub_success():
    """resolve_authority_npub returns upstream npub for an active operator."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    result = await registry.resolve_authority_npub("npub1operator")
    assert result == "npub1authority"
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_authority_npub_no_upstream_raises():
    """resolve_authority_npub raises for Prime Authority (no upstream)."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    with pytest.raises(RegistryError, match="no upstream authority"):
        await registry.resolve_authority_npub("npub1prime")
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_authority_npub_not_found_raises():
    """resolve_authority_npub raises for unknown npub."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    with pytest.raises(RegistryError, match="not found"):
        await registry.resolve_authority_npub("npub1unknown")
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_authority_npub_inactive_raises():
    """resolve_authority_npub raises for banned member."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    with pytest.raises(RegistryError, match="not active"):
        await registry.resolve_authority_npub("npub1banned")
    await registry.close()


# --- Cache TTL ---


@pytest.mark.asyncio
async def test_cache_ttl_respected():
    """Registry re-fetches after TTL expires."""
    registry = DPYCRegistry(url="https://example.com/members.json", cache_ttl_seconds=1)
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    # First call fetches
    await registry.check_membership("npub1operator")
    assert registry._client.get.call_count == 1

    # Second call uses cache
    await registry.check_membership("npub1operator")
    assert registry._client.get.call_count == 1

    # Force cache to expire by backdating cache_time
    registry._cache_time = time.monotonic() - 2.0

    # Third call re-fetches
    await registry.check_membership("npub1operator")
    assert registry._client.get.call_count == 2
    await registry.close()


# --- HTTP error ---


@pytest.mark.asyncio
async def test_http_error_raises_registry_error():
    """HTTP errors are wrapped in RegistryError."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({}, status_code=500)
    )

    with pytest.raises(RegistryError, match="fetch failed"):
        await registry.check_membership("npub1operator")
    await registry.close()


# --- Bare list format ---


@pytest.mark.asyncio
async def test_bare_list_format():
    """Registry accepts bare list format (no wrapper object)."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response(SAMPLE_MEMBERS)  # bare list, no wrapper
    )

    result = await registry.check_membership("npub1operator")
    assert result["npub"] == "npub1operator"
    await registry.close()


# --- invalidate_cache ---


@pytest.mark.asyncio
async def test_invalidate_cache():
    """invalidate_cache forces next lookup to re-fetch."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    await registry.check_membership("npub1operator")
    assert registry._client.get.call_count == 1

    registry.invalidate_cache()

    await registry.check_membership("npub1operator")
    assert registry._client.get.call_count == 2
    await registry.close()


# --- Convenience function ---


@pytest.mark.asyncio
async def test_convenience_function():
    """Module-level resolve_authority_npub() creates a one-shot registry."""
    mock_resp = _mock_response({"members": SAMPLE_MEMBERS})

    with patch("tollbooth.registry.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_resp)
        MockClient.return_value = instance

        result = await resolve_authority_npub(
            "npub1operator",
            registry_url="https://example.com/members.json",
        )
        assert result == "npub1authority"
        instance.aclose.assert_awaited_once()


# --- Default URL ---


def test_default_registry_url():
    """DEFAULT_REGISTRY_URL points to dpyc-community on GitHub."""
    assert "dpyc-community" in DEFAULT_REGISTRY_URL
    assert "read-only-lookup-cache.json" in DEFAULT_REGISTRY_URL


# --- resolve_service_by_name (instance method) ---


@pytest.mark.asyncio
async def test_resolve_service_by_name_found():
    """resolve_service_by_name returns correct dict for a matching service."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    result = await registry.resolve_service_by_name("tollbooth-oauth2-collector")
    assert result == {
        "npub": "npub1advocate",
        "url": "https://collector.example.com",
        "name": "tollbooth-oauth2-collector",
    }
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_service_by_name_not_found():
    """resolve_service_by_name raises RegistryError when no match."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    with pytest.raises(RegistryError, match="No active member with service"):
        await registry.resolve_service_by_name("nonexistent-service")
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_service_by_name_skips_inactive():
    """resolve_service_by_name skips banned members."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    with pytest.raises(RegistryError, match="No active member with service"):
        await registry.resolve_service_by_name("banned-svc")
    await registry.close()


@pytest.mark.asyncio
async def test_resolve_service_by_name_works_across_roles():
    """resolve_service_by_name finds services on any role (operator, authority, etc)."""
    registry = DPYCRegistry(url="https://example.com/members.json")
    registry._client = AsyncMock()
    registry._client.get = AsyncMock(
        return_value=_mock_response({"members": SAMPLE_MEMBERS})
    )

    # Operator service
    result = await registry.resolve_service_by_name("my-mcp")
    assert result["npub"] == "npub1operator"

    # Authority service
    result = await registry.resolve_service_by_name("tollbooth-authority")
    assert result["npub"] == "npub1authority"

    # Prime Authority service
    result = await registry.resolve_service_by_name("dpyc-oracle")
    assert result["npub"] == "npub1prime"

    await registry.close()


# --- resolve_service_by_name (convenience function) ---


@pytest.mark.asyncio
async def test_resolve_service_by_name_convenience():
    """Module-level resolve_service_by_name() creates a one-shot registry."""
    mock_resp = _mock_response({"members": SAMPLE_MEMBERS})

    with patch("tollbooth.registry.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.get = AsyncMock(return_value=mock_resp)
        MockClient.return_value = instance

        result = await resolve_service_by_name(
            "tollbooth-oauth2-collector",
            registry_url="https://example.com/members.json",
        )
        assert result["npub"] == "npub1advocate"
        assert result["url"] == "https://collector.example.com"
        instance.aclose.assert_awaited_once()
