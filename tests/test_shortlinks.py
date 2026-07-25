"""Tests for shortlinks.py — tollbooth-shortlinks MCP utility."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tollbooth.shortlinks import create_shortlink

FAKE_SHORT_URL = "https://tb.link/abc123"


def _sse_response(short_url: str, *, status_code: int = 200) -> httpx.Response:
    """Build a fake SSE-style httpx.Response."""
    import json

    sse_payload = json.dumps({
        "result": {
            "structuredContent": {
                "success": True,
                "short_url": short_url,
            }
        }
    })
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = f"data: {sse_payload}\n"
    return resp


@pytest.mark.asyncio
async def test_create_shortlink_happy_path():
    """Happy path: SSE response with short_url."""
    resp = _sse_response(FAKE_SHORT_URL)

    mock_client = AsyncMock()
    mock_client.post.return_value = resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("tollbooth.shortlinks.httpx.AsyncClient", return_value=mock_client),
        patch("tollbooth.shortlinks._resolve_shortlinks_url", return_value="https://example.com/mcp/"),
    ):
        result = await create_shortlink("https://example.com/long-url")

    assert result == FAKE_SHORT_URL
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[1]["json"]["params"]["arguments"]["url"] == "https://example.com/long-url"


@pytest.mark.asyncio
async def test_create_shortlink_with_slug():
    """Slug is passed through in the arguments."""
    resp = _sse_response(FAKE_SHORT_URL)

    mock_client = AsyncMock()
    mock_client.post.return_value = resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("tollbooth.shortlinks.httpx.AsyncClient", return_value=mock_client),
        patch("tollbooth.shortlinks._resolve_shortlinks_url", return_value="https://example.com/mcp/"),
    ):
        result = await create_shortlink("https://example.com/long-url", slug="my-slug")

    assert result == FAKE_SHORT_URL
    args = mock_client.post.call_args[1]["json"]["params"]["arguments"]
    assert args["slug"] == "my-slug"


@pytest.mark.asyncio
async def test_create_shortlink_http_error():
    """Non-200 status returns None."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 500
    resp.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.post.return_value = resp
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("tollbooth.shortlinks.httpx.AsyncClient", return_value=mock_client),
        patch("tollbooth.shortlinks._resolve_shortlinks_url", return_value="https://example.com/mcp/"),
    ):
        result = await create_shortlink("https://example.com/long-url")

    assert result is None


@pytest.mark.asyncio
async def test_create_shortlink_network_error():
    """Network exception returns None (non-fatal)."""
    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.ConnectError("connection refused")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("tollbooth.shortlinks.httpx.AsyncClient", return_value=mock_client),
        patch("tollbooth.shortlinks._resolve_shortlinks_url", return_value="https://example.com/mcp/"),
    ):
        result = await create_shortlink("https://example.com/long-url")

    assert result is None
