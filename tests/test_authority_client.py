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
        {"npub": "npub1operator", "amount_sats": 100, "dpop_token": ""},
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


@pytest.mark.asyncio
async def test_certify_propagates_error_code():
    """The Authority's structured error_code rides on the exception so callers
    can branch on it (e.g. surface a kind 'Authority is broke' situation)."""
    error_response = {
        "success": False,
        "error_code": "insufficient_balance",
        "error": "Insufficient balance: 0 sats available, 20 required for "
                 "authority_certify_credits.",
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
        with pytest.raises(AuthorityCertifyError) as exc_info:
            await certifier.certify_credits(100)

    assert exc_info.value.error_code == "insufficient_balance"


@pytest.mark.asyncio
async def test_certify_error_code_empty_when_absent():
    """A refusal without an error_code yields an empty string, not a crash."""
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(
        return_value=[_text_block({"success": False, "error": "nope"})]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        with pytest.raises(AuthorityCertifyError) as exc_info:
            await certifier.certify_credits(100)

    assert exc_info.value.error_code == ""


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
        {"npub": "npub1operator", "amount_sats": 50, "dpop_token": ""},
    )


@pytest.mark.asyncio
async def test_certify_sends_dpop_token_not_proof():
    """Regression: the certify_credits call must send the operator's identity
    token under the ``dpop_token`` kwarg (the wheel-0.57.0 rename), never the old
    ``proof``. Sending ``proof`` makes the Authority's pydantic-typed tool reject
    the call with 'unexpected keyword argument: proof', breaking every patron
    credit purchase."""
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(
        return_value=[_text_block({"success": True, "certificate": "c"})]
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        await certifier.certify_credits(100)

    sent_args = mock_client.call_tool.await_args.args[1]
    assert "dpop_token" in sent_args, "certify must send the token as dpop_token"
    assert "proof" not in sent_args, "the pre-0.57.0 'proof' kwarg must not return"


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
# check_balance() — wire-name proof binding (#188)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_balance_signs_for_wire_tool_name():
    """check_balance must sign the proof for the exact wire tool name it calls.

    Regression for #188 / cypher-mcp#70: historically authenticate("check_balance")
    while calling "authority_check_balance", so the Authority rejected every
    proof with tool_mismatch. Mirror certify_credits: sign for the wire name.
    """
    balance_response = {
        "success": True,
        "balance_sats": 500,
        "npub": "npub1operator",
    }

    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block(balance_response)])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    mock_signer = MagicMock()
    mock_signer.authenticate.return_value = {
        "npub": "npub1operator",
        "dpop_token": "",
    }

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        certifier._signer = mock_signer
        result = await certifier.check_balance()

    assert result["success"] is True
    assert result["balance_sats"] == 500
    mock_client.call_tool.assert_awaited_once_with(
        "authority_check_balance",
        {"npub": "npub1operator", "dpop_token": ""},
    )
    mock_signer.authenticate.assert_called_once_with("authority_check_balance")
