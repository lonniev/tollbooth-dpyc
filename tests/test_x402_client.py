"""Tests for tollbooth.x402_client — transparent x402 payment adapter."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest


# Skip entire module if optional deps not installed
pytest.importorskip("x402")
pytest.importorskip("eth_account")

from tollbooth.x402_client import X402Client, _parse_chain_id, x402_wallet_template


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Deterministic test key (DO NOT use in production — this is a well-known test key)
_TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
_TEST_ADDRESS = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

_PAYMENT_REQUIRED = {
    "x402Version": 2,
    "error": "Payment required",
    "resource": {"url": "https://api.example.com/data", "description": "test"},
    "accepts": [
        {
            "scheme": "exact",
            "network": "eip155:8453",
            "asset": "USDC",
            "amount": "100000",
            "payTo": "0x1234567890abcdef1234567890abcdef12345678",
            "maxTimeoutSeconds": 300,
            "extra": {},
        }
    ],
}

_PAYMENT_REQUIRED_HEADER = base64.b64encode(
    json.dumps(_PAYMENT_REQUIRED).encode()
).decode()


def _mock_response(status: int, headers: dict | None = None) -> httpx.Response:
    """Build a mock httpx.Response."""
    return httpx.Response(
        status_code=status,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.example.com/data"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPassthrough:
    """Non-402 responses pass through untouched."""

    @pytest.mark.asyncio
    async def test_200_passthrough(self):
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.return_value = _mock_response(200)
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 200
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_500_passthrough(self):
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.return_value = _mock_response(500)
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 500
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_400_passthrough(self):
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.return_value = _mock_response(400)
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 400
            mock.assert_called_once()


class TestX402PaymentDance:
    """402 with payment-required header triggers sign-and-retry."""

    @pytest.mark.asyncio
    async def test_402_payment_and_retry(self):
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        resp_402 = _mock_response(402, {"payment-required": _PAYMENT_REQUIRED_HEADER})
        resp_200 = _mock_response(200)

        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.side_effect = [resp_402, resp_200]
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 200
            assert mock.call_count == 2
            # Second call should include X-PAYMENT header
            retry_call = mock.call_args_list[1]
            assert "X-PAYMENT" in retry_call.kwargs.get("headers", {})

    @pytest.mark.asyncio
    async def test_402_retry_still_fails(self):
        """Upstream returns 500 after payment — returned as-is, no retry loop."""
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        resp_402 = _mock_response(402, {"payment-required": _PAYMENT_REQUIRED_HEADER})
        resp_500 = _mock_response(500)

        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.side_effect = [resp_402, resp_500]
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 500
            assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_missing_payment_header(self):
        """402 without payment-required header — return as-is (not x402)."""
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.return_value = _mock_response(402)  # no header
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 402
            mock.assert_called_once()  # no retry

    @pytest.mark.asyncio
    async def test_max_retries_respected(self):
        """Prevent infinite 402 loops — stops after max_retries."""
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS, max_retries=2)
        resp_402 = _mock_response(402, {"payment-required": _PAYMENT_REQUIRED_HEADER})

        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.return_value = resp_402  # always 402
            resp = await client.get("https://api.example.com/data")
            assert resp.status_code == 402
            # 1 initial + 2 retries = 3 total
            assert mock.call_count == 3


class TestPaymentSigning:
    """Verify the payment payload is correctly structured."""

    @pytest.mark.asyncio
    async def test_payment_payload_structure(self):
        client = X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS)
        resp_402 = _mock_response(402, {"payment-required": _PAYMENT_REQUIRED_HEADER})
        resp_200 = _mock_response(200)

        with patch.object(client._http, "request", new_callable=AsyncMock) as mock:
            mock.side_effect = [resp_402, resp_200]
            await client.get("https://api.example.com/data")

            # Decode the payment payload from the retry
            retry_headers = mock.call_args_list[1].kwargs.get("headers", {})
            raw = base64.b64decode(retry_headers["X-PAYMENT"])
            payload = json.loads(raw)

            assert payload["x402Version"] == 2
            assert "payload" in payload
            assert "signature" in payload["payload"]
            assert "authorization" in payload["payload"]
            auth = payload["payload"]["authorization"]
            assert auth["from"] == _TEST_ADDRESS
            assert auth["to"] == _PAYMENT_REQUIRED["accepts"][0]["payTo"]


class TestChainIdParsing:
    """CAIP-2 network identifier parsing."""

    def test_caip2_base(self):
        assert _parse_chain_id("eip155:8453") == 8453

    def test_caip2_ethereum(self):
        assert _parse_chain_id("eip155:1") == 1

    def test_named_base(self):
        assert _parse_chain_id("base") == 8453

    def test_named_ethereum(self):
        assert _parse_chain_id("ethereum") == 1

    def test_unknown_defaults_to_base(self):
        assert _parse_chain_id("unknown") == 8453


class TestCredentialTemplate:
    """Credential template for Secure Courier delivery."""

    def test_template_structure(self):
        tpl = x402_wallet_template()
        assert tpl.service == "x402-wallet"
        assert tpl.version == 1
        assert "wallet_private_key" in tpl.fields
        assert "wallet_address" in tpl.fields
        assert "facilitator_url" in tpl.fields
        assert tpl.fields["wallet_private_key"].sensitive is True
        assert tpl.fields["wallet_address"].sensitive is False


class TestContextManager:
    """Async context manager protocol."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with X402Client(_TEST_PRIVATE_KEY, _TEST_ADDRESS) as client:
            assert client is not None
