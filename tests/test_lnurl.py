"""Tests for LNURL-pay Lightning address resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tollbooth.lnurl import (
    LnurlResolutionError,
    _parse_lightning_address,
    resolve_lightning_address,
)


# ---------------------------------------------------------------------------
# _parse_lightning_address
# ---------------------------------------------------------------------------


class TestParseLightningAddress:
    def test_valid(self) -> None:
        user, domain = _parse_lightning_address("alice@example.com")
        assert user == "alice"
        assert domain == "example.com"

    def test_missing_at(self) -> None:
        with pytest.raises(LnurlResolutionError, match="missing @"):
            _parse_lightning_address("alice.example.com")

    def test_multiple_at(self) -> None:
        with pytest.raises(LnurlResolutionError, match="multiple @"):
            _parse_lightning_address("alice@bob@example.com")

    def test_empty_user(self) -> None:
        with pytest.raises(LnurlResolutionError, match="empty user"):
            _parse_lightning_address("@example.com")

    def test_empty_domain(self) -> None:
        with pytest.raises(LnurlResolutionError, match="empty domain"):
            _parse_lightning_address("alice@")


# ---------------------------------------------------------------------------
# Helpers — mock httpx responses
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


LNURL_META = {
    "callback": "https://example.com/lnurlp/alice/callback",
    "minSendable": 1000,       # 1 sat in msats
    "maxSendable": 100000000,  # 100,000 sats in msats
    "tag": "payRequest",
}

BOLT11_INVOICE = "lnbc200n1pjexamplefakeinvoice"

CALLBACK_RESPONSE = {"pr": BOLT11_INVOICE}


# ---------------------------------------------------------------------------
# resolve_lightning_address — happy path
# ---------------------------------------------------------------------------


class TestResolveLightningAddress:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        """Full round-trip: well-known → callback → BOLT11 invoice."""
        responses = [
            _mock_response(200, LNURL_META),
            _mock_response(200, CALLBACK_RESPONSE),
        ]

        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(side_effect=responses)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            result = await resolve_lightning_address("alice@example.com", 20)

        assert result == BOLT11_INVOICE
        # First call: well-known URL
        first_call = client_instance.get.call_args_list[0]
        assert first_call[0][0] == "https://example.com/.well-known/lnurlp/alice"
        # Second call: callback with amount in msats
        second_call = client_instance.get.call_args_list[1]
        assert second_call[0][0] == LNURL_META["callback"]
        assert second_call[1]["params"]["amount"] == 20000  # 20 sats * 1000

    @pytest.mark.asyncio
    async def test_with_comment(self) -> None:
        """Comment is passed to the callback."""
        responses = [
            _mock_response(200, LNURL_META),
            _mock_response(200, CALLBACK_RESPONSE),
        ]

        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(side_effect=responses)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            result = await resolve_lightning_address(
                "alice@example.com", 20, comment="royalty payout",
            )

        assert result == BOLT11_INVOICE
        second_call = client_instance.get.call_args_list[1]
        assert second_call[1]["params"]["comment"] == "royalty payout"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestResolveLightningAddressErrors:
    @pytest.mark.asyncio
    async def test_invalid_address_format(self) -> None:
        with pytest.raises(LnurlResolutionError, match="missing @"):
            await resolve_lightning_address("no-at-sign", 100)

    @pytest.mark.asyncio
    async def test_well_known_http_error(self) -> None:
        """Non-200 from well-known endpoint."""
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(
                return_value=_mock_response(404, {})
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="HTTP 404"):
                await resolve_lightning_address("alice@example.com", 100)

    @pytest.mark.asyncio
    async def test_well_known_error_status(self) -> None:
        """LNURL endpoint returns status=ERROR."""
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(
                return_value=_mock_response(200, {
                    "status": "ERROR", "reason": "User not found",
                })
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="User not found"):
                await resolve_lightning_address("alice@example.com", 100)

    @pytest.mark.asyncio
    async def test_missing_callback(self) -> None:
        """LNURL response missing callback field."""
        meta_no_callback = {
            "minSendable": 1000,
            "maxSendable": 100000000,
        }
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(
                return_value=_mock_response(200, meta_no_callback)
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="missing 'callback'"):
                await resolve_lightning_address("alice@example.com", 100)

    @pytest.mark.asyncio
    async def test_amount_below_min(self) -> None:
        """Amount below minSendable."""
        meta = {**LNURL_META, "minSendable": 50000}  # 50 sats min
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(
                return_value=_mock_response(200, meta)
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="below minimum"):
                await resolve_lightning_address("alice@example.com", 10)

    @pytest.mark.asyncio
    async def test_amount_above_max(self) -> None:
        """Amount above maxSendable."""
        meta = {**LNURL_META, "maxSendable": 10000}  # 10 sats max
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(
                return_value=_mock_response(200, meta)
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="above maximum"):
                await resolve_lightning_address("alice@example.com", 100)

    @pytest.mark.asyncio
    async def test_callback_http_error(self) -> None:
        """Non-200 from callback endpoint."""
        responses = [
            _mock_response(200, LNURL_META),
            _mock_response(500, {}),
        ]
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(side_effect=responses)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="HTTP 500"):
                await resolve_lightning_address("alice@example.com", 20)

    @pytest.mark.asyncio
    async def test_callback_error_status(self) -> None:
        """Callback returns status=ERROR."""
        responses = [
            _mock_response(200, LNURL_META),
            _mock_response(200, {"status": "ERROR", "reason": "Invoice creation failed"}),
        ]
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(side_effect=responses)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="Invoice creation failed"):
                await resolve_lightning_address("alice@example.com", 20)

    @pytest.mark.asyncio
    async def test_callback_missing_pr(self) -> None:
        """Callback response missing 'pr' field."""
        responses = [
            _mock_response(200, LNURL_META),
            _mock_response(200, {"routes": []}),
        ]
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(side_effect=responses)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="missing 'pr'"):
                await resolve_lightning_address("alice@example.com", 20)

    @pytest.mark.asyncio
    async def test_network_error(self) -> None:
        """httpx network failure wraps into LnurlResolutionError."""
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(
                side_effect=httpx.ConnectError("DNS resolution failed")
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            with pytest.raises(LnurlResolutionError, match="Network error"):
                await resolve_lightning_address("alice@example.com", 20)

    @pytest.mark.asyncio
    async def test_max_sendable_zero_means_unlimited(self) -> None:
        """maxSendable=0 is treated as unlimited (no upper bound check)."""
        meta = {**LNURL_META, "maxSendable": 0}
        responses = [
            _mock_response(200, meta),
            _mock_response(200, CALLBACK_RESPONSE),
        ]
        with patch("tollbooth.lnurl.httpx.AsyncClient") as mock_client_cls:
            client_instance = AsyncMock()
            client_instance.get = AsyncMock(side_effect=responses)
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = client_instance

            result = await resolve_lightning_address("alice@example.com", 999999)
        assert result == BOLT11_INVOICE
