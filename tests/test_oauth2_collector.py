"""Tests for tollbooth.oauth2_collector — generic OAuth2 Authorization Code flow helpers."""

import base64
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.oauth2_collector import (
    OAuthCollectorError,
    begin_oauth_flow,
    build_authorize_url,
    decrypt_collector_code,
    exchange_code_for_token,
    retrieve_code_from_collector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_AUTHORIZE = "https://provider.example.com/oauth/authorize"
_TEST_TOKEN = "https://provider.example.com/oauth/token"


def _fake_encrypt(code: str, state: str) -> str:
    """Encrypt a code the same way the collector does (XOR + base64)."""
    key = hashlib.sha256(state.encode()).digest()
    code_bytes = code.encode()
    encrypted = bytes(c ^ key[i % 32] for i, c in enumerate(code_bytes))
    return base64.urlsafe_b64encode(encrypted).decode()


# ---------------------------------------------------------------------------
# build_authorize_url tests
# ---------------------------------------------------------------------------


class TestBuildAuthorizeUrl:
    """Tests for build_authorize_url."""

    def test_constructs_url_with_required_params(self):
        url = build_authorize_url(_TEST_AUTHORIZE, "my-key", "https://cb.example.com", "state123")
        assert url.startswith(_TEST_AUTHORIZE + "?")
        assert "response_type=code" in url
        assert "client_id=my-key" in url
        assert "redirect_uri=" in url
        assert "state=state123" in url
        assert "response_mode" not in url

    def test_includes_scope_when_provided(self):
        url = build_authorize_url(
            _TEST_AUTHORIZE, "key", "https://cb.example.com", "st", scope="readonly"
        )
        assert "scope=readonly" in url

    def test_omits_scope_when_none(self):
        url = build_authorize_url(_TEST_AUTHORIZE, "key", "https://cb.example.com", "st")
        assert "scope=" not in url

    def test_includes_extra_params(self):
        url = build_authorize_url(
            _TEST_AUTHORIZE,
            "key",
            "https://cb.example.com",
            "st",
            extra_params={"audience": "api.example.com"},
        )
        assert "audience=api.example.com" in url

    def test_npub_as_state(self):
        url = build_authorize_url(_TEST_AUTHORIZE, "key", "https://cb.example.com", "npub1abc123")
        assert "state=npub1abc123" in url


# ---------------------------------------------------------------------------
# begin_oauth_flow tests
# ---------------------------------------------------------------------------


class TestBeginOAuthFlow:
    """Tests for begin_oauth_flow."""

    def test_returns_pending_with_url(self):
        result = begin_oauth_flow(
            patron_npub="npub1abc",
            client_id="my-app-key",
            redirect_uri="https://collector.example.com/oauth/callback",
            authorize_endpoint=_TEST_AUTHORIZE,
            scope="readonly",
        )
        assert result["status"] == "pending"
        assert "authorize_url" in result
        assert _TEST_AUTHORIZE in result["authorize_url"]
        assert "state=npub1abc" in result["authorize_url"]

    def test_uses_provider_name_in_message(self):
        result = begin_oauth_flow(
            "npub1x", "key", "https://cb.example.com", _TEST_AUTHORIZE, provider_name="Schwab"
        )
        assert "Schwab" in result["message"]

    def test_default_provider_name(self):
        result = begin_oauth_flow("npub1x", "key", "https://cb.example.com", _TEST_AUTHORIZE)
        assert "the provider" in result["message"]

    def test_idempotent(self):
        r1 = begin_oauth_flow("npub1same", "key", "https://cb.example.com", _TEST_AUTHORIZE)
        r2 = begin_oauth_flow("npub1same", "key", "https://cb.example.com", _TEST_AUTHORIZE)
        assert r1["authorize_url"] == r2["authorize_url"]


# ---------------------------------------------------------------------------
# exchange_code_for_token tests
# ---------------------------------------------------------------------------


class TestExchangeCodeForToken:
    """Tests for exchange_code_for_token."""

    @pytest.mark.asyncio
    async def test_exchanges_code(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "at-123",
            "refresh_token": "rt-456",
            "expires_in": 1800,
            "token_type": "Bearer",
        }
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            token = await exchange_code_for_token(
                code="auth-code-xyz",
                client_id="app-key",
                client_secret="app-secret",
                redirect_uri="https://example.com/cb",
                token_endpoint=_TEST_TOKEN,
            )

        assert token["access_token"] == "at-123"
        assert token["refresh_token"] == "rt-456"
        assert "expires_at" in token
        assert token["expires_at"] > time.time()

        # Verify it posted to the correct endpoint
        call_args = mock_http.post.call_args
        assert call_args[0][0] == _TEST_TOKEN

        # Verify Basic auth header
        headers = call_args.kwargs.get("headers", call_args[1].get("headers", {}))
        assert "Basic" in headers.get("Authorization", "")

    @pytest.mark.asyncio
    async def test_default_expires_at_when_missing(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"access_token": "at"}
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        before = time.time()
        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            token = await exchange_code_for_token(
                "code", "id", "secret", "https://cb", _TEST_TOKEN
            )

        # Default expires_in is 1800
        assert token["expires_at"] >= before + 1800


# ---------------------------------------------------------------------------
# decrypt_collector_code tests
# ---------------------------------------------------------------------------


class TestDecryptCollectorCode:
    """Tests for decrypt_collector_code (XOR with SHA-256 keystream)."""

    def test_roundtrip(self):
        encrypted = _fake_encrypt("my-secret-code", "my-state-token")
        assert decrypt_collector_code(encrypted, "my-state-token") == "my-secret-code"

    def test_npub_as_state_roundtrip(self):
        npub = "npub1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3aael2"
        code = "authorization-code-from-provider"
        encrypted = _fake_encrypt(code, npub)
        assert decrypt_collector_code(encrypted, npub) == code

    def test_wrong_state_raises_or_differs(self):
        encrypted = _fake_encrypt("my-secret-code", "correct-state")
        with pytest.raises(OAuthCollectorError):
            decrypt_collector_code(encrypted, "wrong-state")

    def test_empty_code_roundtrip(self):
        encrypted = _fake_encrypt("", "state")
        assert decrypt_collector_code(encrypted, "state") == ""


# ---------------------------------------------------------------------------
# retrieve_code_from_collector tests
# ---------------------------------------------------------------------------


class TestRetrieveCodeFromCollector:
    """Tests for retrieve_code_from_collector."""

    @pytest.mark.asyncio
    async def test_returns_decrypted_code_on_success(self):
        state = "npub1abc123"
        encrypted = _fake_encrypt("auth-code-abc", state)

        sse_body = (
            "event: message\n"
            "data: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "structuredContent": {"found": True, "code": encrypted}
                    },
                }
            )
            + "\n\n"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sse_body

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            result = await retrieve_code_from_collector(
                "https://collector.example.com", state
            )

        assert result == "auth-code-abc"
        call_args = mock_http.post.call_args
        assert call_args[0][0] == "https://collector.example.com/mcp/"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        sse_body = (
            "event: message\n"
            "data: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "structuredContent": {
                            "found": False,
                            "error": "not found or expired",
                        }
                    },
                }
            )
            + "\n\n"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sse_body

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            result = await retrieve_code_from_collector(
                "https://collector.example.com", "npub1abc"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_strips_trailing_slash(self):
        state = "npub1xyz"
        encrypted = _fake_encrypt("xyz", state)

        sse_body = (
            "event: message\n"
            "data: "
            + json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "structuredContent": {"found": True, "code": encrypted}
                    },
                }
            )
            + "\n\n"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sse_body

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            result = await retrieve_code_from_collector(
                "https://collector.example.com/", state
            )

        assert result == "xyz"
        call_args = mock_http.post.call_args
        assert call_args[0][0] == "https://collector.example.com/mcp/"
