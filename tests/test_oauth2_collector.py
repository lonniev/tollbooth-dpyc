"""Tests for tollbooth.oauth2_collector — generic OAuth2 Authorization Code flow helpers."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.oauth2_collector import (
    OAuthCollectorError,
    begin_oauth_flow,
    build_authorize_url,
    decrypt_collector_code,
    encrypt_collector_code,
    exchange_code_for_token,
    retrieve_code_from_collector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_AUTHORIZE = "https://provider.example.com/oauth/authorize"
_TEST_TOKEN = "https://provider.example.com/oauth/token"

# Fixed operator keypair for collector-seal tests. Public half is what the
# collector encrypts TO; private half is what the MCP opens the seal WITH.
_OPERATOR = PrivateKey()
_OPERATOR_NSEC = _OPERATOR.bech32()
_OPERATOR_NPUB = _OPERATOR.public_key.bech32()
_OPERATOR_PUB_HEX = _OPERATOR.public_key.hex()


def _seal(code: str, operator_pubkey: str = _OPERATOR_NPUB) -> str:
    """Seal a code the same way the collector must (NIP-44 to operator pubkey)."""
    return encrypt_collector_code(code, operator_pubkey)


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
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "at-123",
            "refresh_token": "rt-456",
            "expires_in": 1800,
            "token_type": "Bearer",
        }

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
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "at"}

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

    @pytest.mark.asyncio
    async def test_provider_400_keeps_body_and_names_fault(self):
        """The defect in #172: a 400 used to escape as HTTPStatusError with the
        body unread. Exchange must classify like refresh and keep X's words."""
        import httpx

        from tollbooth.oauth2_collector import OAuthRefreshDenied

        req = httpx.Request("POST", _TEST_TOKEN)
        mock_response = httpx.Response(
            400,
            json={
                "error": "invalid_client",
                "error_description": "Client authentication failed due to unknown client",
            },
            request=req,
        )

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ), pytest.raises(OAuthRefreshDenied) as caught:
            await exchange_code_for_token(
                "auth-code-xyz", "app-key", "app-secret",
                "https://example.com/cb", _TEST_TOKEN,
            )

        assert caught.value.oauth_error == "invalid_client"
        assert caught.value.fault == "client"
        assert caught.value.status_code == 400
        assert "unknown client" in caught.value.detail

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("oauth_error", "fault"), [
        ("invalid_grant", "grant"),
        ("invalid_client", "client"),
        ("unauthorized_client", "client"),
        ("invalid_request", "request"),
    ])
    async def test_exchange_refusal_names_whose_fault_it_is(self, oauth_error, fault):
        import httpx

        from tollbooth.oauth2_collector import OAuthRefreshDenied

        req = httpx.Request("POST", _TEST_TOKEN)
        mock_response = httpx.Response(
            400, json={"error": oauth_error}, request=req,
        )
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ), pytest.raises(OAuthRefreshDenied) as caught:
            await exchange_code_for_token(
                "code", "id", "secret", "https://cb", _TEST_TOKEN,
            )
        assert caught.value.fault == fault
        assert caught.value.oauth_error == oauth_error

    @pytest.mark.asyncio
    async def test_exchange_unnamed_4xx_is_unavailable_not_fatal(self):
        import httpx

        from tollbooth.oauth2_collector import OAuthRefreshUnavailable

        req = httpx.Request("POST", _TEST_TOKEN)
        mock_response = httpx.Response(
            403, text="<html>Forbidden</html>", request=req,
        )
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ), pytest.raises(OAuthRefreshUnavailable) as caught:
            await exchange_code_for_token(
                "code", "id", "secret", "https://cb", _TEST_TOKEN,
            )
        assert caught.value.status_code == 403

    @pytest.mark.asyncio
    async def test_exchange_redacts_auth_code_from_provider_echo(self):
        import httpx

        from tollbooth.oauth2_collector import OAuthRefreshDenied

        secret_code = "auth-code-secret-value"
        req = httpx.Request("POST", _TEST_TOKEN)
        mock_response = httpx.Response(
            400,
            json={
                "error": "invalid_grant",
                "error_description": f"code {secret_code} is invalid",
            },
            request=req,
        )
        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ), pytest.raises(OAuthRefreshDenied) as caught:
            await exchange_code_for_token(
                secret_code, "id", "secret", "https://cb", _TEST_TOKEN,
            )
        assert secret_code not in caught.value.detail
        assert "<redacted>" in caught.value.detail


# ---------------------------------------------------------------------------
# encrypt/decrypt_collector_code tests (NIP-44 sealed to operator)
# ---------------------------------------------------------------------------


class TestCollectorCodeSeal:
    """NIP-44 sealed envelope — only the operator nsec opens the code (#228)."""

    def test_roundtrip_with_operator_npub(self):
        encrypted = encrypt_collector_code("my-secret-code", _OPERATOR_NPUB)
        assert decrypt_collector_code(encrypted, _OPERATOR_NSEC) == "my-secret-code"

    def test_roundtrip_with_operator_hex_pubkey(self):
        code = "authorization-code-from-provider"
        encrypted = encrypt_collector_code(code, _OPERATOR_PUB_HEX)
        assert decrypt_collector_code(encrypted, _OPERATOR.hex()) == code

    def test_envelope_shape(self):
        encrypted = encrypt_collector_code("shape-check", _OPERATOR_NPUB)
        envelope = json.loads(encrypted)
        assert envelope["v"] == 2
        assert isinstance(envelope["epk"], str) and len(envelope["epk"]) == 64
        assert isinstance(envelope["ct"], str) and envelope["ct"]

    def test_ephemeral_sender_changes_each_seal(self):
        a = json.loads(encrypt_collector_code("same", _OPERATOR_NPUB))
        b = json.loads(encrypt_collector_code("same", _OPERATOR_NPUB))
        assert a["epk"] != b["epk"]
        assert a["ct"] != b["ct"]

    def test_public_npub_alone_cannot_decrypt(self):
        """THE #228 invariant: Neon-row + public npub must NOT open the seal.

        Pre-fix, key = SHA-256(state) with state = public npub, so any holder
        of the row decrypted the OAuth code. Post-fix the envelope only opens
        with the operator nsec.
        """
        patron = PrivateKey()
        public_npub = patron.public_key.bech32()  # attacker knows this
        code = "oauth-auth-code-VERY-SECRET"
        sealed = encrypt_collector_code(code, _OPERATOR_NPUB)

        # Attacker tries every public value they have — state, operator npub,
        # even the envelope's own ephemeral pubkey. None is an nsec.
        for guess in (public_npub, _OPERATOR_NPUB, json.loads(sealed)["epk"], ""):
            with pytest.raises(OAuthCollectorError):
                decrypt_collector_code(sealed, guess)

        # Only the real operator nsec opens it.
        assert decrypt_collector_code(sealed, _OPERATOR_NSEC) == code

    def test_wrong_operator_nsec_raises(self):
        sealed = encrypt_collector_code("my-secret-code", _OPERATOR_NPUB)
        other = PrivateKey()
        with pytest.raises(OAuthCollectorError):
            decrypt_collector_code(sealed, other.bech32())

    def test_legacy_sha256_state_blob_is_rejected(self):
        """v1 AES-GCM(SHA-256(state)) blobs must not silently 'decrypt'."""
        # A plausible-looking pre-fix blob (urlsafe b64 of 12+16 bytes).
        import base64
        import os

        legacy = base64.urlsafe_b64encode(os.urandom(28)).decode()
        with pytest.raises(OAuthCollectorError):
            decrypt_collector_code(legacy, _OPERATOR_NSEC)

    def test_empty_code_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            encrypt_collector_code("", _OPERATOR_NPUB)

    def test_encrypt_signature_does_not_take_state(self):
        """Regression guard: state must never re-enter the cipher API."""
        import inspect

        params = list(inspect.signature(encrypt_collector_code).parameters)
        assert params == ["code", "operator_pubkey"]
        params = list(inspect.signature(decrypt_collector_code).parameters)
        assert params == ["encrypted", "operator_nsec"]


# ---------------------------------------------------------------------------
# retrieve_code_from_collector tests
# ---------------------------------------------------------------------------


class TestRetrieveCodeFromCollector:
    """Tests for retrieve_code_from_collector."""

    @pytest.mark.asyncio
    async def test_returns_decrypted_code_on_success(self):
        state = "npub1abc123"  # lookup key only
        encrypted = _seal("auth-code-abc")

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
                "https://collector.example.com", state, _OPERATOR_NSEC,
            )

        assert result == "auth-code-abc"
        call_args = mock_http.post.call_args
        assert call_args[0][0] == "https://collector.example.com/mcp/"
        # State is still what we look the row up by — never the cipher key.
        posted = call_args.kwargs.get("json") or call_args[1].get("json")
        assert posted["params"]["arguments"]["state"] == state

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
                "https://collector.example.com", "npub1abc", _OPERATOR_NSEC,
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_strips_trailing_slash(self):
        state = "npub1xyz"
        encrypted = _seal("xyz")

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
                "https://collector.example.com/", state, _OPERATOR_NSEC,
            )

        assert result == "xyz"
        call_args = mock_http.post.call_args
        assert call_args[0][0] == "https://collector.example.com/mcp/"

    @pytest.mark.asyncio
    async def test_returns_decrypted_code_on_plain_json(self):
        # The collector may answer with Content-Type: application/json
        # (no SSE "data: " framing) — our Accept header allows it. The code
        # must be detected here too, not treated as "not yet available".
        state = "npub1json"
        encrypted = _seal("auth-code-json")

        json_body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "structuredContent": {"found": True, "code": encrypted}
                },
            }
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json_body
        mock_response.json.return_value = json.loads(json_body)

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            result = await retrieve_code_from_collector(
                "https://collector.example.com", state, _OPERATOR_NSEC,
            )

        assert result == "auth-code-json"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found_plain_json(self):
        json_body = json.dumps(
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

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json_body
        mock_response.json.return_value = json.loads(json_body)

        mock_http = AsyncMock()
        mock_http.post.return_value = mock_response
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "tollbooth.oauth2_collector.httpx.AsyncClient", return_value=mock_http
        ):
            result = await retrieve_code_from_collector(
                "https://collector.example.com", "npub1json", _OPERATOR_NSEC,
            )

        assert result is None


# --- OAuth state packing: operator npub round-trips via the state -------------
def test_pack_oauth_state_joins_patron_and_operator():
    from tollbooth.oauth2_collector import pack_oauth_state

    assert pack_oauth_state("npub1patron", "npub1operator") == "npub1patron.npub1operator"


def test_unpack_oauth_state_roundtrips():
    from tollbooth.oauth2_collector import pack_oauth_state, unpack_oauth_state

    patron, operator = "npub1patron", "npub1operator"
    assert unpack_oauth_state(pack_oauth_state(patron, operator)) == (patron, operator)


def test_unpack_legacy_patron_only_state_yields_empty_operator():
    # A pre-cutover state (patron npub only) has no operator half — the collector
    # keys on that to refuse it rather than seal to the wrong (public) key.
    from tollbooth.oauth2_collector import unpack_oauth_state

    assert unpack_oauth_state("npub1patrononly") == ("npub1patrononly", "")
