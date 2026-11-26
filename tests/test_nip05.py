"""NIP-05 address resolution for patron identification (issue #407).

Patrons may pass ``name@domain`` anywhere an npub is accepted; the wheel
resolves it via ``https://domain/.well-known/nostr.json`` before validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pynostr.key import PrivateKey

from tollbooth.nip05 import Nip05ResolutionError, clear_nip05_cache, resolve_nip05
from tollbooth.runtime import OperatorRuntime, resolve_npub
from tollbooth.tools import proof as proof_tools


@pytest.fixture(autouse=True)
def _clear_nip05_cache() -> None:
    clear_nip05_cache()
    yield
    clear_nip05_cache()


def _key() -> tuple[str, str]:
    pk = PrivateKey()
    return pk.public_key.hex(), pk.public_key.bech32()


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


class TestResolveNip05:
    def test_happy_path(self) -> None:
        hex_pk, npub = _key()
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(200, {"names": {"alice": hex_pk}})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            assert resolve_nip05("alice@example.com") == npub
            client.get.assert_called_once_with(
                "https://example.com/.well-known/nostr.json",
                params={"name": "alice"},
            )

    def test_name_case_insensitive(self) -> None:
        hex_pk, npub = _key()
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            # Server returns lowercase key; query may be mixed case.
            client.get.return_value = _mock_response(200, {"names": {"alice": hex_pk}})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            assert resolve_nip05("Alice@Example.COM") == npub

    def test_missing_name(self) -> None:
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(200, {"names": {"bob": "ab" * 32}})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            with pytest.raises(Nip05ResolutionError, match="not found"):
                resolve_nip05("alice@example.com")

    def test_http_error(self) -> None:
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(404, {})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            with pytest.raises(Nip05ResolutionError, match="HTTP 404"):
                resolve_nip05("alice@example.com")

    def test_invalid_address(self) -> None:
        with pytest.raises(Nip05ResolutionError, match="missing @"):
            resolve_nip05("alice.example.com")


class TestResolveNpubAcceptsNip05:
    """The defect #407 confirms: name@domain must resolve, not raise."""

    def test_npub_still_works(self) -> None:
        _, npub = _key()
        assert resolve_npub(npub) == npub

    def test_nip05_resolves_via_resolve_npub(self) -> None:
        hex_pk, npub = _key()
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(200, {"names": {"alice": hex_pk}})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            assert resolve_npub("alice@example.com") == npub

    def test_runtime_wrapper(self) -> None:
        hex_pk, npub = _key()
        rt = OperatorRuntime.__new__(OperatorRuntime)
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(200, {"names": {"alice": hex_pk}})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            assert rt.resolve_npub("alice@example.com") == npub

    def test_validation_error_accepts_nip05(self) -> None:
        hex_pk, _ = _key()
        rt = OperatorRuntime.__new__(OperatorRuntime)
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(200, {"names": {"alice": hex_pk}})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            assert rt.npub_validation_error("alice@example.com", param="patron_npub") is None


class TestRequestNpubProofAcceptsNip05:
    @pytest.mark.asyncio
    async def test_request_resolves_nip05_before_dm(self) -> None:
        _, npub = _key()
        rt = MagicMock()
        rt.npub_validation_error.return_value = None
        rt.resolve_npub.return_value = npub
        rt.holder_auth_enabled = True
        rt._npub_proof_greeting = None

        async def _courier():
            return MagicMock()

        rt.courier = _courier

        async def _send(*_a, **_k):
            return {"success": True, "nonce": "n1"}

        with (
            patch("tollbooth.holder_auth.current_holder", return_value="holder-1"),
            patch(
                "tollbooth.tools.proof.send_approval_request",
                side_effect=_send,
            ) as send_async,
        ):
            result = await proof_tools.request_npub_proof_tool(
                rt, "alice@example.com", service_name="demo",
            )

        assert result["success"] is True
        # Must DM the resolved npub, never the NIP-05 string.
        args, _kwargs = send_async.call_args
        assert args[1] == npub
        rt.resolve_npub.assert_called_with("alice@example.com")
