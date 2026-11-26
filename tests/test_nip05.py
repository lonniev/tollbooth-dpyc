"""NIP-05 address resolution for patron identification (issue #407).

Patrons may pass ``name@domain`` anywhere an npub is accepted; the wheel
resolves it via ``https://domain/.well-known/nostr.json`` before validation.
"""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pynostr.key import PrivateKey

from tollbooth import nip05
from tollbooth.nip05 import Nip05ResolutionError, clear_nip05_cache, resolve_nip05
from tollbooth.runtime import OperatorRuntime, resolve_npub
from tollbooth.tools import proof as proof_tools


@pytest.fixture(autouse=True)
def _clear_nip05_cache() -> None:
    clear_nip05_cache()
    yield
    clear_nip05_cache()


@pytest.fixture(autouse=True)
def _public_dns(request: pytest.FixtureRequest):
    """Answer every DNS lookup with a public address, unless a test opts out."""
    if "real_dns_check" in request.keywords:
        yield
        return
    with patch(
        "tollbooth.nip05.socket.getaddrinfo",
        return_value=[(None, None, None, "", ("93.184.215.14", 443))],
    ):
        yield


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

            with pytest.raises(Nip05ResolutionError, match="has no NIP-05 name .alice."):
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

    def test_html_page_says_the_domain_publishes_no_names(self) -> None:
        resp = _mock_response(200)
        resp.json.side_effect = ValueError("Expecting value")
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = resp
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            with pytest.raises(
                Nip05ResolutionError,
                match=r"example\.com doesn't publish NIP-05 names\. Check the part of alice@example\.com after the @",
            ):
                resolve_nip05("alice@example.com")

    def test_json_without_names_map_says_the_same(self) -> None:
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = MagicMock()
            client.get.return_value = _mock_response(200, {"hello": "world"})
            client.__enter__ = MagicMock(return_value=client)
            client.__exit__ = MagicMock(return_value=False)
            mock_cls.return_value = client

            with pytest.raises(Nip05ResolutionError, match="doesn't publish NIP-05 names"):
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


def _client_returning(resp: httpx.Response) -> MagicMock:
    client = MagicMock()
    client.get.return_value = resp
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


class TestLookupIsFenced:
    """The caller names the domain, so the server must not fetch just anything."""

    def test_redirects_are_not_followed(self) -> None:
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            mock_cls.return_value = _client_returning(_mock_response(302, {}))
            with pytest.raises(Nip05ResolutionError, match="redirect"):
                resolve_nip05("alice@example.com")
            assert mock_cls.call_args.kwargs["follow_redirects"] is False

    @pytest.mark.parametrize(
        "address",
        [
            "alice@10.0.0.1",
            "alice@169.254.169.254",
            "alice@[::1]",
            "alice@127.0.0.1",
            "alice@example.com:8443",
            "alice@localhost.localdomain",
            "alice@db.internal",
            "alice@printer.local",
            "alice@router.home.arpa",
            "alice@-bad.example.com",
            "alice@example.123",
        ],
    )
    def test_non_public_domains_are_refused_before_any_fetch(self, address: str) -> None:
        with patch("tollbooth.nip05.httpx.Client") as mock_cls:
            with pytest.raises(Nip05ResolutionError):
                resolve_nip05(address)
            mock_cls.assert_not_called()

    @pytest.mark.real_dns_check
    @pytest.mark.parametrize("ip", ["10.1.2.3", "127.0.0.1", "169.254.169.254", "fd00::1", "::1"])
    def test_hostname_resolving_to_a_private_address_is_refused(self, ip: str) -> None:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        with patch(
            "tollbooth.nip05.socket.getaddrinfo",
            return_value=[(family, socket.SOCK_STREAM, 6, "", (ip, 443))],
        ), patch("tollbooth.nip05.httpx.Client") as mock_cls:
            with pytest.raises(Nip05ResolutionError, match="non-public"):
                resolve_nip05("alice@rebind.example.com")
            mock_cls.assert_not_called()

    @pytest.mark.real_dns_check
    def test_one_private_answer_among_public_ones_is_refused(self) -> None:
        with patch(
            "tollbooth.nip05.socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 443)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443)),
            ],
        ), patch("tollbooth.nip05.httpx.Client") as mock_cls:
            with pytest.raises(Nip05ResolutionError, match="non-public"):
                resolve_nip05("alice@mixed.example.com")
            mock_cls.assert_not_called()


class TestCacheExpires:
    def test_a_remapped_address_is_seen_after_the_ttl(self) -> None:
        old_hex, old_npub = _key()
        new_hex, new_npub = _key()
        clock = [1000.0]
        with patch("tollbooth.nip05.time.monotonic", side_effect=lambda: clock[0]), \
                patch("tollbooth.nip05.httpx.Client") as mock_cls:
            client = _client_returning(_mock_response(200, {"names": {"alice": old_hex}}))
            mock_cls.return_value = client
            assert resolve_nip05("alice@example.com") == old_npub

            client.get.return_value = _mock_response(200, {"names": {"alice": new_hex}})
            clock[0] += nip05.CACHE_TTL_S - 1
            assert resolve_nip05("alice@example.com") == old_npub  # still cached

            clock[0] += 2
            assert resolve_nip05("alice@example.com") == new_npub  # looked up again
