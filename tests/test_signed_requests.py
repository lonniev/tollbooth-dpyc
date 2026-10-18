"""Signed requests: whoever holds an npub's key needs no grant.

Pins:
  - a NIP-98 signature is bound to the request's method, path and body, is
    fresh, and is good once; the host may differ (a site's proxy forwards it);
  - over HTTP, a request signed by alice acts as alice, and as nobody else;
  - a service with no holders (SignedRequestProvider) serves every caller and
    knows a signer by its npub, never refusing an unsigned or bad request;
  - a high-value tool's patron-proof constraint is met by the patron's own key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.utilities.tests import run_server_async
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tests.test_holder_auth import MemoryHolderStore, _free_port
from tollbooth.constants import ErrorCode
from tollbooth.identity_proof import (
    PROOF_REASON_EXPIRED,
    PROOF_REASON_MALFORMED_JSON,
    PROOF_REASON_REPLAYED,
    PROOF_REASON_REQUEST_MISMATCH,
    sign_request,
    verify_request_signature,
)
from tollbooth.mcp_client import service_client
from tollbooth.runtime import OperatorRuntime, register_standard_tools

BODY = b'{"jsonrpc":"2.0","id":1,"method":"tools/call"}'
SHA = hashlib.sha256(BODY).hexdigest()


def _verify(header: str, *, method: str = "POST", path: str = "/mcp", sha: str = SHA):
    return verify_request_signature(header.removeprefix("Nostr "), method=method, path=path, body_sha256=sha)


def test_a_signed_request_proves_its_signer():
    key = PrivateKey()
    header = sign_request(key.bech32(), method="post", url="https://op.example/mcp", body=BODY)
    assert _verify(header) == (key.public_key.bech32(), None)


def test_the_site_host_may_differ_but_the_path_may_not():
    key = PrivateKey()
    proxied = sign_request(key.bech32(), method="POST", url="https://site.example/mcp", body=BODY)
    assert _verify(proxied)[0] == key.public_key.bech32()
    elsewhere = sign_request(key.bech32(), method="POST", url="https://op.example/other", body=BODY)
    assert _verify(elsewhere) == (None, PROOF_REASON_REQUEST_MISMATCH)


@pytest.mark.parametrize("change", ["method", "body"])
def test_a_signature_authorizes_only_its_own_method_and_body(change):
    header = sign_request(PrivateKey().bech32(), method="POST", url="https://op.example/mcp", body=BODY)
    kwargs = {"method": "GET"} if change == "method" else {"sha": hashlib.sha256(b"{}").hexdigest()}
    assert _verify(header, **kwargs) == (None, PROOF_REASON_REQUEST_MISMATCH)


def test_a_signature_is_good_once():
    header = sign_request(PrivateKey().bech32(), method="POST", url="https://op.example/mcp", body=BODY)
    assert _verify(header)[1] is None
    assert _verify(header) == (None, PROOF_REASON_REPLAYED)


def test_a_stale_signature_is_refused():
    key = PrivateKey()
    event = Event(
        pubkey=key.public_key.hex(), kind=27235, content="", created_at=int(time.time()) - 600,
        tags=[["u", "https://op.example/mcp"], ["method", "POST"], ["payload", SHA]],
    )
    event.sign(key.hex())
    header = base64.b64encode(json.dumps(event.to_dict()).encode()).decode()
    assert _verify(header) == (None, PROOF_REASON_EXPIRED)


def test_a_header_that_is_not_base64_json_is_malformed():
    assert _verify("not base64 at all") == (None, PROOF_REASON_MALFORMED_JSON)


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_signer_acts_as_its_own_npub():
    op, alice, bob = PrivateKey(), PrivateKey(), PrivateKey()
    port = _free_port()
    base = f"http://127.0.0.1:{port}"

    mcp = FastMCP("signed-test")
    rt = OperatorRuntime(nsec_env_var="__UNUSED__", holder_auth_url=base)
    rt._nsec = op.bech32()
    rt.holder_store = AsyncMock(return_value=MemoryHolderStore())  # type: ignore[method-assign]
    rt.proof_grant_revocations = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(is_revoked=AsyncMock(return_value=False)),
    )
    register_standard_tools(mcp, "test", rt, service_name="test")

    @mcp.tool
    async def act_as(npub: str) -> dict[str, Any]:
        return await rt.require_caller_proof(npub) or {"acted_as": npub}

    async def call(nsec: str | None, args: dict[str, Any]) -> dict[str, Any]:
        async with service_client(f"{base}/mcp", nsec) as client:
            return (await client.call_tool("act_as", args)).structured_content or {}

    async with run_server_async(mcp, port=port):
        alice_npub = alice.public_key.bech32()
        assert await call(alice.bech32(), {"npub": alice_npub}) == {"acted_as": alice_npub}
        # Her key proves her, and only her.
        refused = await call(alice.bech32(), {"npub": bob.public_key.bech32()})
        assert refused["error_code"] == ErrorCode.PROOF_REQUIRED
        # Anonymous is nobody.
        assert (await call(None, {"npub": alice_npub}))["error_code"] == ErrorCode.PROOF_REQUIRED


@pytest.mark.asyncio
async def test_a_service_with_no_holders_serves_everyone_and_knows_a_signer():
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    from tollbooth.holder_auth import SignedRequestProvider, signed_npub

    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    mcp = FastMCP("oracle-like", auth=SignedRequestProvider())

    @mcp.tool
    def whoami() -> str:
        return signed_npub() or "anonymous"

    async def ask(**transport: Any) -> str:
        async with Client(StreamableHttpTransport(url, **transport)) as client:
            return str((await client.call_tool("whoami", {})).data)

    alice = PrivateKey()
    elsewhere = sign_request(alice.bech32(), method="POST", url=url, body=b"{}")
    async with run_server_async(mcp, port=port):
        # No Authorization header at all: served, as nobody.
        assert await ask() == "anonymous"
        assert await ask(headers={"Authorization": "Bearer anonymous"}) == "anonymous"
        # A signature for some other body is no proof — but it is not refused.
        assert await ask(headers={"Authorization": elsewhere}) == "anonymous"
        # A request alice signs is known as alice.
        async with service_client(url, alice.bech32()) as client:
            assert (await client.call_tool("whoami", {})).data == alice.public_key.bech32()


def test_the_patron_proof_constraint_is_met_by_the_patrons_own_key():
    from tollbooth.constraints.patron_proof import PatronProofConstraint

    npub = PrivateKey().public_key.bech32()
    context = SimpleNamespace(patron=SimpleNamespace(npub=npub), env=SimpleNamespace(tool_name="t"))
    with patch("tollbooth.holder_auth.signed_npub", return_value=npub):
        assert PatronProofConstraint().evaluate(context).allowed  # type: ignore[arg-type]
    with patch("tollbooth.holder_auth.signed_npub", return_value=None):
        assert not PatronProofConstraint().evaluate(context).allowed  # type: ignore[arg-type]
