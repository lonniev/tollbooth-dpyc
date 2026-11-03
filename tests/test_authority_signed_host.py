"""An Authority accepts a signed request only when it was signed for its host (scan F30).

Operators sign certify_credits straight to their Authority, through no proxy.
Without the host check, an Authority that received such a request could replay
it at a sibling Authority within the signature's window. Operators still accept
any host, because a site's same-origin proxy forwards requests signed for the site.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp import FastMCP
from fastmcp.utilities.tests import run_server_async
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tests.test_holder_auth import MemoryHolderStore, _free_port
from tests.test_signed_requests import BODY, _verify
from tollbooth.identity_proof import PROOF_REASON_REQUEST_MISMATCH, sign_request
from tollbooth.mcp_client import service_client
from tollbooth.runtime import OperatorRuntime, register_standard_tools


def test_a_bound_host_refuses_a_request_signed_for_another():
    key = PrivateKey()
    sibling = sign_request(key.bech32(), method="POST", url="https://sibling.example/mcp", body=BODY)
    assert _verify_for(sibling, "authority.example") == (None, PROOF_REASON_REQUEST_MISMATCH)
    own = sign_request(key.bech32(), method="POST", url="https://Authority.example/mcp", body=BODY)
    assert _verify_for(own, "authority.example") == (key.public_key.bech32(), None)


def test_no_bound_host_still_accepts_a_proxied_request():
    key = PrivateKey()
    proxied = sign_request(key.bech32(), method="POST", url="https://site.example/mcp", body=BODY)
    assert _verify(proxied)[0] == key.public_key.bech32()


def _verify_for(header: str, host: str):
    import hashlib

    from tollbooth.identity_proof import verify_request_signature

    return verify_request_signature(
        header.removeprefix("Nostr "), method="POST", path="/mcp",
        body_sha256=hashlib.sha256(BODY).hexdigest(), host=host,
    )


def test_register_authority_tools_binds_the_authority_to_its_own_host():
    from tollbooth.authority import AUTHORITY_TOOL_REGISTRY, register_authority_tools
    from tollbooth.tool_identity import STANDARD_IDENTITIES

    mcp = FastMCP("authority-test")
    rt = OperatorRuntime(
        tool_registry={**STANDARD_IDENTITIES, **AUTHORITY_TOOL_REGISTRY},
        nsec_env_var="__UNUSED__", vault_source="env",
        holder_auth_url="https://Authority.example",
    )
    register_standard_tools(mcp, "authority", rt, service_name="authority")
    assert mcp.auth.signed_host is None
    register_authority_tools(mcp, rt)
    assert mcp.auth.signed_host == "authority.example"


@pytest.mark.asyncio
async def test_over_http_a_request_signed_for_another_host_is_nobody():
    op, alice = PrivateKey(), PrivateKey()
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    mcp = FastMCP("signed-host-test")
    rt = OperatorRuntime(nsec_env_var="__UNUSED__", holder_auth_url=base)
    rt._nsec = op.bech32()
    rt.holder_store = AsyncMock(return_value=MemoryHolderStore())  # type: ignore[method-assign]
    rt.proof_grant_revocations = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(is_revoked=AsyncMock(return_value=False)),
    )
    register_standard_tools(mcp, "test", rt, service_name="test")
    mcp.auth.bind_signed_requests_to("127.0.0.1")

    @mcp.tool
    async def act_as(npub: str) -> dict[str, Any]:
        return await rt.require_caller_proof(npub) or {"acted_as": npub}

    async def call(url: str) -> dict[str, Any]:
        async with service_client(url, alice.bech32()) as client:
            return (await client.call_tool("act_as", {"npub": alice.public_key.bech32()})).structured_content or {}

    async with run_server_async(mcp, port=port):
        assert await call(f"{base}/mcp") == {"acted_as": alice.public_key.bech32()}
        # Same server, same path, signed for a different host name: the
        # connection is refused at the door, as any bad signature is.
        with pytest.raises(Exception, match="Server returned an error response|401"):
            await call(f"http://localhost:{port}/mcp")
