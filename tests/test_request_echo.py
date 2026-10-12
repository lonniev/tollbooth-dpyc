"""request_echo: what a client sends, fingerprinted — never repeated back."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.utilities.tests import run_server_async

from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.tool_identity import STANDARD_IDENTITIES, capability_uuid
from tollbooth.tools.request_echo import build_request_echo, fingerprint


def _echo(**overrides):
    args = {
        "method": "POST",
        "path": "/mcp",
        "query": [],
        "headers": [],
        "cookies": {},
        "meta": None,
        "client_info": None,
        "process_id": 1,
    }
    args.update(overrides)
    return build_request_echo(**args)


def test_secret_shaped_values_are_fingerprinted_never_echoed():
    out = _echo(
        query=[("k", "s3cret-k")],
        headers=[("Authorization", "Bearer tok"), ("X-Conversation", "conv-1")],
        cookies={"sid": "cookie-value"},
        meta={"com.example/conversation": {"id": "conv-1"}, "progressToken": 7},
    )
    blob = json.dumps(out)
    for raw in ("s3cret-k", "Bearer tok", "conv-1", "cookie-value"):
        assert raw not in blob
    assert out["url"]["query"]["k"] == fingerprint("s3cret-k")
    assert out["headers"]["authorization"] == fingerprint("Bearer tok")
    assert out["cookies"]["sid"] == fingerprint("cookie-value")
    assert out["meta"]["com.example/conversation"]["id"] == fingerprint("conv-1")
    assert out["meta"]["progressToken"] == fingerprint(7)


def test_client_software_and_protocol_names_stay_readable():
    info = {"name": "claude-ai", "version": "1.0"}
    out = _echo(
        headers=[("User-Agent", "Claude-User"), ("Mcp-Protocol-Version", "2025-11-25")],
        meta={"io.modelcontextprotocol/clientInfo": info},
        client_info=info,
    )
    assert out["headers"]["user-agent"] == "Claude-User"
    assert out["headers"]["mcp-protocol-version"] == "2025-11-25"
    assert out["meta"]["io.modelcontextprotocol/clientInfo"] == info
    assert out["client_info"] == info


def test_equal_values_share_a_fingerprint():
    assert fingerprint("same") == fingerprint("same")
    assert fingerprint("same") != fingerprint("other")


def test_request_echo_is_a_free_standard_tool():
    identity = STANDARD_IDENTITIES[capability_uuid("request_echo")]
    assert identity.capability == "request_echo"
    assert identity.category == "free"


@pytest.mark.asyncio
async def test_round_trip_over_http_sees_query_cookie_and_meta():
    mcp = FastMCP("echo-test")
    rt = OperatorRuntime(nsec_env_var="__UNUSED__")
    register_standard_tools(mcp, "test", rt, service_name="test")

    async with run_server_async(mcp) as url:
        transport = StreamableHttpTransport(
            f"{url}?k=probe-k",
            headers={"Cookie": "sid=probe-cookie"},
        )
        async with Client(transport) as client:
            result = await client.call_tool(
                "test_request_echo", {}, meta={"com.example/conversation": "conv-A"}
            )

    out = result.structured_content
    assert out["url"]["path"] == "/mcp"
    assert out["url"]["query"] == {"k": fingerprint("probe-k")}
    assert out["cookies"] == {"sid": fingerprint("probe-cookie")}
    assert out["meta"]["com.example/conversation"] == fingerprint("conv-A")
    assert out["client_info"]["name"]
    assert "probe-k" not in json.dumps(out)
