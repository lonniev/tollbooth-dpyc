"""How one DPYC process calls another's MCP endpoint.

A caller that holds an npub's key signs every request with it (NIP-98), and
the server acts as that npub. A caller acting as nobody still carries a
bearer, because an endpoint with holder bearer on refuses a request with no
``Authorization`` header; any value the server did not mint is an anonymous
caller there, so this placeholder is all it needs.
"""

from __future__ import annotations

from typing import Any

ANONYMOUS_BEARER = "anonymous"

# For a raw JSON-RPC POST to another service's /mcp (no MCP client library).
JSONRPC_POST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "Authorization": f"Bearer {ANONYMOUS_BEARER}",
}


def service_client(url: str, nsec: str | None = None) -> Any:
    """A FastMCP client for another DPYC service.

    With ``nsec``, every request is signed by that key; without it, the caller
    is anonymous.
    """
    from fastmcp import Client

    if nsec:
        from tollbooth.nostr_auth import NostrRequestAuth

        return Client(url, auth=NostrRequestAuth(nsec))
    return Client(url, auth=ANONYMOUS_BEARER)
