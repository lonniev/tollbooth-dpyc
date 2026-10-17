"""How one DPYC process calls another's MCP endpoint.

A service-to-service call proves itself in its arguments (an inline kind-27235
proof signed with the caller's key), never with OAuth. It still carries a
bearer, because an endpoint with holder bearer on refuses a request with no
``Authorization`` header; any value the server did not mint is an anonymous
caller there, so this placeholder is all it needs.
"""

from __future__ import annotations

from typing import Any

ANONYMOUS_BEARER = "anonymous"


def service_client(url: str) -> Any:
    """A FastMCP client for another DPYC service, carrying the anonymous bearer."""
    from fastmcp import Client

    return Client(url, auth=ANONYMOUS_BEARER)
