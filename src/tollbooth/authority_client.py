"""Server-to-server MCP client for obtaining Authority certificates.

Used by operator servers (thebrain-mcp, excalibur-mcp) to auto-certify
credit purchases without requiring the AI agent to call the Authority directly.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from fastmcp import Client  # type: ignore[import-untyped]
except ImportError:
    Client = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class AuthorityCertifyError(Exception):
    """Raised when Authority certification fails."""


class AuthorityCertifier:
    """Server-to-server MCP client for obtaining Authority certificates.

    Opens a short-lived ``fastmcp.Client`` SSE connection with Horizon OAuth
    auto-negotiation. One connection per ``certify()`` call (credit-critical
    path — correctness over connection pooling).
    """

    def __init__(
        self,
        authority_url: str,
        operator_npub: str,
        certify_tool_name: str = "authority_certify_credits",
    ) -> None:
        self._authority_url = authority_url
        self._operator_npub = operator_npub
        self._certify_tool_name = certify_tool_name

    async def certify(self, amount_sats: int) -> dict[str, Any]:
        """Call the Authority's certify_credits tool and return the certificate dict.

        Returns dict with keys: certificate, jti, amount_sats, fee_sats,
        net_sats, expires_at (as returned by the Authority).

        Raises ``AuthorityCertifyError`` on any failure (connection, auth, tool error).
        """
        if Client is None:
            raise AuthorityCertifyError(
                "fastmcp package required for auto-certification. "
                "Install with: pip install fastmcp"
            )

        try:
            async with Client(self._authority_url, auth="oauth") as client:
                result = await client.call_tool(
                    self._certify_tool_name,
                    {
                        "operator_id": self._operator_npub,
                        "amount_sats": amount_sats,
                    },
                )
        except AuthorityCertifyError:
            raise
        except Exception as e:
            raise AuthorityCertifyError(
                f"Failed to connect to Authority at {self._authority_url}: {e}"
            ) from e

        return self._parse_result(result)

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Extract the certificate dict from the MCP tool result."""
        # fastmcp.Client.call_tool returns a list of content blocks
        if isinstance(result, list):
            for block in result:
                # TextContent has .text attribute
                if hasattr(block, "text"):
                    import json

                    try:
                        data = json.loads(block.text)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(data, dict):
                        if data.get("success") is False:
                            raise AuthorityCertifyError(
                                f"Authority refused certification: "
                                f"{data.get('error', 'unknown error')}"
                            )
                        if "certificate" in data:
                            return data
            raise AuthorityCertifyError(
                f"Authority returned unexpected response format: {result}"
            )

        if isinstance(result, dict):
            if result.get("success") is False:
                raise AuthorityCertifyError(
                    f"Authority refused certification: "
                    f"{result.get('error', 'unknown error')}"
                )
            if "certificate" in result:
                return result

        raise AuthorityCertifyError(
            f"Authority returned unexpected response format: {result}"
        )
