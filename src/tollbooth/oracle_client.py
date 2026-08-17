"""Server-to-server MCP client for Oracle delegation.

Used by operator servers (thebrain-mcp, excalibur-mcp) to delegate
community queries to the DPYC Oracle without requiring a separate
MCP connection from the AI agent.

Oracle tools are free and unauthenticated — no credits required.
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    from fastmcp import Client  # type: ignore[import-untyped]
except ImportError:
    Client = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class OracleClientError(Exception):
    """Raised when an Oracle delegation call fails."""


class OracleClient:
    """Server-to-server MCP client for Oracle tool calls.

    Opens a short-lived ``fastmcp.Client`` SSE connection per
    ``call_tool()`` invocation. Oracle tools are free and
    unauthenticated, so no credits or certificates are needed.
    """

    def __init__(self, oracle_url: str) -> None:
        self._oracle_url = oracle_url

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call an Oracle tool by name and return the parsed result dict.

        Raises ``OracleClientError`` on any failure (connection, parse, tool error).
        """
        if Client is None:
            raise OracleClientError(
                "fastmcp package required for Oracle delegation. "
                "Install with: pip install fastmcp"
            )

        try:
            async with Client(self._oracle_url, auth="oauth") as client:
                result = await client.call_tool(tool_name, arguments or {})
        except OracleClientError:
            raise
        except Exception as e:
            raise OracleClientError(
                f"Failed to connect to Oracle at {self._oracle_url}: {e}"
            ) from e

        return self._parse_result(result)

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Extract a dict from the MCP tool result.

        Duck-types CallToolResult unwrapping (same pattern as
        ``AuthorityCertifier._parse_result``).
        """
        # Unwrap CallToolResult with .data dict (structured output)
        if hasattr(result, "data") and isinstance(result.data, dict):
            return result.data

        # Unwrap CallToolResult with .content list (text blocks)
        if hasattr(result, "content") and isinstance(
            result.content, list
        ):
            result = result.content

        # Parse list of text content blocks
        if isinstance(result, list):
            for block in result:
                if hasattr(block, "text"):
                    try:
                        data = json.loads(block.text)
                    except (json.JSONDecodeError, TypeError):
                        # Plain text response (e.g. markdown from how_to_join, about)
                        return {"text": block.text}
                    if isinstance(data, dict):
                        return data
                    # JSON-parsed but not a dict (e.g. a list or scalar)
                    return {"text": block.text}
            raise OracleClientError(
                f"Oracle returned unexpected response format: {result}"
            )

        if isinstance(result, dict):
            return result

        raise OracleClientError(
            f"Oracle returned unexpected response format: {result}"
        )

    async def check_ban_status(self, npub: str) -> dict[str, Any]:
        """Check whether *npub* is banned via the Oracle.

        Returns ``{"banned": bool, "reason": str | None}``.
        Raises ``OracleClientError`` on connection/parse failure.
        """
        return await self.call_tool("check_ban_status", {"npub": npub})

    async def get_relays(self) -> list[str]:
        """Fetch the DPYC Nostr relay set (primary-first) via the Oracle.

        Replaces reading ``relays.json`` from GitHub directly. Raises
        ``OracleClientError`` if the Oracle is unreachable or returns no relays.
        """
        result = await self.call_tool("get_relays", {})
        relays = result.get("relays")
        if not isinstance(relays, list) or not relays:
            raise OracleClientError(f"Oracle returned no relays: {result}")
        return relays

    async def resolve_authority_for(self, npub: str) -> dict[str, Any] | None:
        """Resolve the certifying Authority ``{npub, url, name}`` for an operator.

        Returns ``None`` when the Oracle can't resolve one (unknown npub, trust
        root, or no service listed). Raises ``OracleClientError`` only on a
        connection/parse failure — a resolvable "no authority" answer is data.
        """
        result = await self.call_tool("resolve_authority_for", {"npub": npub})
        if not result.get("success"):
            return None
        authority = result.get("authority")
        return authority if isinstance(authority, dict) else None

    async def resolve_service(
        self, name: str = "", npub: str = ""
    ) -> dict[str, Any] | None:
        """Resolve a DPYC service by name or npub via the Oracle.

        Returns ``{npub, url, name, role, purchase_mode}`` or ``None`` if not
        found. Raises ``OracleClientError`` on connection/parse failure.
        """
        result = await self.call_tool(
            "resolve_service", {"name": name, "npub": npub}
        )
        if not result.get("success"):
            return None
        service = result.get("service")
        return service if isinstance(service, dict) else None


def default_oracle_client() -> OracleClient:
    """An ``OracleClient`` pointed at the SDK's baked-in Oracle endpoint.

    This is the one fixed coordinate an nsec-only operator may know a priori
    (``DPYC_ORACLE_MCP_URL``). Everything else — relays, its Authority, sibling
    services — is asked of the Oracle, never read from GitHub.
    """
    from tollbooth.constants import DPYC_ORACLE_MCP_URL

    return OracleClient(DPYC_ORACLE_MCP_URL)
