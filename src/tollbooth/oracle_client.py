"""Server-to-server MCP client for Oracle delegation.

Used by operator servers (thebrain-mcp, excalibur-mcp) to delegate
community queries to the DPYC Oracle without requiring a separate
MCP connection from the AI agent.

Oracle tools are free and unauthenticated — no credits required.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
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

    Opens a short-lived ``fastmcp.Client`` connection per ``call_tool()``
    invocation, unless the client came from :meth:`session`, which shares one
    connection across every call. Oracle tools are free and unauthenticated,
    so no credits or certificates are needed.
    """

    def __init__(self, oracle_url: str, *, connection: Any = None) -> None:
        self._oracle_url = oracle_url
        self._connection = connection

    @asynccontextmanager
    async def session(self) -> AsyncIterator[OracleClient]:
        """Yield a client whose calls all share one open connection.

        A cold start asks the Oracle more than one question; a connection per
        question pays the handshake each time and forces the questions into
        sequence. Calls through a session may run concurrently — MCP
        multiplexes them over the one connection.
        """
        _require_fastmcp()
        async with AsyncExitStack() as stack:
            try:
                connection = await stack.enter_async_context(
                    Client(self._oracle_url, auth="oauth")
                )
            except Exception as e:
                raise OracleClientError(
                    f"Failed to connect to Oracle at {self._oracle_url}: {e}"
                ) from e
            yield OracleClient(self._oracle_url, connection=connection)

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call an Oracle tool by name and return the parsed result dict.

        Raises ``OracleClientError`` on any failure (connection, parse, tool error).
        """
        _require_fastmcp()
        try:
            if self._connection is not None:
                result = await self._connection.call_tool(tool_name, arguments or {})
            else:
                async with Client(self._oracle_url, auth="oauth") as client:
                    result = await client.call_tool(tool_name, arguments or {})
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

    async def report_relay_failure(
        self,
        relay: str,
        reporter_npub: str,
        signed_event: str,
        mode: str = "unknown",
    ) -> dict[str, Any]:
        """Tell the Oracle a curated relay was unreachable, so it re-measures it.

        A report does not set rank: the Oracle probes the relay itself and its own
        measurement decides, so a mistaken report costs one probe and changes
        nothing. Report failures only — a relay that works needs no announcement.

        ``signed_event`` must be a Nostr event signed by ``reporter_npub`` whose
        content names ``relay``, signed within the Oracle's freshness window (it
        binds the signature to a moment so a report cannot be replayed forever).

        Raises ``OracleClientError`` on connection/parse failure. A refusal the
        Oracle can articulate — not a member, relay not curated — comes back as
        data in the returned dict, because those are answers, not faults.
        """
        return await self.call_tool(
            "report_relay_failure",
            {
                "relay": relay,
                "reporter_npub": reporter_npub,
                "signed_event": signed_event,
                "mode": mode,
            },
        )

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


def _require_fastmcp() -> None:
    if Client is None:
        raise OracleClientError(
            "fastmcp package required for Oracle delegation. "
            "Install with: pip install fastmcp"
        )


def default_oracle_client() -> OracleClient:
    """An ``OracleClient`` pointed at the SDK's baked-in Oracle endpoint.

    This is the one fixed coordinate an nsec-only operator may know a priori
    (``DPYC_ORACLE_MCP_URL``). Everything else — relays, its Authority, sibling
    services — is asked of the Oracle, never read from GitHub.
    """
    from tollbooth.constants import DPYC_ORACLE_MCP_URL

    return OracleClient(DPYC_ORACLE_MCP_URL)
