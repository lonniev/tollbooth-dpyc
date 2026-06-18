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
    """Raised when Authority certification fails.

    Carries the Authority's structured ``error_code`` when the failure was a
    tool-level refusal (e.g. ``insufficient_balance`` when the Authority's own
    certification balance is exhausted), so callers can branch on the code
    rather than parsing prose. Empty for connection/transport failures.
    """

    def __init__(self, message: str, *, error_code: str = "") -> None:
        super().__init__(message)
        self.error_code = error_code


class AuthorityCertifier:
    """Server-to-server MCP client for obtaining Authority certificates.

    Opens a short-lived ``fastmcp.Client`` SSE connection. One connection
    per ``certify()`` call (credit-critical path — correctness over
    connection pooling).
    """

    def __init__(
        self,
        authority_url: str,
        operator_npub: str,
        operator_nsec: str = "",
        certify_tool_name: str = "authority_certify_credits",
    ) -> None:
        self._authority_url = authority_url
        self._operator_npub = operator_npub
        self._operator_nsec = operator_nsec
        self._certify_tool_name = certify_tool_name

    def _make_proof(self, tool_name: str) -> str:
        """Create a kind-27235 proof for the given tool."""
        if not self._operator_nsec:
            return ""
        from tollbooth.identity_proof import create_proof
        return create_proof(self._operator_nsec, tool_name)

    async def certify_credits(self, amount_sats: int) -> dict[str, Any]:
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
                        "npub": self._operator_npub,
                        "amount_sats": amount_sats,
                        # The Authority's verify_proof gate uses the runtime
                        # mcp_name (`<slug>_<func>`, e.g. "authority_certify_credits")
                        # since wheel 0.24.0. Sign for whatever wire name the
                        # caller is invoking — that's what the verifier sees.
                        "proof": self._make_proof(self._certify_tool_name),
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
        # Unwrap CallToolResult (fastmcp dataclass) — duck typing to avoid import
        if hasattr(result, "data") and isinstance(getattr(result, "data"), dict):
            result = result.data
        elif hasattr(result, "content") and isinstance(
            getattr(result, "content"), list
        ):
            result = result.content

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
                                f"{data.get('error', 'unknown error')}",
                                error_code=str(data.get("error_code", "")),
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
                    f"{result.get('error', 'unknown error')}",
                    error_code=str(result.get("error_code", "")),
                )
            if "certificate" in result:
                return result

        raise AuthorityCertifyError(
            f"Authority returned unexpected response format: {result}"
        )

    async def check_balance(self) -> dict[str, Any]:
        """Call the Authority's check_balance tool for this operator.

        Returns the operator's tax balance at the Authority — the sats
        available for certifying patron purchases.
        """
        if Client is None:
            raise AuthorityCertifyError("fastmcp package required")

        try:
            async with Client(self._authority_url, auth="oauth") as client:
                result = await client.call_tool(
                    "authority_check_balance",
                    {
                        "npub": self._operator_npub,
                        "proof": self._make_proof("check_balance"),
                    },
                )
        except Exception as e:
            raise AuthorityCertifyError(
                f"Failed to check balance at Authority: {e}"
            ) from e

        return self._parse_balance(result)

    def _parse_balance(self, result: Any) -> dict[str, Any]:
        """Extract the balance dict from the MCP tool result."""
        import json

        if isinstance(result, list):
            for block in result:
                if hasattr(block, "text"):
                    try:
                        data = json.loads(block.text)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(data, dict) and "success" in data:
                        return data

        if hasattr(result, "content") and isinstance(result.content, list):
            return self._parse_balance(result.content)

        if isinstance(result, dict):
            return result

        return {"success": False, "error": f"Unexpected response: {result}"}
