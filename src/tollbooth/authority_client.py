"""Server-to-server MCP client for obtaining Authority certificates.

Used by operator servers (thebrain-mcp, excalibur-mcp) to auto-certify
credit purchases without requiring the AI agent to call the Authority directly.
"""

from __future__ import annotations

import logging
from typing import Any

from tollbooth.patron_signer import PatronSigner

try:
    from fastmcp import Client  # type: ignore[import-untyped]
except ImportError:
    Client = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class AuthorityCertifyError(Exception):
    """Raised when Authority certification fails.

    Carries the Authority's structured ``error_code`` when the failure was a
    tool-level refusal (e.g. ``insufficient_balance`` when the *purchasing
    actor's* certification balance held at the Authority is exhausted — the
    ``certify_credits`` fee debits the caller's ledger, not the Authority's
    own funds), so callers can branch on the code rather than parsing prose.
    Empty for connection/transport failures.
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
        self._certify_tool_name = certify_tool_name
        # The one home for patron-side signing (shared with the agent keyring).
        self._signer = PatronSigner(operator_npub, operator_nsec)

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
                # The Authority's verify_proof gate binds the proof to the runtime
                # mcp_name (`<slug>_<func>`, e.g. "authority_certify_credits") since
                # wheel 0.24.0 — so we sign for the exact wire name we invoke.
                result = await client.call_tool(
                    self._certify_tool_name,
                    self._signer.authenticate(
                        self._certify_tool_name, {"amount_sats": amount_sats}
                    ),
                )
        except AuthorityCertifyError:
            raise
        except Exception as e:
            raise AuthorityCertifyError(
                f"Failed to connect to Authority at {self._authority_url}: {e}"
            ) from e

        return self._parse_result(result)

    async def report_neon_quota_exceeded(self, detail: str = "") -> dict[str, Any]:
        """Tell this operator's Authority that the operator's Neon books are
        402-locked (compute/storage quota exhausted).

        Best-effort telemetry: the Authority provisions and is responsible for
        the books, so it should learn the instant they lock — from the operator,
        not from a patron complaint. Signs for the exact wire name the same way
        ``certify_credits`` does. Returns the Authority's ack dict; raises
        ``AuthorityCertifyError`` on transport failure so the caller can decide
        (the caller runs this fire-and-forget, so a raise is only logged)."""
        if Client is None:
            raise AuthorityCertifyError("fastmcp package required for Authority alerts.")
        tool = "authority_receive_neon_402_alert"
        try:
            async with Client(self._authority_url, auth="oauth") as client:
                result = await client.call_tool(
                    tool, self._signer.authenticate(tool, {"detail": detail[:500]})
                )
        except Exception as e:
            raise AuthorityCertifyError(
                f"Failed to reach Authority at {self._authority_url}: {e}"
            ) from e
        # Best-effort — unwrap to a dict if we can, else return a bare ack.
        if hasattr(result, "data") and isinstance(result.data, dict):
            return result.data  # type: ignore[no-any-return]
        return {"success": True}

    def _parse_result(self, result: Any) -> dict[str, Any]:
        """Extract the certificate dict from the MCP tool result."""
        # Unwrap CallToolResult (fastmcp dataclass) — duck typing to avoid import
        if hasattr(result, "data") and isinstance(result.data, dict):
            result = result.data
        elif hasattr(result, "content") and isinstance(
            result.content, list
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
                # NOTE: historically signs for "check_balance" (not the wire name
                # "authority_check_balance"); preserved as-is by this refactor. See the
                # tool-name-mismatch follow-up.
                result = await client.call_tool(
                    "authority_check_balance",
                    self._signer.authenticate("check_balance"),
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
