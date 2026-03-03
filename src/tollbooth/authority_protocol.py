"""Protocol defining the AuthorityProtocol — certification and operator ledger.

The Authority is the institutional backbone of the Tollbooth ecosystem.
It registers MCP operators, collects a certification fee via Bitcoin
Lightning, and issues signed certificates proving an operator has paid.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tollbooth.actor_types import ActorRole, ToolPath, ToolPathInfo


@runtime_checkable
class AuthorityProtocol(Protocol):
    """Contract for a DPYC Authority MCP server.

    Hot-path tools operate on the local operator ledger.
    Cold-path tools reach BTCPay or the DPYC registry.
    """

    @property
    def slug(self) -> str:
        """Short identifier for tool-name prefixing."""
        ...

    @classmethod
    def tool_catalog(cls) -> list[ToolPathInfo]:
        """Return metadata for every tool this actor exposes."""
        ...

    # ── Hot-path (local ledger) ──────────────────────────────────

    async def certify_credits(
        self, operator_id: str, amount_sats: int
    ) -> dict[str, Any]:
        """(hot) Core product — certify a credit purchase for an operator."""
        ...

    async def register_operator(self, npub: str) -> dict[str, Any]:
        """(hot) Register a new operator in the ledger."""
        ...

    async def operator_status(self) -> dict[str, Any]:
        """(hot) Return the calling operator's registration info."""
        ...

    async def check_balance(self) -> dict[str, Any]:
        """(hot) Return the calling operator's credit balance."""
        ...

    async def account_statement(self) -> dict[str, Any]:
        """(hot) Return the calling operator's transaction history."""
        ...

    async def account_statement_infographic(self) -> dict[str, Any]:
        """(hot) Return a visual summary of the operator's account."""
        ...

    async def service_status(self) -> dict[str, Any]:
        """(hot) Return the Authority's health and version info."""
        ...

    async def report_upstream_purchase(
        self, amount_sats: int
    ) -> dict[str, Any]:
        """(hot) Admin-only: record an upstream credit purchase."""
        ...

    # ── Cold-path (BTCPay) ───────────────────────────────────────

    async def purchase_credits(self, amount_sats: int) -> dict[str, Any]:
        """(cold) Create a Lightning invoice for credit purchase."""
        ...

    async def check_payment(self, invoice_id: str) -> dict[str, Any]:
        """(cold) Poll a Lightning invoice for settlement status."""
        ...

    # ── Cold-path (registry) ─────────────────────────────────────

    async def check_dpyc_membership(self, npub: str) -> dict[str, Any]:
        """(cold) Check whether an npub is a registered DPYC member."""
        ...
