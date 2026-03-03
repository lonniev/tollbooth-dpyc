"""Protocol defining the OperatorProtocol — user ledger and delegation.

The Operator runs an MCP service and manages patron (user) credit
balances. It delegates certification to the Authority and community
queries to the Oracle, either directly or via the Authority.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tollbooth.actor_types import ActorRole, ToolPath, ToolPathInfo


@runtime_checkable
class OperatorProtocol(Protocol):
    """Contract for a DPYC Operator MCP server.

    Hot-path tools operate on the local patron ledger.
    Cold-path tools reach BTCPay or delegate to Authority/Oracle.

    Operator methods take explicit ``npub`` because the patron is
    always identified by their Nostr public key.
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

    async def check_balance(self, npub: str) -> dict[str, Any]:
        """(hot) Return the patron's credit balance."""
        ...

    async def account_statement(self, npub: str) -> dict[str, Any]:
        """(hot) Return the patron's transaction history."""
        ...

    async def account_statement_infographic(
        self, npub: str
    ) -> dict[str, Any]:
        """(hot) Return a visual summary of the patron's account."""
        ...

    async def restore_credits(
        self, npub: str, invoice_id: str
    ) -> dict[str, Any]:
        """(hot) Restore credits from a previously paid invoice."""
        ...

    async def service_status(self) -> dict[str, Any]:
        """(hot) Return the Operator's health and version info."""
        ...

    # ── Cold-path (BTCPay) ───────────────────────────────────────

    async def purchase_credits(
        self, npub: str, amount_sats: int, certificate: str
    ) -> dict[str, Any]:
        """(cold) Create a Lightning invoice for patron credit purchase."""
        ...

    async def check_payment(
        self, npub: str, invoice_id: str
    ) -> dict[str, Any]:
        """(cold) Poll a Lightning invoice for settlement status."""
        ...

    # ── Cold-path (delegates to Authority) ───────────────────────

    async def certify_credits(
        self, operator_id: str, amount_sats: int
    ) -> dict[str, Any]:
        """(cold, delegates to Authority) Certify a credit purchase."""
        ...

    async def register_operator(self, npub: str) -> dict[str, Any]:
        """(cold, delegates to Authority) Register as an operator."""
        ...

    async def operator_status(self) -> dict[str, Any]:
        """(cold, delegates to Authority) Get operator registration info."""
        ...

    # ── Cold-path (delegates via Authority to Oracle) ────────────

    async def lookup_member(self, npub: str) -> dict[str, Any] | str:
        """(cold, delegates to Oracle) Look up a DPYC member."""
        ...

    async def how_to_join(self) -> str:
        """(cold, delegates to Oracle) Get onboarding instructions."""
        ...

    async def get_tax_rate(self) -> dict[str, Any]:
        """(cold, delegates to Oracle) Get the current tax rate."""
        ...

    async def about(self) -> str:
        """(cold, delegates to Oracle) Describe the DPYC ecosystem."""
        ...

    async def network_advisory(self) -> str:
        """(cold, delegates to Oracle) Get active network advisories."""
        ...
