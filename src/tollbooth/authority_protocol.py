"""Protocol defining the AuthorityProtocol — certification and operator ledger.

The Authority is the institutional backbone of the Tollbooth ecosystem.
It registers MCP operators, collects a certification fee via Bitcoin
Lightning, and issues signed certificates proving an operator has paid.

``AUTHORITY_BASE_CATALOG`` is the library-level default tool catalog.
Authority implementations inherit it and extend with custom tools.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tollbooth.actor_types import ActorRole, ToolPath, ToolPathInfo


# ── Library-level base catalog ────────────────────────────────────────

AUTHORITY_BASE_CATALOG: list[ToolPathInfo] = [
    # ── Hot-path (local ledger) ────────────────────────────────────
    ToolPathInfo(
        tool_name="certify_credits",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Core product — certify a credit purchase for an operator. "
            "Deducts certification fee and returns a Schnorr-signed "
            "Nostr event certificate."
        ),
    ),
    ToolPathInfo(
        tool_name="register_operator",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Provision a new operator in the Authority ledger. Called by "
            "the Authority itself after receiving and approving a Nostr "
            "DM delegation request from a requester npub. Creates the "
            "operator's ledger entry with a zero balance. This is the "
            "Authority-side counterpart of the Operator's "
            "register_operator delegation tool."
        ),
    ),
    ToolPathInfo(
        tool_name="operator_status",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Return the calling operator's registration info and "
            "the Authority's Nostr npub for certificate verification."
        ),
    ),
    ToolPathInfo(
        tool_name="check_balance",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Return the calling operator's credit balance.",
    ),
    ToolPathInfo(
        tool_name="account_statement",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Return the calling operator's transaction history.",
    ),
    ToolPathInfo(
        tool_name="account_statement_infographic",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Return a visual summary of the operator's account.",
    ),
    ToolPathInfo(
        tool_name="service_status",
        path=ToolPath.HOT,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Return the Authority's health and version info.",
    ),
    ToolPathInfo(
        tool_name="report_upstream_purchase",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Admin-only: record an upstream credit purchase.",
    ),
    # ── Cold-path (BTCPay) ─────────────────────────────────────────
    ToolPathInfo(
        tool_name="purchase_credits",
        path=ToolPath.COLD,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Create a Lightning invoice for credit purchase.",
    ),
    ToolPathInfo(
        tool_name="check_payment",
        path=ToolPath.COLD,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Poll a Lightning invoice for settlement status.",
    ),
    # ── Cold-path (registry) ───────────────────────────────────────
    ToolPathInfo(
        tool_name="check_dpyc_membership",
        path=ToolPath.COLD,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Check whether an npub is a registered DPYC member.",
    ),
]


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
        """(hot) Provision a new operator in the ledger.

        Called after the Authority receives and approves a Nostr DM
        delegation request from a requester npub.
        """
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
