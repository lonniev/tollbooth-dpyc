"""Protocol defining the OperatorProtocol — user ledger and delegation.

The Operator runs an MCP service and manages patron (user) credit
balances. It delegates certification to the Authority and community
queries to the Oracle, either directly or via the Authority.

``OPERATOR_BASE_CATALOG`` is the library-level default tool catalog.
Operators inherit it and extend with domain-specific tools::

    from tollbooth.operator_protocol import OPERATOR_BASE_CATALOG

    class MyOperator:
        @classmethod
        def tool_catalog(cls):
            return OPERATOR_BASE_CATALOG + [
                ToolPathInfo(tool_name="my_tool", path=ToolPath.HOT, ...),
            ]
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tollbooth.actor_types import ActorRole, ToolPath, ToolPathInfo


# ── Library-level base catalog ────────────────────────────────────────
#
# Every Operator inherits these tools.  The agent_hint field is the
# single source of truth for MCP tool descriptions across all operators.

OPERATOR_BASE_CATALOG: list[ToolPathInfo] = [
    # ── Hot-path (local ledger) ────────────────────────────────────
    ToolPathInfo(
        tool_name="check_balance",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Return the patron's credit balance.",
    ),
    ToolPathInfo(
        tool_name="account_statement",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Return the patron's transaction history.",
    ),
    ToolPathInfo(
        tool_name="account_statement_infographic",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="READ",
        agent_hint="Return a visual summary of the patron's account.",
    ),
    ToolPathInfo(
        tool_name="restore_credits",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Restore credits from a previously paid invoice.",
    ),
    ToolPathInfo(
        tool_name="service_status",
        path=ToolPath.HOT,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Return the Operator's health and version info.",
    ),
    # ── Hot-path (Secure Courier) ──────────────────────────────────
    ToolPathInfo(
        tool_name="session_status",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Check the patron's current session state. Shows whether "
            "credentials are active or onboarding is needed."
        ),
    ),
    ToolPathInfo(
        tool_name="request_credential_channel",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Open a Secure Courier channel. Sends a welcome DM with "
            "credential template instructions to the patron's Nostr client."
        ),
    ),
    ToolPathInfo(
        tool_name="receive_credentials",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Pick up credentials from the Secure Courier. Checks vault "
            "first (instant), then polls Nostr relays. On first-time "
            "receipt, sends the credential card (ncred1...) back to the "
            "patron via Nostr DM for scan-and-paste reuse. Accepts an "
            "optional credential_card parameter to bypass the relay flow."
        ),
    ),
    ToolPathInfo(
        tool_name="forget_credentials",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Delete vaulted credentials for key rotation. The patron "
            "must re-deliver via Secure Courier after this call."
        ),
    ),
    # ── Delegation (BTCPay via Authority) ──────────────────────────
    ToolPathInfo(
        tool_name="purchase_credits",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.AUTHORITY,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Create a Lightning invoice for patron credit purchase.",
    ),
    ToolPathInfo(
        tool_name="check_payment",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.AUTHORITY,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Poll a Lightning invoice for settlement status.",
    ),
    # ── Delegation (Authority) ─────────────────────────────────────
    ToolPathInfo(
        tool_name="certify_credits",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.AUTHORITY,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Certify a credit purchase via the Authority.",
    ),
    ToolPathInfo(
        tool_name="register_operator",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.AUTHORITY,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Register as an operator via the Authority.",
    ),
    ToolPathInfo(
        tool_name="operator_status",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.AUTHORITY,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint="Get operator registration info from the Authority.",
    ),
    # ── Delegation (Oracle — direct routing) ───────────────────────
    ToolPathInfo(
        tool_name="lookup_member",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.ORACLE,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Look up a DPYC member via the Oracle.",
    ),
    ToolPathInfo(
        tool_name="how_to_join",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.ORACLE,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Get onboarding instructions from the Oracle.",
    ),
    ToolPathInfo(
        tool_name="get_tax_rate",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.ORACLE,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Get the current tax rate from the Oracle.",
    ),
    ToolPathInfo(
        tool_name="about",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.ORACLE,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Describe the DPYC ecosystem via the Oracle.",
    ),
    ToolPathInfo(
        tool_name="network_advisory",
        path=ToolPath.DELEGATION,
        delegates_to=ActorRole.ORACLE,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="Get active network advisories from the Oracle.",
    ),
]


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

    # ── Hot-path (Secure Courier) ────────────────────────────────

    async def session_status(self) -> dict[str, Any]:
        """(hot) Check the patron's current session state."""
        ...

    async def request_credential_channel(
        self, service: str, greeting: str, recipient_npub: str | None,
    ) -> dict[str, Any]:
        """(hot) Open a Secure Courier channel for credential delivery."""
        ...

    async def receive_credentials(
        self, sender_npub: str, service: str, credential_card: str,
    ) -> dict[str, Any]:
        """(hot) Pick up credentials from the Secure Courier."""
        ...

    async def forget_credentials(
        self, sender_npub: str, service: str,
    ) -> dict[str, Any]:
        """(hot) Delete vaulted credentials for key rotation."""
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
