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

``OPERATOR_OBSOLETE_PRACTICES`` lists patterns that agents should stop
attempting.  Operators should surface these in ``session_status``
responses or a dedicated ``get_practices`` tool so that AI agents get
a proactive "unlearn this" signal at session start.  Operators may
extend with their own entries::

    from tollbooth.operator_protocol import OPERATOR_OBSOLETE_PRACTICES

    my_obsolete = OPERATOR_OBSOLETE_PRACTICES + [
        ObsoletePractice(
            pattern="my_old_tool()",
            replaced_by="my_new_tool()",
            reason="Replaced in v2.0.",
            deprecated_since="2026-04-01",
        ),
    ]
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tollbooth.actor_types import ActorRole, ObsoletePractice, ToolPath, ToolPathInfo


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
            "credentials are active or onboarding is needed. "
            "See OPERATOR_OBSOLETE_PRACTICES for patterns to avoid."
        ),
        supersedes=(
            "Replaces activate_session(passphrase). Passphrase-based "
            "session activation has been removed. Use the Secure "
            "Courier flow: request_credential_channel + receive_credentials."
        ),
    ),
    ToolPathInfo(
        tool_name="request_credential_channel",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Open a Secure Courier channel. Sends a welcome DM with "
            "credential template instructions to the patron's Nostr "
            "client. IMPORTANT: Always ask the patron which npub they "
            "want to use for THIS specific service and role. Do NOT "
            "reuse an npub from earlier in the conversation or from "
            "a different service — a patron may hold different npubs "
            "for different roles (patron vs operator vs authority) and "
            "different services."
        ),
        supersedes=(
            "Replaces register_credentials(api_key, brain_id, passphrase). "
            "Credentials must never appear in the chat window. The "
            "Secure Courier delivers them via encrypted Nostr DMs."
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
            "optional credential_card parameter to bypass the relay flow. "
            "IMPORTANT: The sender_npub must be the patron's chosen "
            "npub for this service. Do NOT substitute an npub from a "
            "different service or conversation context — each npub is "
            "a key on the patron's keyring, and the wrong key won't "
            "open this door."
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


# ── Obsolete practices ────────────────────────────────────────────
#
# Patterns that agents should stop attempting.  Surface these in
# session_status responses or a dedicated get_practices tool so
# agents get a proactive "unlearn this" signal at session start.
#
# Operators: extend with your own entries for service-specific
# deprecated patterns.

OPERATOR_OBSOLETE_PRACTICES: list[ObsoletePractice] = [
    ObsoletePractice(
        pattern="activate_session(passphrase)",
        replaced_by=(
            "receive_credentials(sender_npub) via the Secure Courier flow. "
            "Call session_status to check state, then "
            "request_credential_channel + receive_credentials if needed."
        ),
        reason=(
            "Passphrase-based session activation has been removed. "
            "Credentials are now delivered via encrypted Nostr DMs "
            "(Secure Courier) and never appear in the chat window."
        ),
        deprecated_since="2026-03-01",
    ),
    ObsoletePractice(
        pattern="register_credentials(api_key, brain_id, passphrase)",
        replaced_by=(
            "request_credential_channel(recipient_npub) to open a Secure "
            "Courier channel, then receive_credentials(sender_npub) to "
            "pick up the encrypted credentials from the Nostr relay."
        ),
        reason=(
            "Typing credentials into the chat window is a security risk. "
            "The Secure Courier delivers them via NIP-44 encrypted Nostr "
            "DMs so they never appear in conversation history."
        ),
        deprecated_since="2026-03-01",
    ),
    ObsoletePractice(
        pattern="Reusing an npub from a different service or conversation context",
        replaced_by=(
            "Always ask the patron which npub they want to use for THIS "
            "specific service and role. A patron may hold different npubs "
            "for different roles (patron, operator, authority) and "
            "different services. Each npub is a key on the patron's "
            "keyring — do not borrow keys from other doors."
        ),
        reason=(
            "One human may be a patron to several MCPs, each with "
            "different API credentials associated to unique npubs. "
            "Sometimes one npub may be reused across credentials, but "
            "this is the patron's choice, not the agent's assumption."
        ),
        deprecated_since="2026-03-04",
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
