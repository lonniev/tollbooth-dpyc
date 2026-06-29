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

import asyncio
import logging
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
    # ── Hot-path (Bitcoin Notarization) ────────────────────────────
    ToolPathInfo(
        tool_name="notarize_ledger",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="RESTRICTED",
        agent_hint=(
            "Create a Bitcoin-notarized snapshot of all patron balances. "
            "Builds a Merkle tree and submits the root to OpenTimestamps "
            "calendars. Bitcoin confirmation in 1-6 hours. Operator-only."
        ),
    ),
    ToolPathInfo(
        tool_name="get_notarization_proof",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="READ",
        agent_hint=(
            "Get a verifiable Merkle inclusion proof that a patron's "
            "balance was included in a Bitcoin-notarized snapshot."
        ),
    ),
    ToolPathInfo(
        tool_name="list_notarizations",
        path=ToolPath.HOT,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint="List recent Bitcoin notarization records.",
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
            "Courier flow: request_credential_channel + "
            "receive_credentials(sender_npub, service, dpop_token)."
        ),
    ),
    ToolPathInfo(
        tool_name="get_operator_onboarding_status",
        path=ToolPath.HOT,
        requires_auth=False,
        cost_tier="FREE",
        agent_hint=(
            "Report the operator's configuration readiness. Returns which "
            "operator settings are configured, which are missing, and how "
            "to deliver each. For patron-level credential status, use "
            "get_patron_onboarding_status instead."
        ),
    ),
    ToolPathInfo(
        tool_name="get_patron_onboarding_status",
        path=ToolPath.HOT,
        requires_auth=True,
        cost_tier="FREE",
        agent_hint=(
            "Report a specific patron's credential readiness. Returns "
            "which patron secrets (API keys, tokens) are configured and "
            "which are missing. For services using OAuth2, reports that "
            "no patron credentials are needed. Call this to determine "
            "whether a patron needs to deliver credentials via Secure "
            "Courier before they can use the service."
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
        agent_hint=(
            "Request operator registration under a specific Authority. "
            "Sends a delegation request via Nostr DM to the specified "
            "Authority npub. This tool does NOT call the Oracle — the "
            "Authority receives and approves the request, then provisions "
            "the new Operator itself. Requires both authority_npub (who "
            "you are asking) and requester_npub (who wants to become an "
            "Operator). Not to be confused with citizen registration, "
            "which the Operator handles directly via the Oracle's "
            "request_citizenship / confirm_citizenship flow."
        ),
        # NOTE: DM-send logic — the Nostr DM to the Authority should
        # live in tollbooth-dpyc as a shared utility alongside the
        # existing Secure Courier infrastructure (NIP-44 encryption,
        # relay publishing).  A candidate location is a new
        # `operator_registration.py` module or an extension of
        # `nostr_courier.py`.  Until implemented, agents should
        # describe the intended DM but cannot send it programmatically.
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
            "receive_credentials(sender_npub, service, dpop_token) via the Secure "
            "Courier flow. Call session_status to check state, then "
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
            "Courier channel, then receive_credentials(sender_npub, service, "
            "dpop_token) to pick up the encrypted credentials from the Nostr relay."
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

    async def get_operator_onboarding_status(self) -> dict[str, Any]:
        """(hot) Report the operator's configuration readiness."""
        ...

    async def get_patron_onboarding_status(self, patron_npub: str) -> dict[str, Any]:
        """(hot) Report a patron's credential readiness."""
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
        self, sender_npub: str, service: str, dpop_token: str,
        credential_card: str,
    ) -> dict[str, Any]:
        """(hot) Pick up credentials from the Secure Courier.

        ``dpop_token`` is the session phrase returned by
        ``request_credential_channel`` — required for the deterministic,
        pinned-relay drain (or omit it when redeeming a ``credential_card``).
        """
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
        """(cold, delegates to Authority) Request operator registration.

        Sends a delegation request via Nostr DM to the Authority npub.
        The Authority provisions the new Operator upon approval.  This
        tool does NOT invoke the Oracle directly — the Authority does.
        """
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


# ── Catalog conformance validation ────────────────────────────────

_conformance_logger = logging.getLogger("tollbooth.conformance")


class OperatorConformanceError(Exception):
    """Raised when strict validation finds missing base-catalog tools."""


def _check_catalog(
    registered_names: set[str],
    slug: str,
    catalog: list[ToolPathInfo] | None = None,
    strict: bool = False,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Compare *registered_names* against the base catalog.

    Returns the list of missing tool names (from the catalog perspective).
    Checks for both ``{slug}_{tool_name}`` and bare ``{tool_name}``.
    """
    if catalog is None:
        catalog = OPERATOR_BASE_CATALOG
    if logger is None:
        logger = _conformance_logger

    missing: list[str] = []
    for entry in catalog:
        prefixed = f"{slug}_{entry.tool_name}"
        if prefixed not in registered_names and entry.tool_name not in registered_names:
            missing.append(entry.tool_name)

    if missing:
        msg = (
            f"Operator '{slug}' is missing {len(missing)} base-catalog "
            f"tool(s): {', '.join(missing)}"
        )
        if strict:
            raise OperatorConformanceError(msg)
        logger.warning(msg)

    return missing


def validate_operator_tools(
    mcp_server: Any,
    slug: str,
    catalog: list[ToolPathInfo] | None = None,
    strict: bool = False,
) -> list[str]:
    """Synchronous catalog validation — call before ``mcp.run()``.

    Introspects the FastMCP server's registered tools and checks them
    against the base catalog (or a custom *catalog*).

    Returns the list of missing tool names (empty if fully conformant).
    Raises :class:`OperatorConformanceError` when *strict* is True and
    tools are missing.
    """
    tools = asyncio.run(mcp_server.list_tools())
    registered = {t.name for t in tools}
    return _check_catalog(registered, slug, catalog, strict)


async def async_validate_operator_tools(
    mcp_server: Any,
    slug: str,
    catalog: list[ToolPathInfo] | None = None,
    strict: bool = False,
) -> list[str]:
    """Async catalog validation — for lifespan hooks or async startup."""
    tools = await mcp_server.list_tools()
    registered = {t.name for t in tools}
    return _check_catalog(registered, slug, catalog, strict)
