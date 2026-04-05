"""Tool identity layer — stable UUIDs for capability-based tool pricing.

Each tool capability gets a deterministic UUID v5 derived from a fixed
DPYC namespace and a canonical capability name.  The pricing model in
Neon references these UUIDs, not MCP tool names, so a pricing model
can work across different MCPs that implement the same capability.

Code declares capability + category + intent.  Neon owns the economics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

# Fixed namespace for UUID v5 generation — never change this.
DPYC_NAMESPACE = uuid.UUID("d9a3f1c7-4e2b-4a8f-b6d5-1c3e7f9a2b4d")


def capability_uuid(capability: str) -> str:
    """Derive a deterministic UUID v5 from a canonical capability name."""
    return str(uuid.uuid5(DPYC_NAMESPACE, capability))


@dataclass(frozen=True)
class ToolIdentity:
    """Declares what a tool IS — not what it costs.

    Attributes:
        capability: Canonical name (e.g. ``"check_balance"``).
            Two MCPs implementing the same capability use the same name
            and therefore converge on the same UUID.
        category: Access/billing category.
            ``"free"`` and ``"restricted"`` are gated by the runtime
            without consulting Neon.  All others (``"read"``, ``"write"``,
            ``"heavy"``, etc.) require a price entry in Neon.
        intent: Human-readable purpose.
    """

    capability: str
    category: str  # "free" | "read" | "write" | "heavy" | "restricted"
    intent: str

    @property
    def tool_id(self) -> str:
        """Deterministic UUID v5 for this capability."""
        return capability_uuid(self.capability)


# ======================================================================
# Standard identities for all wheel-provided tools.
#
# Keyed by UUID (tool_id). Every operator inherits these via the wheel.
# ======================================================================

_STANDARD_LIST: list[ToolIdentity] = [
    # -- Credit tools --
    ToolIdentity(capability="check_balance", category="free",
                 intent="Check a patron's credit balance."),
    ToolIdentity(capability="purchase_credits", category="free",
                 intent="Buy credits via Bitcoin Lightning."),
    ToolIdentity(capability="check_payment", category="free",
                 intent="Check payment status of a Lightning invoice."),
    ToolIdentity(capability="restore_credits", category="free",
                 intent="Restore credits from a previously paid invoice."),
    ToolIdentity(capability="account_statement", category="free",
                 intent="Generate a patron's account statement."),
    ToolIdentity(capability="account_statement_infographic", category="read",
                 intent="Generate a visual SVG infographic of account statement."),

    # -- Service status --
    ToolIdentity(capability="service_status", category="free",
                 intent="Check health and configuration of this service."),

    # -- Onboarding --
    ToolIdentity(capability="get_operator_onboarding_status", category="free",
                 intent="Report operator configuration readiness."),
    ToolIdentity(capability="get_patron_onboarding_status", category="free",
                 intent="Report a patron's credential readiness."),
    ToolIdentity(capability="session_status", category="free",
                 intent="Check operator lifecycle state and readiness."),

    # -- Secure Courier --
    ToolIdentity(capability="request_credential_channel", category="free",
                 intent="Open a Secure Courier channel for credential delivery."),
    ToolIdentity(capability="receive_credentials", category="free",
                 intent="Pick up credentials from the Secure Courier."),
    ToolIdentity(capability="forget_credentials", category="free",
                 intent="Delete vaulted credentials for a service."),
    ToolIdentity(capability="request_patron_credentials", category="free",
                 intent="Open a Secure Courier channel for patron credentials."),
    ToolIdentity(capability="receive_patron_credentials", category="free",
                 intent="Pick up patron credentials from the Secure Courier."),

    # -- Oracle delegation --
    # Oracle tools are free and use the oracle_ namespace. They don't
    # need pricing entries or registry identities — they're never gated.

    # -- Authority delegation --
    ToolIdentity(capability="check_authority_balance", category="free",
                 intent="Check operator's tax balance at the Authority."),

    # -- Pricing CRUD --
    ToolIdentity(capability="get_pricing_model", category="free",
                 intent="Get the active pricing model."),
    ToolIdentity(capability="set_pricing_model", category="restricted",
                 intent="Set the active pricing model. Operator only."),
    ToolIdentity(capability="reset_pricing_model", category="restricted",
                 intent="Delete all pricing models and re-initialize. Operator only."),
    ToolIdentity(capability="check_price", category="free",
                 intent="Preview the effective cost of a tool call."),
    ToolIdentity(capability="list_constraint_types", category="free",
                 intent="List available constraint types and parameter schemas."),

    # -- OTS notarization --
    ToolIdentity(capability="notarize_ledger", category="restricted",
                 intent="Submit patron balance Merkle root to Bitcoin via OTS."),
    ToolIdentity(capability="get_notarization_proof", category="free",
                 intent="Generate a Merkle inclusion proof for a patron balance."),
    ToolIdentity(capability="list_notarizations", category="free",
                 intent="List recent Bitcoin notarization records."),
]

# Build the UUID-keyed dict from the list.
STANDARD_IDENTITIES: dict[str, ToolIdentity] = {
    ti.tool_id: ti for ti in _STANDARD_LIST
}
