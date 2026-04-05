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
# Keyed by the short function name used in register_standard_tools().
# Every operator inherits these via the wheel.
# ======================================================================

STANDARD_IDENTITIES: dict[str, ToolIdentity] = {
    # -- Credit tools --
    "check_balance": ToolIdentity(
        capability="check_balance",
        category="free",
        intent="Check a patron's credit balance.",
    ),
    "purchase_credits": ToolIdentity(
        capability="purchase_credits",
        category="free",
        intent="Buy credits via Bitcoin Lightning.",
    ),
    "check_payment": ToolIdentity(
        capability="check_payment",
        category="free",
        intent="Check payment status of a Lightning invoice.",
    ),
    "restore_credits": ToolIdentity(
        capability="restore_credits",
        category="free",
        intent="Restore credits from a previously paid invoice.",
    ),
    "account_statement": ToolIdentity(
        capability="account_statement",
        category="free",
        intent="Generate a patron's account statement.",
    ),
    "account_statement_infographic": ToolIdentity(
        capability="account_statement_infographic",
        category="read",
        intent="Generate a visual SVG infographic of account statement.",
    ),

    # -- Service status --
    "service_status": ToolIdentity(
        capability="service_status",
        category="free",
        intent="Check health and configuration of this service.",
    ),

    # -- Onboarding --
    "get_operator_onboarding_status": ToolIdentity(
        capability="get_operator_onboarding_status",
        category="free",
        intent="Report operator configuration readiness.",
    ),
    "get_patron_onboarding_status": ToolIdentity(
        capability="get_patron_onboarding_status",
        category="free",
        intent="Report a patron's credential readiness.",
    ),
    "session_status": ToolIdentity(
        capability="session_status",
        category="free",
        intent="Check operator lifecycle state and readiness.",
    ),

    # -- Secure Courier --
    "request_credential_channel": ToolIdentity(
        capability="request_credential_channel",
        category="free",
        intent="Open a Secure Courier channel for credential delivery.",
    ),
    "receive_credentials": ToolIdentity(
        capability="receive_credentials",
        category="free",
        intent="Pick up credentials from the Secure Courier.",
    ),
    "forget_credentials": ToolIdentity(
        capability="forget_credentials",
        category="free",
        intent="Delete vaulted credentials for a service.",
    ),
    "request_patron_credentials": ToolIdentity(
        capability="request_patron_credentials",
        category="free",
        intent="Open a Secure Courier channel for patron credentials.",
    ),
    "receive_patron_credentials": ToolIdentity(
        capability="receive_patron_credentials",
        category="free",
        intent="Pick up patron credentials from the Secure Courier.",
    ),

    # -- Oracle delegation --
    "how_to_join": ToolIdentity(
        capability="how_to_join",
        category="free",
        intent="Get DPYC onboarding instructions.",
    ),
    "get_tax_rate": ToolIdentity(
        capability="get_tax_rate",
        category="free",
        intent="Get the current DPYC certification tax rate.",
    ),
    "lookup_member": ToolIdentity(
        capability="lookup_member",
        category="free",
        intent="Look up a DPYC community member by npub.",
    ),
    "about": ToolIdentity(
        capability="about",
        category="free",
        intent="Describe the DPYC ecosystem.",
    ),
    "network_advisory": ToolIdentity(
        capability="network_advisory",
        category="free",
        intent="Get active network advisories.",
    ),

    # -- Authority delegation --
    "check_authority_balance": ToolIdentity(
        capability="check_authority_balance",
        category="free",
        intent="Check operator's tax balance at the Authority.",
    ),

    # -- Pricing CRUD --
    "get_pricing_model": ToolIdentity(
        capability="get_pricing_model",
        category="free",
        intent="Get the active pricing model.",
    ),
    "set_pricing_model": ToolIdentity(
        capability="set_pricing_model",
        category="restricted",
        intent="Set the active pricing model. Operator only.",
    ),
    "reset_pricing_model": ToolIdentity(
        capability="reset_pricing_model",
        category="restricted",
        intent="Delete all pricing models and re-initialize. Operator only.",
    ),
    "check_price": ToolIdentity(
        capability="check_price",
        category="free",
        intent="Preview the effective cost of a tool call.",
    ),
    "list_constraint_types": ToolIdentity(
        capability="list_constraint_types",
        category="free",
        intent="List available constraint types and parameter schemas.",
    ),

    # -- OTS notarization --
    "notarize_ledger": ToolIdentity(
        capability="notarize_ledger",
        category="restricted",
        intent="Submit patron balance Merkle root to Bitcoin via OTS.",
    ),
    "get_notarization_proof": ToolIdentity(
        capability="get_notarization_proof",
        category="free",
        intent="Generate a Merkle inclusion proof for a patron balance.",
    ),
    "list_notarizations": ToolIdentity(
        capability="list_notarizations",
        category="free",
        intent="List recent Bitcoin notarization records.",
    ),
}
