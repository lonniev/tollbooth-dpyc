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
        capability: Canonical short name (e.g. ``"check_balance"``).
            Used only for UUID derivation — two MCPs implementing the
            same capability converge on the same UUID.
        category: Access/billing category.
            ``"free"`` and ``"restricted"`` are gated by the runtime
            without consulting Neon.  All others (``"read"``, ``"write"``,
            ``"heavy"``, etc.) require a price entry in Neon.
        intent: Human-readable purpose.
        mcp_name: Full namespace-scoped MCP tool name
            (e.g. ``"taxsort_check_balance"``).  Set during
            ``register_standard_tools`` when the slug is known.
            This is the ONE display name used everywhere outside
            the server process.
    """

    capability: str
    category: str  # "free" | "read" | "write" | "heavy" | "restricted"
    intent: str
    mcp_name: str = ""

    # Pricing hints — seed defaults for _build_initial_pricing_model.
    # Neon pricing model is the runtime source of truth.
    pricing_hint_type: str = "flat"    # "flat" | "percent"
    pricing_hint_value: int = 0        # flat sats or percent rate
    pricing_hint_param: str = ""       # kwarg name for percent base (e.g. "amount_sats")
    pricing_hint_min: int = 0          # floor in sats
    # Categorical-multiplier hint for tools whose price scales by enum-
    # valued kwargs (e.g. Optionality's deal_scenario: ``difficulty`` ×
    # ``mode``). Shape: ``(("difficulty", (("apprentice", 1.0), …)), …)``.
    # Frozen tuple form keeps the dataclass hashable.
    pricing_hint_multipliers: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()

    @property
    def tool_id(self) -> str:
        """Deterministic UUID v5 for this capability."""
        return capability_uuid(self.capability)

    @property
    def display_name(self) -> str:
        """The human-visible tool name — always the full MCP name."""
        return self.mcp_name or self.capability


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
    ToolIdentity(capability="restore_credits", category="restricted",
                 intent="Operator-only: credit a patron's ledger from a "
                        "BTCPay-settled invoice. Discretionary recovery for "
                        "support escalations and infrastructure incidents."),
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

    # -- OAuth2 (conditional — only if oauth_provider is configured) --
    ToolIdentity(capability="begin_oauth", category="free",
                 intent="Start OAuth2 authorization flow."),
    ToolIdentity(capability="check_oauth_status", category="free",
                 intent="Check OAuth2 authorization status."),

    # -- Npub ownership proof --
    ToolIdentity(capability="request_npub_proof", category="free",
                 intent="Request npub ownership proof from patron via DM."),
    ToolIdentity(capability="receive_npub_proof", category="free",
                 intent="Receive and cache npub ownership proof."),
    ToolIdentity(capability="check_proof_status", category="free",
                 intent="Read-only check of a proof_token's remaining validity."),

    # -- Oracle delegation --
    # Wire-exposed under the operator's `_oracle_` namespace (e.g.
    # `optionality_oracle_about`). Routed through a separate free
    # decorator that bypasses pricing/gating entirely, so the price
    # here is informational only — the wheel does not consult it at
    # call time. Included in STANDARD_IDENTITIES anyway so the
    # initial pricing model is a complete inventory of what's exposed
    # (otherwise the Studio's Reconcile flow keeps offering them as
    # "new tools" on every reset).
    ToolIdentity(capability="oracle_about", category="free",
                 intent="Describe the DPYC ecosystem via the Oracle."),
    ToolIdentity(capability="oracle_get_tax_rate", category="free",
                 intent="Get the current DPYC certification tax rate."),
    ToolIdentity(capability="oracle_how_to_join", category="free",
                 intent="Get DPYC onboarding instructions from the Oracle."),
    ToolIdentity(capability="oracle_lookup_member", category="free",
                 intent="Look up a DPYC community member by npub."),
    ToolIdentity(capability="oracle_network_advisory", category="free",
                 intent="Get active network advisories from the Oracle."),

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

    # -- Patron credential CRUD --
    ToolIdentity(capability="update_patron_credential", category="free",
                 intent="Add or update a single patron credential field."),
    ToolIdentity(capability="delete_patron_credential", category="free",
                 intent="Remove a single patron credential field."),
    ToolIdentity(capability="get_patron_credential_fields", category="free",
                 intent="List stored patron credential field names."),

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
