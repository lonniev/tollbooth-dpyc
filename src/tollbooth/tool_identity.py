"""Tool identity layer — explicit UUIDs that survive renames.

Each tool's identity is a UUID string that the operator picks once at
tool birth and never changes. UUIDs are treated as opaque by the wheel:
not parsed, not derived from any other string. The pricing model in
Neon references these UUIDs as primary keys.

What this enables:
    The operator can rename a function (``def get_thought_by_name`` →
    ``def lookup_node``), rebrand a slug (``brain`` → ``personal-brain``),
    or change a display label — the UUID stays. Pricing-model rows
    never orphan from refactoring.

Historical note:
    Prior versions derived ``tool_id`` from ``capability_uuid(string)``
    where ``string`` was an in-code capability label. Renaming that
    label changed the UUID and orphaned every pricing-model row keyed
    by the old UUID. The fix is to declare UUIDs as frozen constants.
    ``capability_uuid()`` remains in this module as a one-time helper
    operators can call at the REPL to generate an initial UUID seed
    from a name they like — but the wheel itself never calls it.

Code declares tool_id + category + intent. Neon owns the economics.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

# Fixed namespace for UUID v5 generation — never change this.
DPYC_NAMESPACE = uuid.UUID("d9a3f1c7-4e2b-4a8f-b6d5-1c3e7f9a2b4d")


def capability_uuid(capability: str) -> str:
    """One-time UUID seed helper. NOT called at runtime by the wheel.

    Operators can use this at the REPL to generate a starting UUID from
    a name they like, then paste the result into their ToolIdentity
    declaration. After that, the UUID is opaque — renaming ``capability``
    in the source doesn't change ``tool_id``.

    For brand-new tools with no historical UUID, ``uuid.uuid4()`` is
    equally fine.
    """
    return str(uuid.uuid5(DPYC_NAMESPACE, capability))


@dataclass(frozen=True)
class ToolIdentity:
    """Declares what a tool IS — not what it costs.

    Attributes:
        tool_id: Opaque, frozen UUID string. Author picks once at tool
            birth (via ``capability_uuid("...")`` or ``uuid.uuid4()``)
            and never changes. The wheel treats this as ``str`` and
            does not parse it. Renaming any other field never changes
            the identity.
        capability: Human-readable short label (e.g. ``"check_balance"``).
            Used for display only — NOT for UUID derivation. Safe to
            rename. Defaults to empty.
        category: Access/billing category.
            ``"free"`` and ``"restricted"`` are gated by the runtime
            without consulting Neon. All others (``"read"``, ``"write"``,
            ``"heavy"``, etc.) require a price entry in Neon.
        intent: Human-readable purpose.
        mcp_name: Full namespace-scoped MCP tool name
            (e.g. ``"taxsort_check_balance"``). Set during
            ``register_standard_tools`` when the slug is known.
            This is the display name used everywhere outside the
            server process.
    """

    tool_id: str
    category: str  # "free" | "read" | "write" | "heavy" | "restricted"
    intent: str
    capability: str = ""
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
    def display_name(self) -> str:
        """The human-visible tool name — always the full MCP name."""
        return self.mcp_name or self.capability or self.tool_id


# ======================================================================
# Standard identities for all wheel-provided tools.
#
# Keyed by UUID (tool_id). Every operator inherits these via the wheel.
# ======================================================================

# Frozen UUIDs for every wheel-provided standard tool. These values
# were derived from `capability_uuid("<capability>")` at the time of
# this refactor, then pinned as constants. They MUST NOT be regenerated
# from any string going forward — operators' pricing-model rows in Neon
# reference these UUIDs as primary keys.

# Credit tools
CHECK_BALANCE_UUID                   = "f09d5935-e58d-52a6-a5d3-ff82794ec7fb"
PURCHASE_CREDITS_UUID                = "969d9474-1336-578a-ae57-fc7da2376bf9"
CHECK_PAYMENT_UUID                   = "3153ee9a-75fc-5ade-9438-6b6ec6aff87e"
RESTORE_CREDITS_UUID                 = "b6ecce2e-e465-5f3e-a900-0d830e13d583"
ACCOUNT_STATEMENT_UUID               = "c72d17a3-7938-57ab-aa4c-32cc760bcf8f"
ACCOUNT_STATEMENT_INFOGRAPHIC_UUID   = "a864c193-a360-5621-b2b6-80694ea5cf2c"

# Service status
SERVICE_STATUS_UUID                  = "20a7c4aa-1e30-5b2e-b6e9-083524d9b324"

# Onboarding
GET_OPERATOR_ONBOARDING_STATUS_UUID  = "3d2324db-146e-5322-8cb6-a72dcac06286"
GET_PATRON_ONBOARDING_STATUS_UUID    = "cf068462-ee01-54f3-b0ab-a8289a2b64e8"
SESSION_STATUS_UUID                  = "5580628c-81c1-582e-9977-406d25b7b972"

# Secure Courier
REQUEST_CREDENTIAL_CHANNEL_UUID      = "e9fda3d1-998d-5317-88e8-86b8d29ff826"
RECEIVE_CREDENTIALS_UUID             = "20c06e15-ab91-569e-99f1-c9b2b7d80816"
FORGET_CREDENTIALS_UUID              = "4c0d5cbb-0ce2-5382-9104-969642a2a9e4"
REQUEST_PATRON_CREDENTIALS_UUID      = "0048005c-ccd5-5d71-a36c-2f5dca126cef"
RECEIVE_PATRON_CREDENTIALS_UUID      = "c2ff1869-c70b-53ef-bc16-d2f36744c76b"

# OAuth2 (conditional)
BEGIN_OAUTH_UUID                     = "f9a51b5a-e557-5345-a8cc-3df7355d51bd"
CHECK_OAUTH_STATUS_UUID              = "25acf3be-327d-53c1-94f1-c704d5354a96"

# Npub ownership proof
REQUEST_NPUB_PROOF_UUID              = "dbd9e8fb-d5f6-5015-8345-1c36229e420c"
RECEIVE_NPUB_PROOF_UUID              = "922ad409-7d07-5cb3-9f66-a6c4eecc24a1"
CHECK_PROOF_STATUS_UUID              = "4a41f606-b1cf-5ca2-8e56-74daebc092fb"

# Oracle delegation
ORACLE_ABOUT_UUID                    = "0acf5be2-48de-56b1-a3c4-ff3e981f3e80"
ORACLE_GET_TAX_RATE_UUID             = "290f77f4-6a6c-5aa1-b2a9-382cb50dc687"
ORACLE_HOW_TO_JOIN_UUID              = "170bd982-daaa-52c1-be78-52100d6311a0"
ORACLE_LOOKUP_MEMBER_UUID            = "b1e75f58-24f1-5f8b-b52b-01e957aa1d61"
ORACLE_NETWORK_ADVISORY_UUID         = "d4d20c39-3582-53e4-812d-e7f49bdcd98a"

# Authority delegation
CHECK_AUTHORITY_BALANCE_UUID         = "e5bfa0f5-772b-538e-809e-9e0f69d656d2"

# Pricing CRUD
GET_PRICING_MODEL_UUID               = "2be47909-70c7-5dcc-b0f4-50198c1f86bd"
SET_PRICING_MODEL_UUID               = "1b2449d0-521e-52b5-b087-eba0b9948889"
RESET_PRICING_MODEL_UUID             = "4735520b-e1e3-5d47-bb71-c26c394f50c1"
CHECK_PRICE_UUID                     = "f0e43654-5dd1-5a67-a8f7-59e2c168a9a2"
LIST_CONSTRAINT_TYPES_UUID           = "9855b0a8-f458-5e79-a0ca-207f142d7800"

# New in this refactor: introspection of canonical identities.
LIST_CANONICAL_IDENTITIES_UUID       = "e7a9c2f6-1d4b-4c3e-8f7a-5b9d2c1e8f3a"

# Patron credential CRUD
UPDATE_PATRON_CREDENTIAL_UUID        = "e896e938-c9f1-5d76-9489-dc928ca907e5"
DELETE_PATRON_CREDENTIAL_UUID        = "daf3ffe7-f5a6-5e38-bf4f-b68580a66e45"
GET_PATRON_CREDENTIAL_FIELDS_UUID    = "16dac337-6482-5c55-9710-ca7aceaa02b6"

# OTS notarization
NOTARIZE_LEDGER_UUID                 = "fa72cde0-35a6-5fb3-8a49-85041cf3fb6c"
GET_NOTARIZATION_PROOF_UUID          = "144b6fd8-6409-5642-a72a-d0b14434de73"
LIST_NOTARIZATIONS_UUID              = "6600a576-84a1-5952-bbb9-350b2ee73ef9"

# Coupon CRUD (wheel 0.41.0+)
MINT_COUPON_UUID                     = "b2f6a56c-6641-52ec-9e83-934a011dca4e"
LIST_COUPONS_UUID                    = "3e5c6e5c-6a49-5646-8d17-a6fff6c72e5b"
UPDATE_COUPON_UUID                   = "566cc15c-e449-5f18-8a71-35f4e7e2f62e"
DELETE_COUPON_UUID                   = "5b7656d5-bbd0-5c4a-af6d-7999c41f54bb"
REDEEM_COUPON_UUID                   = "359cdb8d-3e39-58ef-91b4-42f786891b1e"
LIST_MY_COUPONS_UUID                 = "6f41dfc6-9141-518c-84c9-07b0fdb3fa34"
FORGET_COUPON_UUID                   = "08fc1a2b-5924-5761-9f8d-bc334001776e"


_STANDARD_LIST: list[ToolIdentity] = [
    # -- Credit tools --
    ToolIdentity(tool_id=CHECK_BALANCE_UUID, capability="check_balance", category="free",
                 intent="Check a patron's credit balance."),
    ToolIdentity(tool_id=PURCHASE_CREDITS_UUID, capability="purchase_credits", category="free",
                 intent="Buy credits via Bitcoin Lightning."),
    ToolIdentity(tool_id=CHECK_PAYMENT_UUID, capability="check_payment", category="free",
                 intent="Check payment status of a Lightning invoice."),
    ToolIdentity(tool_id=RESTORE_CREDITS_UUID, capability="restore_credits", category="restricted",
                 intent="Operator-only: credit a patron's ledger from a "
                        "BTCPay-settled invoice. Discretionary recovery for "
                        "support escalations and infrastructure incidents."),
    ToolIdentity(tool_id=ACCOUNT_STATEMENT_UUID, capability="account_statement", category="free",
                 intent="Generate a patron's account statement."),
    ToolIdentity(tool_id=ACCOUNT_STATEMENT_INFOGRAPHIC_UUID, capability="account_statement_infographic", category="read",
                 intent="Generate a visual SVG infographic of account statement."),

    # -- Service status --
    ToolIdentity(tool_id=SERVICE_STATUS_UUID, capability="service_status", category="free",
                 intent="Check health and configuration of this service."),

    # -- Onboarding --
    ToolIdentity(tool_id=GET_OPERATOR_ONBOARDING_STATUS_UUID, capability="get_operator_onboarding_status", category="free",
                 intent="Report operator configuration readiness."),
    ToolIdentity(tool_id=GET_PATRON_ONBOARDING_STATUS_UUID, capability="get_patron_onboarding_status", category="free",
                 intent="Report a patron's credential readiness."),
    ToolIdentity(tool_id=SESSION_STATUS_UUID, capability="session_status", category="free",
                 intent="Check operator lifecycle state and readiness."),

    # -- Secure Courier --
    ToolIdentity(tool_id=REQUEST_CREDENTIAL_CHANNEL_UUID, capability="request_credential_channel", category="free",
                 intent="Open a Secure Courier channel for credential delivery."),
    ToolIdentity(tool_id=RECEIVE_CREDENTIALS_UUID, capability="receive_credentials", category="free",
                 intent="Pick up credentials from the Secure Courier."),
    ToolIdentity(tool_id=FORGET_CREDENTIALS_UUID, capability="forget_credentials", category="free",
                 intent="Delete vaulted credentials for a service."),
    ToolIdentity(tool_id=REQUEST_PATRON_CREDENTIALS_UUID, capability="request_patron_credentials", category="free",
                 intent="Open a Secure Courier channel for patron credentials."),
    ToolIdentity(tool_id=RECEIVE_PATRON_CREDENTIALS_UUID, capability="receive_patron_credentials", category="free",
                 intent="Pick up patron credentials from the Secure Courier."),

    # -- OAuth2 (conditional — only if oauth_provider is configured) --
    ToolIdentity(tool_id=BEGIN_OAUTH_UUID, capability="begin_oauth", category="free",
                 intent="Start OAuth2 authorization flow."),
    ToolIdentity(tool_id=CHECK_OAUTH_STATUS_UUID, capability="check_oauth_status", category="free",
                 intent="Check OAuth2 authorization status."),

    # -- Npub ownership proof --
    ToolIdentity(tool_id=REQUEST_NPUB_PROOF_UUID, capability="request_npub_proof", category="free",
                 intent="Request npub ownership proof from patron via DM."),
    ToolIdentity(tool_id=RECEIVE_NPUB_PROOF_UUID, capability="receive_npub_proof", category="free",
                 intent="Receive and cache npub ownership proof."),
    ToolIdentity(tool_id=CHECK_PROOF_STATUS_UUID, capability="check_proof_status", category="free",
                 intent="Read-only check of a proof_token's remaining validity."),

    # -- Oracle delegation --
    # Wire-exposed under the operator's `_oracle_` namespace (e.g.
    # `optionality_oracle_about`). Routed through a separate free
    # decorator that bypasses pricing/gating entirely, so the price
    # here is informational only — the wheel does not consult it at
    # call time. Included in STANDARD_IDENTITIES anyway so the
    # initial pricing model is a complete inventory of what's exposed
    # (otherwise Studio's Reconcile keeps offering them as "new tools"
    # on every reset).
    ToolIdentity(tool_id=ORACLE_ABOUT_UUID, capability="oracle_about", category="free",
                 intent="Describe the DPYC ecosystem via the Oracle."),
    ToolIdentity(tool_id=ORACLE_GET_TAX_RATE_UUID, capability="oracle_get_tax_rate", category="free",
                 intent="Get the current DPYC certification tax rate."),
    ToolIdentity(tool_id=ORACLE_HOW_TO_JOIN_UUID, capability="oracle_how_to_join", category="free",
                 intent="Get DPYC onboarding instructions from the Oracle."),
    ToolIdentity(tool_id=ORACLE_LOOKUP_MEMBER_UUID, capability="oracle_lookup_member", category="free",
                 intent="Look up a DPYC community member by npub."),
    ToolIdentity(tool_id=ORACLE_NETWORK_ADVISORY_UUID, capability="oracle_network_advisory", category="free",
                 intent="Get active network advisories from the Oracle."),

    # -- Authority delegation --
    ToolIdentity(tool_id=CHECK_AUTHORITY_BALANCE_UUID, capability="check_authority_balance", category="free",
                 intent="Check operator's tax balance at the Authority."),

    # -- Pricing CRUD --
    ToolIdentity(tool_id=GET_PRICING_MODEL_UUID, capability="get_pricing_model", category="free",
                 intent="Get the active pricing model."),
    ToolIdentity(tool_id=SET_PRICING_MODEL_UUID, capability="set_pricing_model", category="restricted",
                 intent="Set the active pricing model. Operator only."),
    ToolIdentity(tool_id=RESET_PRICING_MODEL_UUID, capability="reset_pricing_model", category="restricted",
                 intent="Delete all pricing models and re-initialize. Operator only."),
    ToolIdentity(tool_id=CHECK_PRICE_UUID, capability="check_price", category="free",
                 intent="Preview the effective cost of a tool call."),
    ToolIdentity(tool_id=LIST_CONSTRAINT_TYPES_UUID, capability="list_constraint_types", category="free",
                 intent="List available constraint types and parameter schemas."),
    ToolIdentity(tool_id=LIST_CANONICAL_IDENTITIES_UUID, capability="list_canonical_identities", category="free",
                 intent="Return the canonical (tool_id, mcp_name, category, intent) tuples for "
                        "every tool the running wheel exposes. Source of truth for Reconcile."),

    # -- Patron credential CRUD --
    ToolIdentity(tool_id=UPDATE_PATRON_CREDENTIAL_UUID, capability="update_patron_credential", category="free",
                 intent="Add or update a single patron credential field."),
    ToolIdentity(tool_id=DELETE_PATRON_CREDENTIAL_UUID, capability="delete_patron_credential", category="free",
                 intent="Remove a single patron credential field."),
    ToolIdentity(tool_id=GET_PATRON_CREDENTIAL_FIELDS_UUID, capability="get_patron_credential_fields", category="free",
                 intent="List stored patron credential field names."),

    # -- OTS notarization --
    ToolIdentity(tool_id=NOTARIZE_LEDGER_UUID, capability="notarize_ledger", category="restricted",
                 intent="Submit patron balance Merkle root to Bitcoin via OTS."),
    ToolIdentity(tool_id=GET_NOTARIZATION_PROOF_UUID, capability="get_notarization_proof", category="free",
                 intent="Generate a Merkle inclusion proof for a patron balance."),
    ToolIdentity(tool_id=LIST_NOTARIZATIONS_UUID, capability="list_notarizations", category="free",
                 intent="List recent Bitcoin notarization records."),

    # -- Coupon CRUD (wheel 0.41.0+) --
    # Operator-restricted: mint / list / update / delete.  Patrons hit
    # these tools through the Pricing Studio; the wheel verifies an
    # operator-signed kind-27235 proof against the runtime name.
    ToolIdentity(tool_id=MINT_COUPON_UUID, capability="mint_coupon", category="restricted",
                 intent="Operator-only: mint a new discount coupon."),
    ToolIdentity(tool_id=LIST_COUPONS_UUID, capability="list_coupons", category="restricted",
                 intent="Operator-only: list this operator's coupons."),
    ToolIdentity(tool_id=UPDATE_COUPON_UUID, capability="update_coupon", category="restricted",
                 intent="Operator-only: patch a coupon's editable fields."),
    ToolIdentity(tool_id=DELETE_COUPON_UUID, capability="delete_coupon", category="restricted",
                 intent="Operator-only: delete a coupon and cascade patron redemptions."),
    # Patron-facing free tools — npub proof against the caller, not the operator.
    ToolIdentity(tool_id=REDEEM_COUPON_UUID, capability="redeem_coupon", category="free",
                 intent="Claim a coupon by name; subsequent paid calls apply the discount."),
    ToolIdentity(tool_id=LIST_MY_COUPONS_UUID, capability="list_my_coupons", category="free",
                 intent="List the patron's redeemed coupons on this operator."),
    ToolIdentity(tool_id=FORGET_COUPON_UUID, capability="forget_coupon", category="free",
                 intent="Remove a coupon from the patron's redemption list."),
]

# Build the UUID-keyed dict from the list.
STANDARD_IDENTITIES: dict[str, ToolIdentity] = {
    ti.tool_id: ti for ti in _STANDARD_LIST
}
