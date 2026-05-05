"""Constants for Tollbooth micropayment gating."""

MAX_INVOICE_SATS = 1_000_000  # 0.01 BTC cap per invoice
LOW_BALANCE_FLOOR_API_SATS = 100  # minimum warning threshold


# Error codes returned by debit_or_deny and paid_tool denials.
# These are stable strings that calling agents can branch on
# without parsing prose.  Add new codes here; never inline.
class ErrorCode:
    """Stable string codes for paid-tool denial paths.

    Each denial response from ``debit_or_deny`` and from the
    ``paid_tool`` decorator's catch-errors fallback carries one
    of these values in the ``error_code`` field, so callers can
    branch programmatically.  Companion ``next_steps`` lists are
    included where the situation is patron-actionable.
    """

    NPUB_INVALID = "npub_invalid"
    PROOF_REQUIRED = "proof_required"
    PROOF_INVALID = "proof_invalid"
    PROOF_REFRESH_NEEDED = "proof_refresh_needed"
    RESTRICTED = "restricted"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    TOOL_NOT_PRICED = "tool_not_priced"
    WARMING_UP = "warming_up"
    OPERATOR_NOT_REGISTERED = "operator_not_registered"
    INSUFFICIENT_BALANCE = "insufficient_balance"
    CONSTRAINT_DENIED = "constraint_denied"
    UPSTREAM_AUTH_REFRESH_NEEDED = "upstream_auth_refresh_needed"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    SECURE_COURIER_UNAVAILABLE = "secure_courier_unavailable"
    VAULT_BOOTSTRAPPING = "vault_bootstrapping"

    # OAuth session-restoration situations (from
    # OperatorRuntime.restore_oauth_session → oauth_situation_response).
    # OAUTH_REFRESH_NEEDED is the patron-actionable case: a fresh
    # browser authorization is required.  OPERATOR_NOT_CONFIGURED is
    # an operator-actionable setup state.
    OAUTH_REFRESH_NEEDED = "oauth_refresh_needed"
    OPERATOR_NOT_CONFIGURED = "operator_not_configured"

# Canonical links to DPYC ecosystem repos and live services.
# Operators should include these in service_status responses so
# AI agents can discover sibling services without web search.
ECOSYSTEM_LINKS: dict[str, str] = {
    "dpyc_community": "https://github.com/lonniev/dpyc-community",
    "tollbooth_dpyc": "https://github.com/lonniev/tollbooth-dpyc",
    "tollbooth_authority": "https://github.com/lonniev/tollbooth-authority",
    "thebrain_mcp": "https://github.com/lonniev/thebrain-mcp",
    "excalibur_mcp": "https://github.com/lonniev/excalibur-mcp",
    "dpyc_oracle": "https://github.com/lonniev/dpyc-oracle",
    "tollbooth_sample": "https://github.com/lonniev/tollbooth-sample",
    "dpyc_oracle_mcp": "https://dpyc-oracle.fastmcp.app/mcp",
}
