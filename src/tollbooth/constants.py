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

    # Patron identity
    NPUB_MISSING = "npub_missing"
    NPUB_INVALID = "npub_invalid"

    # Npub ownership proof (from request_npub_proof / receive_npub_proof
    # → cached as dpop_token_hash in ProvenNpubCache).
    PROOF_MISSING = "proof_missing"          # parameter empty
    PROOF_REQUIRED = "proof_required"        # restricted-tool path requiring inline Schnorr
    PROOF_INVALID = "proof_invalid"          # signature does not verify
    PROOF_REFRESH_NEEDED = "proof_refresh_needed"  # dpop_token-keyed dpop_token cache miss

    # Authority-side discretionary consent (register/update/deregister_operator).
    # The Authority's adoption of a new Operator (or modification of an
    # existing one) is gated by a Schnorr proof signed by the Authority's
    # OWN npub — the cryptographic witness that the human who controls the
    # Authority's nsec has authorized this action. A missing or invalid
    # ``authority_proof`` argument trips this code.
    AUTHORITY_CONSENT_REQUIRED = "authority_consent_required"

    # Tool/registry
    RESTRICTED = "restricted"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    TOOL_NOT_PRICED = "tool_not_priced"

    # Operator adoption (deferred courtship)
    ADOPTION_PENDING = "adoption_pending"                  # request awaiting owner decision
    ADOPTION_NOT_FOUND = "adoption_not_found"              # no request for this operator npub
    ADOPTION_ALREADY_PROVISIONED = "adoption_already_provisioned"  # idempotent re-approve guard

    # Lifecycle / infrastructure
    WARMING_UP = "warming_up"
    # Persistence rejected the query with a permanent SQL error (permission
    # denied, missing relation, auth failure). Unlike WARMING_UP this will
    # NOT resolve by retrying — the operator must repair the database.
    PERSISTENCE_MISCONFIGURED = "persistence_misconfigured"
    # The persistence PROVIDER (Neon) answered HTTP 402: the operator's
    # database — the DPYC economy's accounting books — has exhausted its
    # compute/storage quota, so every query is refused at the gateway before
    # any SQL runs. NOT a cold start (WARMING_UP) and NOT a SQL/permission
    # fault (PERSISTENCE_MISCONFIGURED): retrying cannot help. The Authority
    # that provisions these books must upgrade the plan or wait for the quota
    # to reset. Distinct from UPSTREAM_SUBSCRIPTION_REQUIRED, which is a
    # business API's 402 — this is the books themselves going dark.
    PERSISTENCE_QUOTA_EXCEEDED = "persistence_quota_exceeded"
    OPERATOR_NOT_REGISTERED = "operator_not_registered"
    VAULT_BOOTSTRAPPING = "vault_bootstrapping"
    SECURE_COURIER_UNAVAILABLE = "secure_courier_unavailable"

    # Secure Courier deterministic retrieval (receive_credentials /
    # receive_patron_credentials / receive_npub_proof). The client names
    # the response it wants via (sender_npub, service, dpop_token); these codes
    # report why a dpop_token-scoped, pinned-relay drain could not return it.
    DPOP_TOKEN_MISSING = "dpop_token_missing"                      # dpop_token argument empty
    COURIER_NO_PENDING_RECORD = "courier_no_pending_record"  # no open channel for (npub, service)
    COURIER_DPOP_TOKEN_MISMATCH = "courier_dpop_token_mismatch"    # dpop_token does not match the open channel
    COURIER_TOKEN_EXPIRED = "courier_token_expired"        # channel's freshness window has elapsed
    COURIER_NO_PINNED_RELAY = "courier_no_pinned_relay"    # record has no rendezvous relay to drain
    COURIER_NOT_FOUND = "courier_not_found"                # relay drained; no DM matched the dpop_token

    # Billing / pricing
    INSUFFICIENT_BALANCE = "insufficient_balance"
    CONSTRAINT_DENIED = "constraint_denied"
    # The Operator's own balance at its certifying Authority is exhausted, so
    # the Authority refused to certify a patron's credit purchase. This is the
    # Operator's supply problem, not the patron's — surfaced as a kind "please
    # be patient" situation while the Authority is dunned (out of band) to
    # top up. Distinct from INSUFFICIENT_BALANCE (the patron's own balance).
    AUTHORITY_INSUFFICIENT_BALANCE = "authority_insufficient_balance"

    # Generic execution
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    # A caller-facing input problem the operator signalled by raising
    # ValueError (bad key, invalid params, lifecycle situation). Its message
    # is surfaced verbatim so the caller can self-correct.
    TOOL_INPUT_INVALID = "tool_input_invalid"
    UPSTREAM_AUTH_REFRESH_NEEDED = "upstream_auth_refresh_needed"
    # An upstream API answered HTTP 402 because the paid subscription / access
    # tier tied to the credentials this service uses has lapsed or does not
    # cover the request. This is NOT the x402 micropayment protocol (X402Client
    # settles that transparently as Operator COGS) and NOT a patron-balance
    # problem (INSUFFICIENT_BALANCE) — a human must renew or upgrade the plan at
    # the upstream provider. Non-transient: retrying will not help until the
    # subscription is restored. Built by tollbooth.upstream_payment.
    UPSTREAM_SUBSCRIPTION_REQUIRED = "upstream_subscription_required"

    # OAuth session-restoration situations (from
    # OperatorRuntime.restore_oauth_session → oauth_situation_response).
    # 1:1 with the situation strings — same recovery flow may be shared
    # via next_steps, but the error_code preserves the diagnostic
    # specificity the calling agent needs to phrase patron-facing output.
    OAUTH_TOKEN_EXPIRED = "oauth_token_expired"          # returning patron, refresh token aged out / revoked
    OAUTH_NOT_YET_AUTHORIZED = "oauth_not_yet_authorized"  # first-time patron, no token in vault
    OAUTH_NOT_WIRED = "oauth_not_wired"                  # operator MCP has no OAuthProviderConfig
    OPERATOR_CREDENTIALS_MISSING = "operator_credentials_missing"  # vault load failed
    OAUTH_SITUATION_UNKNOWN = "oauth_situation_unknown"  # fallthrough — situation string echoed in message

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
