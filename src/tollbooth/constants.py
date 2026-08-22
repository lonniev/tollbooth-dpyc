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
    COURIER_RELAY_UNREACHABLE = "courier_relay_unreachable"  # pinned relay never answered; nothing was drained

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
    # An upstream refused because it is being asked to serve too fast. The only
    # situation in this family that a retry can actually clear.
    UPSTREAM_RATE_LIMITED = "upstream_rate_limited"

    # The Operator's LLM provider (model router or lab) refused the call. These are
    # Operator COGS problems — the patron can do nothing about any of them, so each is
    # non-transient and each is worth waking the Operator for. Built by
    # tollbooth.llm_route. Distinct from UPSTREAM_SUBSCRIPTION_REQUIRED: a business
    # API's 402 means a human renews a plan, whereas a metered LLM account is topped up.
    LLM_PROVIDER_UNFUNDED = "operator_llm_unfunded"  # account out of credit
    LLM_PROVIDER_AUTH = "operator_llm_auth"          # key missing, wrong, or revoked
    # The configured model slug is one the provider does not recognise — the signature
    # of a marketplace renaming or retiring a model under a running deployment. Permanent
    # until someone edits the configuration, so never report it as retryable.
    LLM_MODEL_UNKNOWN = "operator_llm_model_unknown"

    # OAuth session-restoration situations (from
    # OperatorRuntime.restore_oauth_session → oauth_situation_response).
    # 1:1 with the situation strings — same recovery flow may be shared
    # via next_steps, but the error_code preserves the diagnostic
    # specificity the calling agent needs to phrase patron-facing output.
    # Stated ONLY when the provider answered `invalid_grant` and no earlier
    # lost refresh is on record. Four other situations used to borrow this code
    # and each sent someone to re-authorize a session that was fine — they are
    # the five codes that follow.
    OAUTH_TOKEN_EXPIRED = "oauth_token_expired"          # returning patron, refresh token aged out / revoked
    # The grant died because a refresh answer was LOST, not because it aged out.
    # A provider that rotates single-use refresh tokens (X does) rotates on
    # request ARRIVAL, so a read timeout can leave the provider holding a
    # replacement we never received and our vault holding a spent token. Every
    # later attempt replays the spent one and draws `invalid_grant` — which read
    # as "expired" and sent the operator to reconnect, over and over, with no
    # hint that a specific timeout hours earlier was the cause. Carries the
    # instant of that timeout in `observed_at`. A reconnect IS required; what
    # changes is that the operator learns why, and can see how often it happens.
    OAUTH_REFRESH_TOKEN_LOST = "oauth_refresh_token_lost"
    # The vault holds an access token but no refresh token, so there is nothing
    # to spend when it lapses. Usually a grant issued without offline access.
    # Nothing expired and nothing was refused — reporting this as "expired" hid
    # a scope problem behind a story about time.
    OAUTH_NO_REFRESH_TOKEN = "oauth_no_refresh_token"
    # The refresh raised something we could not classify. Kept RETRYABLE and
    # named as unclassified: an unknown cause reported as a dead grant costs a
    # re-authorization on a guess, and the guess was wrong often enough to be
    # the reason this code exists. The exception type travels in `detail`.
    OAUTH_REFRESH_FAILED_UNCLASSIFIED = "oauth_refresh_failed_unclassified"
    # The provider refused the OPERATOR's app credentials (`invalid_client` /
    # `unauthorized_client`). The patron's authorization is untouched and they
    # can do nothing about this — least of all reconnect, which re-authorizes a
    # live account against a broken app.
    OPERATOR_APP_CREDENTIALS_REJECTED = "operator_app_credentials_rejected"
    # The provider called our refresh request malformed (`invalid_request`).
    # That is a deployment/SDK bug: no patron and no operator can fix it by
    # doing anything to their account.
    OAUTH_REFRESH_REQUEST_MALFORMED = "oauth_refresh_request_malformed"
    # The refresh could not be COMPLETED — timeout, connect failure, 429, 5xx, a
    # body that wasn't JSON. Deliberately distinct from OAUTH_TOKEN_EXPIRED:
    # nothing here says the grant is bad, so this must never send a patron
    # through OAuth again. Retryable, and the only OAuth situation that is.
    OAUTH_REFRESH_UNAVAILABLE = "oauth_refresh_unavailable"
    # The upstream API rejected the ACCESS token (401/403) while our records
    # still considered it fresh. That says nothing about the refresh token, so
    # it is not OAUTH_TOKEN_EXPIRED: an access token is the short-lived half of
    # the grant and a rejected one is routinely cured by spending the refresh
    # token. Reported after the cached expiry has been invalidated, so the very
    # next call renews — retryable, like OAUTH_REFRESH_UNAVAILABLE.
    OAUTH_TOKEN_REJECTED = "oauth_token_rejected"
    # Code→token exchange situations (check_oauth_status). Parallel to the
    # refresh family above, but no session exists yet — the authorization code
    # was refused or the exchange never completed. Kept distinct so a calling
    # agent never phrases a brand-new dance as "your existing session aged out".
    OAUTH_EXCHANGE_GRANT_REJECTED = "oauth_exchange_grant_rejected"  # invalid_grant on the code
    OAUTH_EXCHANGE_REQUEST_MALFORMED = "oauth_exchange_request_malformed"  # invalid_request
    OAUTH_EXCHANGE_UNAVAILABLE = "oauth_exchange_unavailable"  # timeout / 429 / 5xx / unnamed 4xx
    OAUTH_EXCHANGE_FAILED_UNCLASSIFIED = "oauth_exchange_failed_unclassified"
    OAUTH_NOT_YET_AUTHORIZED = "oauth_not_yet_authorized"  # first-time patron, no token in vault
    OAUTH_NOT_WIRED = "oauth_not_wired"                  # operator MCP has no OAuthProviderConfig
    OPERATOR_CREDENTIALS_MISSING = "operator_credentials_missing"  # vault load failed
    OAUTH_SITUATION_UNKNOWN = "oauth_situation_unknown"  # fallthrough — situation string echoed in message

# The DPYC Oracle's public MCP endpoint — the ONE fixed anchor an Operator is
# allowed to know a priori. An nsec-only operator seeds its relays and resolves
# community facts (its Authority, sibling services) through this endpoint via
# MCP-to-MCP calls, so it never reads the dpyc-community registry on GitHub
# itself. Changing this is the only bootstrap coordinate baked into the SDK.
DPYC_ORACLE_MCP_URL = "https://dpyc-oracle.fastmcp.app/mcp"

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
    "dpyc_oracle_mcp": DPYC_ORACLE_MCP_URL,
}
