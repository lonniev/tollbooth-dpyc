# Tollbooth DPYC

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/tollbooth-dpyc)](https://pypi.org/project/tollbooth-dpyc/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

<p align="center">
  <img src="https://raw.githubusercontent.com/lonniev/tollbooth-dpyc/main/docs/tollbooth-hero.png" alt="Milo drives the Lightning Turnpike — Don't Pester Your Customer" width="800">
</p>

**Don't Pester Your Customer** — Bitcoin Lightning micropayments for MCP servers.

> *The metaphors in this project are drawn with admiration from* The Phantom Tollbooth *by Norton Juster, illustrated by Jules Feiffer (1961). Milo, Tock, the Tollbooth, Dictionopolis, and Digitopolis are creations of Mr. Juster's extraordinary imagination. We just built the payment infrastructure.*

---

## The Problem

Thousands of developers are building [MCP](https://modelcontextprotocol.io/) servers — services that let AI agents interact with the world. Knowledge graphs, financial data, social media, medical records. Each one is a city on the map. But the turnpike between them? Wide open. No toll collectors. No sustainable economics.

Every MCP operator faces the same question: *how do I keep the lights on?*

Traditional API keys with monthly billing? You're running a SaaS company now. L402 — Lightning-native pay-per-request? Every single API call requires a payment negotiation. Milo's toy car stops at every intersection to fumble for exact change.

## The Solution

Tollbooth DPYC takes a different approach — one that respects everyone's time:

**Milo drives up to the tollbooth once, buys a roll of tokens with a single Lightning invoice, and drives.** No stops. No negotiations. No per-request friction. The tokens quietly decrement in the background. When the roll runs low, he buys another. The turnpike stays fast.

Prepaid credits over Bitcoin's Lightning Network, gated at the tool level, settled instantly, with no subscription management and no third-party payment processor taking a cut.

---

## Quick Start: Build an Operator in 5 Minutes

The canonical reference operator is [tollbooth-sample](https://github.com/lonniev/tollbooth-sample). Every production operator follows this pattern.

### 1. Install

```bash
pip install "tollbooth-dpyc[nostr]" fastmcp
```

### 2. Define Your Tools

```python
from fastmcp import FastMCP
from tollbooth.tool_identity import ToolIdentity, STANDARD_IDENTITIES, capability_uuid
from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.slug_tools import make_slug_tool

mcp = FastMCP("My Service")

# Domain tools — what your service actually does
_DOMAIN_TOOLS = [
    ToolIdentity(capability="get_weather", category="read", intent="Current weather data"),
    ToolIdentity(capability="get_forecast", category="read", intent="5-day forecast"),
]

TOOL_REGISTRY = {t.id: t for t in _DOMAIN_TOOLS}
```

### 3. Create the Runtime

```python
runtime = OperatorRuntime(
    tool_registry={**STANDARD_IDENTITIES, **TOOL_REGISTRY},
    operator_credential_template=CredentialTemplate(
        service="my-operator",
        fields={
            "btcpay_host": FieldSpec(required=True, sensitive=False, description="BTCPay Server URL"),
            "btcpay_api_key": FieldSpec(required=True, sensitive=True, description="BTCPay API key"),
            "btcpay_store_id": FieldSpec(required=True, sensitive=True, description="BTCPay store ID"),
        },
    ),
    service_name="My Weather Service",
)
```

### 4. Register Standard Tools + Domain Tools

```python
# This single call registers all 21 standard DPYC tools
register_standard_tools(mcp, "weather", runtime,
    service_name="my-weather-service",
    service_version="1.0.0",
)

# Domain tools use the paid_tool decorator
tool = make_slug_tool(mcp, "weather")

@tool
@runtime.paid_tool(capability_uuid("get_weather"))
async def get_weather(city: str, npub: str = "", proof: str = "") -> dict:
    """Get current weather for a city. Costs 1 api_sat."""
    # Your domain logic here
    return {"city": city, "temp": 72, "conditions": "sunny"}
```

### 5. Deploy

Set one environment variable and deploy:

```bash
export TOLLBOOTH_NOSTR_OPERATOR_NSEC=nsec1...
fastmcp run server.py
```

That's it. The runtime bootstraps everything else from the Authority via encrypted Nostr DM.

---

## How It Works

### The Deploy Flow

1. **Set `TOLLBOOTH_NOSTR_OPERATOR_NSEC`** — the operator's Nostr private key. This is the only secret needed at boot.
2. **Runtime bootstraps from Authority** — discovers its Neon Postgres URL from an encrypted DM on Nostr relays, signed by the upstream Authority.
3. **Vault initializes** — AES-256-GCM encrypted credential and ledger storage on Neon Postgres, schema-isolated per operator (`op_{hash}`).
4. **Deliver operator credentials via Secure Courier** — BTCPay connection details arrive via encrypted Nostr DM (human-in-the-loop, never in env vars).
5. **Configure pricing** — set a pricing model via the Pricing Studio iOS app or the `set_pricing_model` tool.
6. **Serve** — patrons purchase credits via Lightning, use tools, credits decrement automatically.

### Credit Lifecycle

1. **Patron calls `purchase_credits`** — the operator obtains an Authority certificate (Schnorr-signed Nostr event) and creates a Lightning invoice.
2. **Patron pays** the Lightning invoice with any wallet.
3. **Patron calls `check_payment`** — on settlement, credits are added as a tranche with optional expiry (`TrancheLifetime`).
4. **Patron uses tools** — `debit_or_deny` gates every paid tool call: validates identity, proof, constraints, pricing, and debits the balance. FIFO tranche consumption.
5. **Balance runs low** — patron buys more credits. No subscription, no interruption.

### The Certification Fee Cascade

When a patron purchases credits, the operator's upstream Authority deducts a small certification fee (default 2%, minimum 10 sats) from the operator's pre-funded reserve. The patron always receives exactly the credits they paid for — the fee is the operator's cost of doing business.

Each Authority is itself an operator of its upstream Authority — the same fee cascades up through the chain to the Prime Authority.

---

## Standard Tools

`register_standard_tools(mcp, slug, runtime)` registers these tools, prefixed with the operator's slug (e.g., `weather_check_balance`):

### Credit & Billing (always registered)

| Tool | Cost | Purpose |
|------|------|---------|
| `check_balance` | Free | Current credit balance, tranches, usage summary |
| `purchase_credits` | Free | Create a Lightning invoice for credit purchase |
| `check_payment` | Free | Poll invoice status; credit balance on settlement |
| `check_price` | Free | Preview the effective cost of a tool call (after constraints) |
| `restore_credits` | Free | Recover credits from a paid invoice lost to cache issues |
| `account_statement` | Free | 30-day purchase and usage history |
| `account_statement_infographic` | 1 sat | SVG visual of account statement |

### Identity & Lifecycle

| Tool | Cost | Purpose |
|------|------|---------|
| `service_status` | Free | Operator health, lifecycle state, version info |
| `session_status` | Free | Operator readiness: ready / warming_up / not_registered / no_identity |
| `get_operator_onboarding_status` | Free | Which operator credentials are configured vs missing |
| `get_patron_onboarding_status` | Free | Which patron credentials are configured vs missing |

### Secure Courier (credential exchange)

| Tool | Cost | Purpose |
|------|------|---------|
| `request_credential_channel` | Free | Send a credential template DM to a patron or operator |
| `receive_credentials` | Free | Pick up credentials from vault (instant) or relay (DM) |
| `forget_credentials` | Free | Delete vaulted credentials for a service |

### Npub Proof

| Tool | Cost | Purpose |
|------|------|---------|
| `request_npub_proof` | Free | Send a proof challenge DM to a patron |
| `receive_npub_proof` | Free | Verify the patron's Schnorr-signed proof response |

### Authority Balance

| Tool | Cost | Purpose |
|------|------|---------|
| `check_authority_balance` | Free | Operator's cert-sat balance at the upstream Authority |

### Oracle Delegation

| Tool | Cost | Purpose |
|------|------|---------|
| `oracle_*` | Free | Delegated calls to the DPYC Oracle (community info, tax rates, membership) |

### Pricing Model

| Tool | Cost | Purpose |
|------|------|---------|
| `get_pricing_model` | Free | Current pricing model with tool registry and constraints |
| `set_pricing_model` | Free | Update the pricing model (operator-restricted, requires proof) |
| `list_constraint_types` | Free | Available constraint types and their parameters |

### OpenTimestamps (when `ots_enabled=True`)

| Tool | Cost | Purpose |
|------|------|---------|
| `notarize_ledger` | Free | Anchor all ledger state to Bitcoin via Merkle tree + OTS |
| `get_notarization_proof` | Free | Merkle inclusion proof for a specific patron |
| `list_notarizations` | Free | All Bitcoin-anchored ledger snapshots |

### Conditional Tools

| Tool | Condition | Purpose |
|------|-----------|---------|
| `request_patron_credentials` | `patron_credential_template` set | Open Courier channel for patron-specific credentials |
| `receive_patron_credentials` | `patron_credential_template` set | Pick up patron credentials |
| `begin_oauth` | `oauth_provider` set | Start OAuth2 authorization flow |
| `check_oauth_status` | `oauth_provider` set | Complete OAuth2 flow after browser authorization |

---

## OperatorRuntime

`OperatorRuntime` is the core protocol engine. All DPYC operations — bootstrap, billing, credentials, pricing, constraints — are delegated to it.

### Constructor Parameters

```python
OperatorRuntime(
    # Required
    tool_registry={**STANDARD_IDENTITIES, **YOUR_TOOLS},  # UUID-keyed ToolIdentity map

    # Credential templates
    operator_credential_template=CredentialTemplate(...),   # BTCPay + operator secrets
    patron_credential_template=CredentialTemplate(...),     # Per-patron API keys (optional)
    operator_credential_greeting="Welcome message...",
    patron_credential_greeting="Welcome message...",
    credential_validator=validate_fn,                       # Called on credential receipt

    # Identity & network
    nsec_env_var="TOLLBOOTH_NOSTR_OPERATOR_NSEC",          # Env var name (default)
    service_name="My Service",
    relays=["wss://relay.damus.io"],                       # Override default relays

    # Billing
    purchase_mode="certified",    # "certified" (default) or "direct" (trust-root only)
    operator_settings={},         # Arbitrary operator config dict

    # Constraints
    constraint_gate=None,         # Legacy — use pricing model pipeline instead

    # OpenTimestamps
    ots_enabled=True,
    ots_calendars=["https://a.pool.opentimestamps.org"],

    # Npub proof
    proven_npub_ttl_seconds=3600, # Default proof cache TTL
    npub_proof_field="confirm",   # Field name in proof DM
    npub_proof_greeting="...",
    on_npub_proven=async_callback, # Called when proof verified

    # OAuth2
    oauth_provider=OAuthProviderConfig(...),  # Enables begin_oauth / check_oauth_status

    # Lifecycle
    on_forget=callback,           # Called when credentials are forgotten
)
```

### Key Methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `debit_or_deny(tool_id, npub, *, proof, tool_kwargs)` | `int` (cost) or `dict` (denial) | Gate a tool call through identity, proof, constraints, pricing, and billing |
| `paid_tool(tool_id)` | decorator | Decorator that wraps `debit_or_deny` around a tool function |
| `vault()` | `NeonVault` | Bootstraps and returns the Neon vault (lazy) |
| `ledger_cache()` | `LedgerCache` | Returns the write-behind ledger cache (lazy) |
| `courier()` | `SecureCourierService` | Returns the Secure Courier (lazy) |
| `operator_npub()` | `str` | Derives the operator's npub from nsec |
| `resolve_npub(npub)` | `str` | Validates an npub (bech32 decode) |
| `resolve_tranche_lifetime()` | `int \| None` | Reads tranche lifetime from the pricing model |
| `load_credentials(fields)` | `dict` | Loads operator credentials from the vault |
| `graceful_shutdown()` | — | Flushes caches and closes connections |

---

## Pricing & Constraints

### Tool Pricing

Every tool has a `ToolIdentity` that declares its pricing:

```python
ToolIdentity(
    capability="get_weather",
    category="read",          # read, write, heavy, free
    intent="Current weather",
    pricing_hint_type="ad_valorem",    # fixed or ad_valorem
    pricing_hint_value=1,              # base cost in sats
    pricing_hint_param="symbols",      # parameter for ad valorem scaling
    pricing_hint_min=1,                # minimum cost
)
```

The `ToolPricing` engine computes the final cost: `fixed + ceil(percentage * param_value)`, clamped to `[min, max]`.

### Constraint Pipeline

Operators configure constraints in the pricing model's `pipeline` array. Each step runs on every paid tool call via `debit_or_deny`. The effective cost (after discounts, free trials, surge) is what the patron pays.

| Constraint | Purpose |
|-----------|---------|
| `free_trial` | First N invocations free per patron |
| `happy_hour` | Time-windowed free/discount with recurrence |
| `surge_pricing` | Demand-elastic pricing via global counters |
| `temporal_window` | Time-of-day / day-of-week access windows |
| `finite_supply` | Total call quotas (per patron or global) |
| `periodic_refresh` | Rate limiting with ISO-8601 refresh windows |
| `coupon` | Code-based discounts with expiration |
| `loyalty_discount` | Spend-based tiered discounts |
| `bulk_bonus` | Volume bonuses on credit purchases |
| `patron_proof` | Require per-call Schnorr proof for high-value tools |
| `json_expression` | Custom boolean logic rules |

**Scoping:** Each pipeline step can target specific tools (`tool_ids`) and/or specific patrons (`patron_npubs`, max 10). Unscoped steps apply to all tools and patrons.

### TrancheLifetime

Credit expiry is a property of money, not a per-tool constraint. `TrancheLifetime` in the pricing model sets how long purchased credits remain valid. Each purchase creates a tranche; FIFO consumption.

---

## Credential Delivery

All secrets — BTCPay keys, API tokens, OAuth credentials — are delivered via **Secure Courier**, not environment variables. This is a human-in-the-loop flow using NIP-44 encrypted Nostr DMs:

1. AI agent calls `request_credential_channel(sender_npub=..., service=...)`
2. Operator/patron receives a DM in their Nostr client with a credential template
3. They fill in the fields and reply manually
4. AI agent calls `receive_credentials(sender_npub=..., service=...)`
5. Credentials are validated, encrypted (AES-256-GCM with AAD), and stored in the Neon vault

Vault-first lookup means returning patrons activate instantly — no relay I/O needed.

On first relay receipt, the service sends an `ncred1...` credential card back via DM. Patrons can scan or paste this card to reactivate later.

---

## Identity & Proofs

Every participant is identified by a [Nostr](https://nostr.com/) keypair. The `npub` is your identity on the DPYC Honor Chain. The `nsec` stays with you — never shared, never sent to a service.

Every tool that accepts `npub` also requires `proof` — a JSON-serialized Schnorr-signed kind-27235 Nostr event proving ownership. No proof, no service.

```python
# Proof format (kind 27235, NIP-98 style)
{
    "pubkey": "<hex_pubkey>",
    "kind": 27235,
    "content": "",
    "created_at": 1713000000,
    "tags": [["u", "<tool_name>"]],
    "sig": "<schnorr_signature>"
}
```

Inline proofs must be less than 60 seconds old. Cached proofs (via `ProvenNpubCache`) support patron-chosen TTL up to 24 hours. Consumed event IDs are tracked to prevent replay.

---

## Architecture

```
tollbooth-authority                 tollbooth-dpyc (this wheel)          your-mcp-server
================================   ================================     ================================
Schnorr signing + tax ledger       OperatorRuntime                      OperatorRuntime(tool_registry=...)
certify_purchase -> Nostr cert     register_standard_tools(mcp, ...)    register_standard_tools(mcp, ...)
Authority BTCPay                   debit_or_deny (gate + billing)       @runtime.paid_tool(uuid)
                                   Secure Courier + vault               Domain-specific tools
                                   Pricing resolver + constraints
                                   DPYCRegistry (service discovery)
```

Dependency flows one way: `your-mcp-server --> tollbooth-dpyc`. Authority is a network peer, not a code dependency.

### Ecosystem Repos

| Repo | Role |
|------|------|
| [tollbooth-sample](https://github.com/lonniev/tollbooth-sample) | **Canonical reference operator** — start here |
| [tollbooth-authority](https://github.com/lonniev/tollbooth-authority) | Tax collection, Schnorr signing, purchase order certification |
| [dpyc-community](https://github.com/lonniev/dpyc-community) | Membership registry, governance, service discovery |
| [dpyc-oracle](https://github.com/lonniev/dpyc-oracle) | Community concierge — onboarding, tax rates, membership |
| [thebrain-mcp](https://github.com/lonniev/thebrain-mcp) | Production operator — PersonalBrain knowledge graph |
| [excalibur-mcp](https://github.com/lonniev/excalibur-mcp) | Production operator — social media posting |
| [schwab-mcp](https://github.com/lonniev/schwab-mcp) | Production operator — brokerage data with OAuth2 |
| [tollbooth-oauth2-collector](https://github.com/lonniev/tollbooth-oauth2-collector) | Community utility — OAuth2 callback mailbox |

---

## Install

```bash
# Standard install — includes Nostr relay support (Secure Courier, audit trail)
pip install tollbooth-dpyc[nostr]

# With QR code rendering for credential cards
pip install tollbooth-dpyc[nostr,qr]
```

Core dependencies (`httpx`, `pynostr`) handle identity, proofs, and HTTP. The `[nostr]` extra adds `websocket-client` for Nostr relay I/O — required by every real operator since Secure Courier credential delivery, audit publishing, and bootstrap all use relay DMs. The `[qr]` extra adds `segno` for credential card QR rendering.

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `TOLLBOOTH_NOSTR_OPERATOR_NSEC` | Yes | The single bootstrap key. Identity, Secure Courier, audit signing. |
| `NEON_DATABASE_URL` | Trust-root only | Neon Postgres URL. Only for `purchase_mode="direct"` (Authority). |
| `TOLLBOOTH_NOSTR_RELAYS` | No | Comma-separated relay URLs (overrides defaults). |

Certified operators (the default) do **not** set `NEON_DATABASE_URL`. They discover it from the Authority via encrypted Nostr DM during bootstrap.

All other secrets (BTCPay, API keys, OAuth tokens) flow through Secure Courier — never env vars.

---

## Vault & Persistence

### Neon Postgres

`NeonVault` provides ACID ledger storage with AES-256-GCM encryption (with AAD), optimistic concurrency, and schema-qualified table names via `_t()`.

`NeonCredentialVault` stores credentials encrypted in schema-isolated Postgres (one `op_{hash}` schema per operator, provisioned by `register_operator`).

### Schema Isolation

The Authority uses the `authority` schema. Each certified operator receives an isolated schema (`op_{hash}`) with a dedicated Postgres LOGIN role. `NeonCredentialVault` is schema-aware via `_t()`.

---

## Nostr Integration

### Audit Trail

`NostrAuditPublisher` publishes a kind-30078 NIP-78 event on every vault write. Content is NIP-44v2 encrypted — only the patron's nsec can read it. Wrap any vault with `AuditedVault` to add the trail transparently.

### Low-Balance Notifications

`NotificationManager` sends proactive NIP-44 DMs when a patron's balance crosses thresholds or tranches approach expiration. Fire-and-forget — never blocks the tool path.

### Credential Cards

`ncred1...` bech32-encoded credential cards package encrypted credentials into scannable QR codes. Built on kind-21420 Nostr events with NIP-44v2 encryption.

---

## OpenTimestamps Bitcoin Anchoring

Anchor ledger state to the Bitcoin blockchain for irrefutable, timestamped proofs. `MerkleTree` builds a SHA-256 tree over all `(npub, ledger_json)` entries. `OTSCalendarClient` submits the root hash to public OTS calendar servers (no API key required).

---

## Development

```bash
git clone https://github.com/lonniev/tollbooth-dpyc.git
cd tollbooth-dpyc
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,nostr,qr]"
pytest tests/ -q
```

## Further Reading

[The Phantom Tollbooth on the Lightning Turnpike](https://stablecoin.myshopify.com/blogs/our-value/the-phantom-tollbooth-on-the-lightning-turnpike) — the full story of how we're monetizing the monetization of AI APIs, and then fading to the background.

## Trademarks

DPYC, Tollbooth DPYC, and Don't Pester Your Customer are trademarks of Lonnie VanZandt. See the [TRADEMARKS.md](https://github.com/lonniev/dpyc-community/blob/main/TRADEMARKS.md) in the dpyc-community repository for usage guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Because in the end, the tollbooth was never the destination. It was always just the beginning of the journey.*
