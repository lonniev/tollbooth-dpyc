# Tollbooth DPYC™

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/tollbooth-dpyc)](https://pypi.org/project/tollbooth-dpyc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

<p align="center">
  <img src="https://raw.githubusercontent.com/lonniev/tollbooth-dpyc/main/docs/tollbooth-hero.png" alt="Milo drives the Lightning Turnpike — Don't Pester Your Customer" width="800">
</p>

**Don't Pester Your Customer™** — Bitcoin Lightning micropayments for MCP servers.

> *The metaphors in this project are drawn with admiration from* The Phantom Tollbooth *by Norton Juster, illustrated by Jules Feiffer (1961). Milo, Tock, the Tollbooth, Dictionopolis, and Digitopolis are creations of Mr. Juster's extraordinary imagination. We just built the payment infrastructure.*

---

## The Problem

Thousands of developers are building [MCP](https://modelcontextprotocol.io/) servers — services that let AI agents like Claude interact with the world. Knowledge graphs, financial data, code repositories, medical records. Each one is a city on the map. But the turnpike between them? Wide open. No toll collectors. No sustainable economics. Just a growing network of roads that nobody's figured out how to fund.

Every MCP operator faces the same question: *how do I keep the lights on?*

Traditional API keys with monthly billing? You're running a SaaS company now. The L402 protocol — Lightning-native pay-per-request? Every single API call requires a payment negotiation. Milo's toy car stops at every intersection to fumble for exact change.

## The Solution

Tollbooth DPYC takes a different approach — one that respects everyone's time:

**Milo drives up to the tollbooth once, buys a roll of tokens with a single Lightning invoice, and drives.** No stops. No negotiations. No per-request friction. The tokens quietly decrement in the background. When the roll runs low, he buys another. The turnpike stays fast.

Prepaid credits over Bitcoin's Lightning Network, gated at the tool level, settled instantly, with no subscription management and no third-party payment processor taking a cut.

## Install

```bash
pip install tollbooth-dpyc

# With Nostr features (Secure Courier, audit trail, notifications, credential exchange)
pip install tollbooth-dpyc[nostr]

# With QR code rendering for credential cards
pip install tollbooth-dpyc[qr]

# Everything
pip install tollbooth-dpyc[nostr,qr]
```

## What's in the Box

### Core

| Module | Purpose |
|--------|---------|
| `TollboothConfig` | Plain frozen dataclass — no pydantic, no env-var reading. Your host constructs it. |
| `UserLedger` | Per-user credit balance with FIFO debit/credit, tranches with TTL, daily usage logs, JSON serialization. |
| `BTCPayClient` | Async HTTP client for BTCPay Server's Greenfield API — invoices, payouts, health checks, sats/BTC conversion. |
| `VaultBackend` | Protocol for pluggable ledger persistence — implement `store_ledger`, `fetch_ledger`, `snapshot_ledger`. |
| `LedgerCache` | In-memory LRU cache with write-behind flush, per-user asyncio locks, dirty tracking. |
| `ToolTier` | Cost tiers for tool-call metering (FREE=0, READ=1, WRITE=5, HEAVY=10 sats per call). |
| `ToolPricing` | Dynamic pricing: fixed + percentage-of-parameter pricing with min/max caps. |

### Actor Protocols

Three protocol interfaces define the DPYC ecosystem roles. Each declares the tool surface for its actor type.

| Module | Purpose |
|--------|---------|
| `OperatorProtocol` | Operator tool surface — balance, statements, Secure Courier, delegation to Authority/Oracle. |
| `OPERATOR_BASE_CATALOG` | Canonical `list[ToolPathInfo]` for all 19 Operator protocol tools. Single source of truth — operators inherit and extend. |
| `AuthorityProtocol` | Authority tool surface — certification, operator registration, tax collection. |
| `AUTHORITY_BASE_CATALOG` | Canonical `list[ToolPathInfo]` for all 11 Authority protocol tools. |
| `OracleProtocol` | Oracle tool surface — community concierge (tax rates, membership, onboarding). |
| `ActorRole` | Enum: OPERATOR, AUTHORITY, ORACLE. |
| `ToolPath` / `ToolPathInfo` | HOT/COLD/DELEGATION path metadata for tool routing. |

### Ready-Made Tool Implementations

The `tollbooth.tools` package provides functions your MCP server wraps as tools. Each takes infrastructure objects (BTCPayClient, LedgerCache) as parameters — you wire them up, Tollbooth handles the logic.

| Function | Purpose |
|----------|---------|
| `purchase_credits_tool` | Creates a BTCPay invoice, records it as pending, returns a checkout link. Validates Authority certificate when `authority_public_key` is configured. |
| `direct_purchase_tool` | Invoice without certificate — for Authority self-purchases (trust anchor has no upstream). |
| `check_payment_tool` | Polls an invoice, credits the balance on settlement, fires the royalty payout. Idempotent. |
| `check_balance_tool` | Returns current balance, usage summary, tier info, and invoice history. Read-only. |
| `restore_credits_tool` | Recovers credits from a paid invoice lost to cache/vault issues. Checks vault first, falls back to BTCPay. |
| `account_statement_tool` | 30-day purchase and usage history for a user. |
| `btcpay_status_tool` | Diagnostics: BTCPay connectivity, store name, API key permissions, royalty config. |
| `compute_low_balance_warning` | Pure function — returns a warning dict if balance is below threshold, `None` if healthy. |
| `anchor_ledger_tool` | Creates a Merkle tree of all ledgers and submits root hash to OTS calendar servers. |
| `get_anchor_proof_tool` | Retrieves a Merkle inclusion proof for a specific user's ledger entry. |
| `list_anchors_tool` | Lists all Bitcoin-anchored ledger snapshots with timestamps and verification status. |

### Certificates & Verification

| Module | Purpose |
|--------|---------|
| `verify_certificate_auto` | Detects certificate format and verifies: Nostr kind-30079 Schnorr (preferred) or Ed25519 JWT (legacy). |
| `verify_nostr_certificate` | Nostr event (kind 30079) certificate verification using Schnorr/BIP-340 via pynostr. |
| Anti-replay JTI store | Built into certificate verification — prevents double-spend of purchase certificates. |

### Authority Client

| Module | Purpose |
|--------|---------|
| `AuthorityCertifier` | Server-to-server MCP client for auto-certifying credit purchases. Opens a short-lived `fastmcp.Client` SSE connection with Horizon OAuth auto-negotiation. |
| `AuthorityCertifyError` | Raised when Authority certification fails (connection, auth, or tool error). |

### Oracle Client

| Module | Purpose |
|--------|---------|
| `OracleClient` | Server-to-server MCP client for delegating Oracle tool calls. Opens a short-lived `fastmcp.Client` SSE connection per call. Oracle tools are free and unauthenticated. |
| `OracleClientError` | Raised when an Oracle delegation call fails (connection or parse error). |

### DPYC Registry

| Module | Purpose |
|--------|---------|
| `DPYCRegistry` | Cached HTTPS fetch of the [dpyc-community](https://github.com/lonniev/dpyc-community) `members.json` registry. Resolves membership, upstream authorities, and service URLs. |
| `resolve_authority_service` | Convenience: finds an operator's upstream Authority service URL in one call. |
| `resolve_authority_npub` | Finds an operator's upstream Authority npub from the registry. |
| `resolve_oracle_service` | Walks the authority chain to the Prime Authority and returns the Oracle service URL. |

### Vault Backends

| Module | Purpose |
|--------|---------|
| `TheBrainVault` | Stores ledger state as thought notes in a TheBrain knowledge graph. Daily-child pattern with link-based member discovery. |
| `NeonVault` | Neon serverless Postgres with ACID transactions, optimistic concurrency, and append-only audit journal. |
| `NeonCredentialVault` | Credential storage on Neon Postgres (implements `CredentialVaultBackend`). |
| `CredentialVaultBackend` | Protocol for pluggable credential storage (store/fetch/delete by service+npub). |

Implement the `VaultBackend` Protocol to add your own (Redis, S3, SQLite, etc.).

### Nostr Integration (optional: `pip install tollbooth-dpyc[nostr]`)

| Module | Purpose |
|--------|---------|
| `SecureCourierService` | High-level wrapper for 3 MCP tools: open channel, receive credentials, forget credentials. On first-time relay receipt, sends the `ncred1...` credential card back to the patron via Nostr DM for scan-and-paste reuse. |
| `NostrCredentialExchange` | NIP-44/NIP-04 encrypted DM credential delivery with vault-first lookup. Relay copies deleted via NIP-09 after pickup. |
| `CredentialTemplate` / `FieldSpec` | Schema and validation for credential payloads. |
| `NostrAuditPublisher` | Publishes kind-30078 NIP-78 events on every vault write for tamper-evident audit trail. NIP-44v2 encrypted. |
| `AuditedVault` | Wraps any `VaultBackend` to add the audit trail transparently. |
| `NotificationManager` | Proactive NIP-44 DMs when patron balance crosses thresholds or tranches approach expiration. Fire-and-forget. |
| `courier_health` / `courier_ping` | Relay diagnostics and outbound DM smoke tests. |

### Credential Cards (optional: `pip install tollbooth-dpyc[qr]`)

| Module | Purpose |
|--------|---------|
| `encode_credential_card` | Creates bech32-encoded `ncred1...` credential cards (kind 21420 Nostr events, NIP-44 encrypted). |
| `decode_credential_card` | Decodes and verifies credential cards with expiration checking. |
| `render_qr` | Renders credential cards as QR codes via segno. |

### Constraint Engine

| Module | Purpose |
|--------|---------|
| `ConstraintEngine` | Evaluates constraint lists with configurable logic: `ALL_MUST_PASS`, `ANY_MUST_PASS`, `FIRST_MATCH`. |
| `ConstraintGate` | Drop-in middleware integrating constraints with the `_debit_or_error` pattern. |
| `TemporalWindowConstraint` | Time-of-day / day-of-week access windows. |
| `FiniteSupplyConstraint` | Total call quotas per patron or globally. |
| `PeriodicRefreshConstraint` | ISO-8601 duration refresh windows (e.g., "10 calls per hour"). |
| `CouponConstraint` | Code-based discounts with expiration and redemption limits. |
| `FreeTrialConstraint` | First N invocations free per patron. |
| `LoyaltyDiscountConstraint` | Spend-based tiered discounts. |
| `BulkBonusConstraint` | Volume bonuses on credit purchases. |
| `HappyHourConstraint` | Time-based discount windows. |
| `JsonExpressionConstraint` | Safe tree-based boolean logic evaluator for custom rules. |

### OpenTimestamps Bitcoin Anchoring

| Module | Purpose |
|--------|---------|
| `MerkleTree` | SHA-256 Merkle tree over all `(npub, ledger_json)` entries. |
| `OTSCalendarClient` | Submits root hash to public OTS calendar servers (no API key required). |
| `InclusionProof` | Verifiable Merkle proofs for individual patron entries. |

### Utilities

| Module | Purpose |
|--------|---------|
| `make_slug_tool` | Decorator for slug-prefixed MCP tool registration (e.g., `brain_check_balance`). |
| `lnurl` | LUD-06/LUD-16 Lightning address resolution (parse → .well-known → callback → BOLT11). |

## Quick Start

```python
from tollbooth import TollboothConfig, UserLedger, BTCPayClient, LedgerCache

# Configure — your host reads env vars, Tollbooth gets a plain dataclass
config = TollboothConfig(
    btcpay_host="https://your-btcpay.example.com",
    btcpay_store_id="your-store-id",
    btcpay_api_key="your-api-key",
    tollbooth_royalty_address="tollbooth@btcpay.digitalthread.link",
)

# Create a BTCPay client
async with BTCPayClient(config.btcpay_host, config.btcpay_api_key, config.btcpay_store_id) as client:
    # Create an invoice for 1000 sats
    invoice = await client.create_invoice(1000, metadata={"user": "milo"})
    print(f"Pay here: {invoice['checkoutLink']}")
```

## Configuration

`TollboothConfig` is a plain frozen dataclass. Your host application constructs it from its own settings (env vars, pydantic-settings, YAML — whatever you prefer). Tollbooth never reads environment variables directly.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `btcpay_host` | `str \| None` | `None` | BTCPay Server URL for creating invoices and checking payments |
| `btcpay_store_id` | `str \| None` | `None` | BTCPay store ID — each operator runs their own store |
| `btcpay_api_key` | `str \| None` | `None` | BTCPay API key — must have invoice + payout permissions |
| `btcpay_tier_config` | `str \| None` | `None` | JSON string mapping tier names to credit multipliers |
| `btcpay_user_tiers` | `str \| None` | `None` | JSON string mapping user IDs to tier names |
| `seed_balance_sats` | `int` | `0` | Free starter balance granted to new users (0 to disable) |
| `tollbooth_royalty_address` | `str \| None` | `None` | Lightning Address for the 2% royalty payout to the Tollbooth originator |
| `tollbooth_royalty_percent` | `float` | `0.02` | Royalty percentage (0.02 = 2%) |
| `tollbooth_royalty_min_sats` | `int` | `10` | Minimum royalty payout in sats (below this, no payout fires) |
| `authority_public_key` | `str \| None` | `None` | Authority's Nostr npub (Schnorr certificates) or Ed25519 public key (legacy JWT). Required for `purchase_credits`. |
| `credit_ttl_seconds` | `int \| None` | `None` | Credit expiration in seconds. `None` = no expiration. |
| `constraints_enabled` | `bool` | `False` | Enable constraint engine evaluation on tool calls. |
| `constraints_config` | `str \| None` | `None` | JSON constraint configuration (see Constraint Engine section). |

## The Three-Party Settlement

Here's where the story takes a turn that even Milo wouldn't expect.

We didn't build Tollbooth to sell. We built it to **give away** — like the Massachusetts Turnpike Authority. The Authority doesn't operate every toll plaza. Independent operators run the booths. What the Authority does is simpler: it collects a small percentage of every fare that flows through infrastructure it designed.

When a user purchases credits, the settlement is three-party:

1. **Milo** pays the operator's Lightning invoice
2. **The operator's** BTCPay Server credits Milo's balance
3. **Automatically, in the background** — BTCPay creates a small payout to the Tollbooth originator's Lightning Address

A royalty. Two percent of the fare. The operator sees it transparently in their BTCPay dashboard. Milo never knows it happened.

The enforcement is both technical and social. At startup, Tollbooth inspects the operator's BTCPay API key permissions. If the key lacks payout capability, **Tollbooth refuses to start**. Not a warning. A hard stop. The social contract, made executable.

## The Economics

**For Milo (the user):** Nothing changes. Buy credits, use tools, drive the turnpike.

**For the operator:** A free, production-tested monetization framework. No license fee. The 2% royalty is a rounding error compared to the revenue you couldn't collect before. The tollbooth pays for itself on the first transaction.

**For the ecosystem:** Revenue scales with adoption, not effort. Every new MCP server that installs Tollbooth becomes a node in the Lightning economy. The infrastructure hums along — collecting its modest fare, maintaining the roads, and making sure the turnpike stays open for everyone.

*It's the transition from mining fees to transaction fees. You stop competing on compute and start collecting on flow.*

## Architecture

Tollbooth is a three-party ecosystem built on the [DPYC Honor Chain](https://github.com/lonniev/dpyc-community):

| Repo | Role |
|------|------|
| [tollbooth-authority](https://github.com/lonniev/tollbooth-authority) | The institution — tax collection, Schnorr signing, purchase order certification |
| **tollbooth-dpyc** (this package) | The booth — operator-side credit ledger, BTCPay client, tool gating, auto-certification |
| [dpyc-community](https://github.com/lonniev/dpyc-community) | The registry — membership, governance, and service discovery |
| [dpyc-oracle](https://github.com/lonniev/dpyc-oracle) | The concierge — community onboarding, tax rates, membership lookup |
| [thebrain-mcp](https://github.com/lonniev/thebrain-mcp) | Reference operator — PersonalBrain knowledge graph, 40+ metered tools |
| [excalibur-mcp](https://github.com/lonniev/excalibur-mcp) | Reference operator — social media posting via Tollbooth credits |

```
tollbooth-authority               tollbooth-dpyc (this package)     your-mcp-server (consumer)
================================  ================================  ================================
Schnorr signing + tax ledger      TollboothConfig                   Settings ──constructs──> TollboothConfig
certify_purchase → Nostr cert     UserLedger + LedgerCache          implements VaultBackend
Authority BTCPay                  BTCPayClient                      TOOL_COSTS maps tools to ToolTier
                                  VaultBackend (Protocol)           AuthorityCertifier (auto-certify)
                                  DPYCRegistry (service discovery)
                                  Constraint engine + pricing
```

Dependency flows one way: `your-mcp-server --> tollbooth-dpyc`. Authority is a network peer, not a code dependency. Core runtime dependencies: `httpx`, `pynostr`. Nostr relay features require `websocket-client` (install with `pip install tollbooth-dpyc[nostr]`).

## Auto-Certification

`AuthorityCertifier` lets operator servers auto-obtain Authority certificates during `purchase_credits` — one tool call instead of two. Opens a short-lived `fastmcp.Client` SSE connection with Horizon OAuth, calls `certify_credits`, and returns the parsed certificate dict.

```python
from tollbooth import AuthorityCertifier

certifier = AuthorityCertifier(
    authority_url="https://authority.fastmcp.app/mcp",
    operator_npub="npub1...",
)
cert = await certifier.certify(amount_sats=100)
# cert["certificate"], cert["jti"], cert["net_sats"], ...
```

The `DPYCRegistry` resolves service URLs automatically:

```python
from tollbooth import resolve_authority_service

service = await resolve_authority_service("npub1operator...")
# service["url"] → "https://authority.fastmcp.app/mcp"
```

## Oracle Delegation

`OracleClient` lets operator servers delegate community queries to the DPYC Oracle — no separate MCP connection from the AI agent needed. Oracle tools are free and unauthenticated.

```python
from tollbooth import OracleClient, resolve_oracle_service

# Resolve Oracle URL via registry (walks authority chain to Prime Authority)
service = await resolve_oracle_service("npub1operator...")
# service["url"] → "https://dpyc-oracle.fastmcp.app/mcp"

# Call any Oracle tool
client = OracleClient(service["url"])
result = await client.call_tool("get_tax_rate")
# result → {"rate_percent": 2, "min_sats": 10, ...}
```

## Constraint Engine

The constraint engine lets operators define access rules and pricing modifiers for their tools. Constraints are evaluated before every tool call via `ConstraintGate`.

**Evaluation modes:** `ALL_MUST_PASS` (default), `ANY_MUST_PASS`, `FIRST_MATCH`.

Configure via JSON:

```json
{
  "constraints": [
    {"type": "free_trial", "max_free_calls": 5},
    {"type": "temporal_window", "start_hour": 9, "end_hour": 17, "timezone": "US/Eastern"},
    {"type": "happy_hour", "start_hour": 17, "end_hour": 19, "discount_percent": 50}
  ],
  "mode": "ALL_MUST_PASS"
}
```

## Secure Courier

Credentials delivered via encrypted Nostr DMs — they never appear in the chat window. `SecureCourierService` wraps four MCP tools (session status, open channel, receive, forget) with NIP-44 encryption, template validation, and vault-first lookup.

On first-time relay receipt, the service automatically sends the `ncred1...` credential card back to the patron via Nostr DM. Patrons can scan or paste this card to reactivate their session later — no Courier exchange needed. Vault restores skip the DM (no spam on reconnect). Delivery is fire-and-forget; failures never block the credential flow.

`CredentialVaultBackend` is a Protocol for pluggable credential storage. After first delivery, `receive()` checks the vault before touching relays. Relay copies are deleted via NIP-09 after pickup.

## Credential Cards

`ncred1...` bech32-encoded credential cards package encrypted credentials into scannable QR codes. Built on kind-21420 Nostr events with NIP-44v2 encryption. Install with `pip install tollbooth-dpyc[qr]` for QR rendering.

## Nostr Audit Trail

`NostrAuditPublisher` publishes a kind-30078 NIP-78 event on every vault write. Content is NIP-44v2 encrypted (ECDH + ChaCha20-Poly1305) so only the patron's nsec can read it. Wrap any `VaultBackend` with `AuditedVault` to add the audit trail transparently.

## Low-Balance Notifications

`NotificationManager` sends proactive NIP-44 DMs when a patron's balance crosses configurable thresholds (WARNING, CRITICAL) or when credit tranches approach expiration. Delivery is fire-and-forget on a daemon thread — never blocks the tool path.

## OpenTimestamps Bitcoin Anchoring

Anchor ledger state to the Bitcoin blockchain for irrefutable, timestamped proofs. `MerkleTree` builds a SHA-256 tree over all `(npub, ledger_json)` entries. `OTSCalendarClient` submits the root hash to public OTS calendar servers (no API key required). `InclusionProof` provides verifiable Merkle proofs for individual patron entries.

## DPYC Identity (Nostr npub)

Every participant in the Tollbooth ecosystem — Authorities, Operators, and Users — is identified by a [Nostr](https://nostr.com/) keypair. The `npub` (public key) is your identity on the DPYC Honor Chain. The `nsec` (private key) stays with you — never shared, never sent to a service.

Generate a keypair:

```bash
pip install nostr-sdk
python scripts/generate_nostr_keypair.py
```

Then set the appropriate environment variable in your `.env`:

| Role | Variable | Purpose |
|------|----------|---------|
| Authority | `DPYC_AUTHORITY_NPUB` | This Authority's identity on the Honor Chain |
| Authority | `DPYC_UPSTREAM_AUTHORITY_NPUB` | The Authority above in the chain (empty for Prime) |
| Operator | `DPYC_OPERATOR_NPUB` | This Operator's identity on the Honor Chain |
| Operator | `DPYC_AUTHORITY_NPUB` | The Authority this Operator is registered with |

Users provide their npub via the Secure Courier credential exchange flow — no env var needed.

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
