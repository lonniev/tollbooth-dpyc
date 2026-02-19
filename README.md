# Tollbooth DPYC

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/pypi/v/tollbooth-dpyc)](https://pypi.org/project/tollbooth-dpyc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)

<p align="center">
  <img src="https://raw.githubusercontent.com/lonniev/tollbooth-dpyc/main/docs/tollbooth-hero.png" alt="Milo drives the Lightning Turnpike — Don't Pester Your Customer" width="800">
</p>

**Don't Pester Your Customer** — Bitcoin Lightning micropayments for MCP servers.

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
```

## What's in the Box

| Module | Purpose |
|--------|---------|
| `TollboothConfig` | Plain frozen dataclass — no pydantic, no env-var reading. Your host constructs it. |
| `UserLedger` | Per-user credit balance with debit/credit/rollback, daily usage logs, JSON serialization. |
| `BTCPayClient` | Async HTTP client for BTCPay Server's Greenfield API — invoices, payouts, health checks. |
| `VaultBackend` | Protocol for pluggable persistence — implement `store_ledger`, `fetch_ledger`, `snapshot_ledger`. |
| `LedgerCache` | In-memory LRU cache with write-behind flush. The hot path for all credit operations. |
| `ToolTier` | Cost tiers for tool-call metering (FREE, READ, WRITE, HEAVY). |
| `tools.credits` | Ready-made tool implementations: `purchase_credits`, `check_payment`, `check_balance`, and more. |

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

## Reference Integration

[thebrain-mcp](https://github.com/lonniev/thebrain-mcp) — the first MCP server powered by Tollbooth. A FastMCP service that gives AI agents access to TheBrain knowledge graphs, with all 40+ tools metered via Tollbooth credits.

## Architecture

Tollbooth is a three-party ecosystem:

| Repo | Role |
|------|------|
| [tollbooth-authority](https://github.com/lonniev/tollbooth-authority) | Tax certification service — EdDSA-signed JWTs, Authority BTCPay |
| **tollbooth-dpyc** (this package) | Operator-side library — credit ledger, BTCPay client, tool gating |
| [thebrain-mcp](https://github.com/lonniev/thebrain-mcp) | Reference integration — first MCP server powered by Tollbooth |

See the [Three-Party Protocol diagram](https://github.com/lonniev/tollbooth-authority/blob/main/docs/diagrams/tollbooth-three-party-protocol.svg) for the full architecture.

```
tollbooth-authority               tollbooth-dpyc (this package)     your-mcp-server (consumer)
================================  ================================  ================================
EdDSA signing + tax ledger        TollboothConfig                   Settings ──constructs──> TollboothConfig
certify_purchase → JWT            UserLedger                        implements VaultBackend
Authority BTCPay                  BTCPayClient                      TOOL_COSTS maps tools to ToolTier
                                  VaultBackend (Protocol)
                                  LedgerCache + credit tools
```

Dependency flows one way: `your-mcp-server --> tollbooth-dpyc`. Authority is a network peer, not a code dependency. Only runtime dependency: `httpx`.

## Further Reading

[The Phantom Tollbooth on the Lightning Turnpike](https://stablecoin.myshopify.com/blogs/our-value/the-phantom-tollbooth-on-the-lightning-turnpike) — the full story of how we're monetizing the monetization of AI APIs, and then fading to the background.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

*Because in the end, the tollbooth was never the destination. It was always just the beginning of the journey.*
