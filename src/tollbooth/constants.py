"""Constants for Tollbooth micropayment gating."""

MAX_INVOICE_SATS = 1_000_000  # 0.01 BTC cap per invoice
LOW_BALANCE_FLOOR_API_SATS = 100  # minimum warning threshold

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
