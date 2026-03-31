"""Tollbooth configuration — plain frozen dataclass, no pydantic.

The host application constructs this from its own settings (env vars,
pydantic-settings, etc.) and passes it to Tollbooth tools.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TollboothConfig:
    btcpay_host: str | None = None
    btcpay_store_id: str | None = None
    btcpay_api_key: str | None = None
    seed_balance_sats: int = 0
    authority_npub: str | None = None  # Nostr npub (Schnorr verification)
    flush_batch_size: int = 10
    flush_staleness_secs: float = 120.0
    # OpenTimestamps Bitcoin anchoring
    ots_enabled: bool = True
    ots_calendars: str | None = None  # Comma-separated URLs; None = defaults
    # Constraint Engine (opt-in)
    constraints_enabled: bool = False
    constraints_config: str | None = None  # JSON string of constraint config
    # Pricing Model cache TTL (PricingResolver)
    pricing_cache_ttl_seconds: float = 300.0
