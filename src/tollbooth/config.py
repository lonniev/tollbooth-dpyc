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
    btcpay_tier_config: str | None = None
    btcpay_user_tiers: str | None = None
    seed_balance_sats: int = 0
    authority_npub: str | None = None  # Nostr npub (Schnorr verification)
    credit_ttl_seconds: int | None = 604800  # 7 days default, None = never
    flush_batch_size: int = 10
    flush_staleness_secs: float = 120.0
    # OpenTimestamps Bitcoin anchoring
    ots_enabled: bool = False
    ots_calendars: str | None = None  # Comma-separated URLs; None = defaults
    # Constraint Engine (opt-in)
    constraints_enabled: bool = False
    constraints_config: str | None = None  # JSON string of constraint config
