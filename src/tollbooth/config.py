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
    tollbooth_royalty_address: str | None = None
    tollbooth_royalty_percent: float = 0.02
    tollbooth_royalty_min_sats: int = 10
    authority_public_key: str | None = None
    authority_url: str | None = None
