"""Configuration via pydantic-settings. Loaded at runtime, never at import time."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class AuthoritySettings(BaseSettings):
    """All env vars for a Tollbooth Authority service.

    Read lazily so test code can patch the environment before the first
    settings access. Each Authority deployment instantiates this (or a
    subclass) on first tool invocation.
    """

    # TheBrain vault for operator ledger persistence (legacy fallback)
    thebrain_api_key: str = ""
    thebrain_vault_brain_id: str = ""
    thebrain_vault_home_id: str = ""

    # Certificate TTL
    certificate_ttl_seconds: int = 600

    # NeonVault (replaces TheBrainVault for ledger persistence)
    neon_database_url: str = ""

    # NOTE: neon_api_key / neon_org_id are NOT env settings. They are OPTIONAL Authority
    # secrets delivered via Secure Courier and read from the operator credential vault by
    # network_books_health (the proactive compute-quota watch). See OPERATOR_CREDENTIAL_TEMPLATE
    # in authority/tools.py — an Authority enables the watch by delivering them, not by env.

    # Nostr audit (optional — enabled when both are set). Relays are governed
    # by the DPYC community registry (relay_registry), not a per-Authority env.
    tollbooth_nostr_audit_enabled: str = ""
    tollbooth_nostr_operator_nsec: str = ""

    # DPYC Registry enforcement (URL comes from tollbooth-dpyc DEFAULT_REGISTRY_URL)
    dpyc_registry_cache_ttl_seconds: int = 300
    dpyc_enforce_membership: bool = False  # opt-in; safe default

    model_config = {"env_file": ".env", "extra": "ignore"}
