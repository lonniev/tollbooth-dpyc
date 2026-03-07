"""Backwards compatibility tests for TollboothConfig.

Ensures that adding constraints_enabled and constraints_config fields
does not break existing code that constructs TollboothConfig with
only the original fields.
"""

from tollbooth.config import TollboothConfig


class TestConfigBackwardsCompatibility:
    def test_no_args(self):
        """TollboothConfig() with no args still works."""
        config = TollboothConfig()
        assert config.btcpay_host is None
        assert config.btcpay_store_id is None
        assert config.btcpay_api_key is None
        assert config.seed_balance_sats == 0
        assert config.constraints_enabled is False
        assert config.constraints_config is None

    def test_btcpay_only(self):
        """TollboothConfig with only btcpay fields works."""
        config = TollboothConfig(btcpay_host="https://pay.example.com")
        assert config.btcpay_host == "https://pay.example.com"
        assert config.constraints_enabled is False
        assert config.constraints_config is None

    def test_all_existing_fields_present(self):
        """All original fields have correct defaults."""
        config = TollboothConfig()
        assert config.btcpay_host is None
        assert config.btcpay_store_id is None
        assert config.btcpay_api_key is None
        assert config.btcpay_tier_config is None
        assert config.btcpay_user_tiers is None
        assert config.seed_balance_sats == 0
        assert config.authority_npub is None
        assert config.credit_ttl_seconds == 604800
        assert config.flush_batch_size == 10
        assert config.flush_staleness_secs == 120.0
        assert config.ots_enabled is False
        assert config.ots_calendars is None

    def test_all_existing_fields_settable(self):
        """All original fields can be set via keyword args."""
        config = TollboothConfig(
            btcpay_host="https://pay.example.com",
            btcpay_store_id="store123",
            btcpay_api_key="key456",
            btcpay_tier_config='{"tiers":{}}',
            btcpay_user_tiers='{"users":{}}',
            seed_balance_sats=500,
            authority_npub="npub1xxx",
            credit_ttl_seconds=3600,
            flush_batch_size=5,
            flush_staleness_secs=60.0,
            ots_enabled=True,
            ots_calendars="https://cal1,https://cal2",
        )
        assert config.btcpay_host == "https://pay.example.com"
        assert config.btcpay_store_id == "store123"
        assert config.btcpay_api_key == "key456"
        assert config.btcpay_tier_config == '{"tiers":{}}'
        assert config.btcpay_user_tiers == '{"users":{}}'
        assert config.seed_balance_sats == 500
        assert config.authority_npub == "npub1xxx"
        assert config.credit_ttl_seconds == 3600
        assert config.flush_batch_size == 5
        assert config.flush_staleness_secs == 60.0
        assert config.ots_enabled is True
        assert config.ots_calendars == "https://cal1,https://cal2"
        # New fields still default
        assert config.constraints_enabled is False
        assert config.constraints_config is None

    def test_frozen(self):
        """TollboothConfig is frozen (immutable)."""
        config = TollboothConfig()
        try:
            config.btcpay_host = "modified"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass  # expected — frozen dataclass

    def test_new_fields_with_old_fields(self):
        """Can mix old and new fields together."""
        config = TollboothConfig(
            btcpay_host="https://pay.example.com",
            seed_balance_sats=100,
            constraints_enabled=True,
            constraints_config='{"tool_constraints":{}}',
        )
        assert config.btcpay_host == "https://pay.example.com"
        assert config.seed_balance_sats == 100
        assert config.constraints_enabled is True
        assert config.constraints_config == '{"tool_constraints":{}}'
