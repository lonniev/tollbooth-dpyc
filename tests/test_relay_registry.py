"""Tests for the DPYC relay registry (relays sourced from the Oracle).

The relays.json parsing (primary-first ordering, wss:// filtering) now lives in
the Oracle; the SDK-side registry just caches whatever ordered list the Oracle
returns. These tests exercise the cache / stale-if-error / seed behavior against
a mocked Oracle client.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.relay_registry import RelayRegistry, RelayRegistryError

RELAYS = ["wss://relay.primal.net", "wss://relay.damus.io", "wss://nos.lol"]


def _oracle_factory(relays_or_exc):
    """Patch target for ``default_oracle_client`` → returns one shared client."""
    client = MagicMock()
    if isinstance(relays_or_exc, Exception):
        client.get_relays = AsyncMock(side_effect=relays_or_exc)
    else:
        client.get_relays = AsyncMock(return_value=relays_or_exc)
    factory = MagicMock(return_value=client)
    factory.client = client  # expose for call-count assertions
    return factory


class TestRelayRegistry:
    def test_fetch_returns_oracle_set(self):
        reg = RelayRegistry()
        with patch("tollbooth.oracle_client.default_oracle_client", _oracle_factory(RELAYS)):
            assert reg.relays() == RELAYS

    def test_cache_hit_skips_second_fetch(self):
        reg = RelayRegistry()
        factory = _oracle_factory(RELAYS)
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            reg.relays()
            reg.relays()  # within TTL — must not re-fetch
        assert factory.client.get_relays.call_count == 1

    def test_stale_cache_refetches_after_ttl(self):
        reg = RelayRegistry(cache_ttl_seconds=0)
        factory = _oracle_factory(RELAYS)
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            reg.relays()
            reg.relays()  # TTL=0 → always stale → re-fetch
        assert factory.client.get_relays.call_count == 2

    def test_stale_if_error_serves_last_known_good(self):
        reg = RelayRegistry(cache_ttl_seconds=0)
        with patch("tollbooth.oracle_client.default_oracle_client", _oracle_factory(RELAYS)):
            first = reg.relays()
        reg._retry_after = 0.0  # skip the post-failure backoff window
        with patch(
            "tollbooth.oracle_client.default_oracle_client",
            _oracle_factory(RuntimeError("oracle down")),
        ):
            second = reg.relays()
        assert second == first

    def test_cold_cache_unreachable_raises(self):
        reg = RelayRegistry()
        with patch(
            "tollbooth.oracle_client.default_oracle_client",
            _oracle_factory(RuntimeError("oracle down")),
        ), pytest.raises(RelayRegistryError):
            reg.relays()

    def test_seed_populates_without_any_fetch(self):
        reg = RelayRegistry()
        reg.seed(RELAYS)
        factory = _oracle_factory(RuntimeError("must not be called"))
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            assert reg.relays() == RELAYS
        assert factory.client.get_relays.call_count == 0

    def test_invalidate_forces_refetch(self):
        reg = RelayRegistry()
        factory = _oracle_factory(RELAYS)
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            reg.relays()
            reg.invalidate()
            reg.relays()
        assert factory.client.get_relays.call_count == 2

    def test_is_warm_reflects_fresh_cache(self):
        reg = RelayRegistry()
        assert reg.is_warm() is False
        reg.seed(RELAYS)
        assert reg.is_warm() is True
        # A zero-TTL registry is stale the moment it is seeded.
        stale = RelayRegistry(cache_ttl_seconds=0)
        stale.seed(RELAYS)
        assert stale.is_warm() is False


class TestEnsureRelaysSeeded:
    """The async warm path that self-provisioning actors (Authorities) must call so
    the synchronous Secure Courier never reaches the asyncio.run bridge."""

    def setup_method(self):
        from tollbooth.relay_registry import invalidate_relays_cache
        invalidate_relays_cache()  # reset the process-global cache before each case

    def teardown_method(self):
        from tollbooth.relay_registry import invalidate_relays_cache
        invalidate_relays_cache()

    async def test_seeds_the_global_cache_from_oracle(self):
        from tollbooth.relay_registry import ensure_relays_seeded, get_relays
        factory = _oracle_factory(RELAYS)
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            await ensure_relays_seeded()
            # The sync consumer now serves the warm cache — no second fetch.
            assert get_relays() == RELAYS
        assert factory.client.get_relays.call_count == 1

    async def test_idempotent_when_already_warm(self):
        from tollbooth.relay_registry import ensure_relays_seeded, seed_relays
        seed_relays(RELAYS)
        factory = _oracle_factory(RuntimeError("must not be called"))
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            await ensure_relays_seeded()  # cache warm → no fetch
        assert factory.client.get_relays.call_count == 0

    async def test_best_effort_on_oracle_failure_leaves_cache_cold(self):
        from tollbooth.relay_registry import _registry, ensure_relays_seeded
        factory = _oracle_factory(RuntimeError("oracle down"))
        with patch("tollbooth.oracle_client.default_oracle_client", factory):
            await ensure_relays_seeded()  # must NOT raise
        assert _registry.is_warm() is False
