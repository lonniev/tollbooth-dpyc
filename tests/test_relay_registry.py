"""Tests for the DPYC relay registry (single source of truth for relays)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tollbooth.relay_registry import RelayRegistry, RelayRegistryError


def _resp(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def _client_returning(resp_or_exc):
    """Build a mock httpx.Client context manager whose .get returns/raises."""
    client = MagicMock()
    if isinstance(resp_or_exc, Exception):
        client.get.side_effect = resp_or_exc
    else:
        client.get.return_value = resp_or_exc
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=client)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


VALID = {
    "version": "1.0.0",
    "updated_at": "2026-07-08T12:00:00Z",
    "relays": [
        {"url": "wss://relay.damus.io"},
        {"url": "wss://relay.primal.net", "primary": True},
        {"url": "wss://nos.lol"},
    ],
}


class TestRelayRegistry:
    def test_fetch_orders_primary_first(self):
        reg = RelayRegistry(url="http://x/relays.json")
        with patch("httpx.Client", return_value=_client_returning(_resp(VALID))):
            relays = reg.relays()
        # primary jumps to the front; the rest keep array order.
        assert relays == [
            "wss://relay.primal.net",
            "wss://relay.damus.io",
            "wss://nos.lol",
        ]

    def test_cache_hit_skips_second_fetch(self):
        reg = RelayRegistry(url="http://x/relays.json")
        mk = patch("httpx.Client", return_value=_client_returning(_resp(VALID)))
        with mk as m:
            reg.relays()
            reg.relays()  # within TTL — must not re-fetch
            assert m.return_value.__enter__.return_value.get.call_count == 1

    def test_stale_cache_refetches_after_ttl(self):
        reg = RelayRegistry(url="http://x/relays.json", cache_ttl_seconds=0)
        with patch("httpx.Client", return_value=_client_returning(_resp(VALID))) as m:
            reg.relays()
            reg.relays()  # TTL=0 → always stale → re-fetch
            assert m.return_value.__enter__.return_value.get.call_count == 2

    def test_stale_if_error_serves_last_known_good(self):
        reg = RelayRegistry(url="http://x/relays.json", cache_ttl_seconds=0)
        # First call succeeds and populates the cache.
        with patch("httpx.Client", return_value=_client_returning(_resp(VALID))):
            first = reg.relays()
        # Force past the failure backoff so the next call actually re-fetches.
        reg._retry_after = 0.0
        # Second refresh fails — stale cache is served rather than raising.
        with patch("httpx.Client", return_value=_client_returning(RuntimeError("down"))):
            second = reg.relays()
        assert second == first

    def test_cold_cache_unreachable_raises(self):
        reg = RelayRegistry(url="http://x/relays.json")
        with patch("httpx.Client", return_value=_client_returning(RuntimeError("down"))):
            with pytest.raises(RelayRegistryError):
                reg.relays()

    def test_rejects_non_wss_and_empty(self):
        reg = RelayRegistry(url="http://x/relays.json")
        bad = {"version": "1.0.0", "updated_at": "x", "relays": [{"url": "http://nope"}]}
        with patch("httpx.Client", return_value=_client_returning(_resp(bad))):
            with pytest.raises(RelayRegistryError):
                reg.relays()

    def test_accepts_bare_string_relays(self):
        reg = RelayRegistry(url="http://x/relays.json")
        payload = {"version": "1.0.0", "updated_at": "x",
                   "relays": ["wss://a.example", "wss://b.example"]}
        with patch("httpx.Client", return_value=_client_returning(_resp(payload))):
            assert reg.relays() == ["wss://a.example", "wss://b.example"]

    def test_invalidate_forces_refetch(self):
        reg = RelayRegistry(url="http://x/relays.json")
        with patch("httpx.Client", return_value=_client_returning(_resp(VALID))) as m:
            reg.relays()
            reg.invalidate()
            reg.relays()
            assert m.return_value.__enter__.return_value.get.call_count == 2
