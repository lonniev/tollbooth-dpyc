"""Ledger cache health, as the tools surface it."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tollbooth.ledger_cache import LedgerCache


def _make_cache() -> LedgerCache:
    v = AsyncMock()
    v.store_ledger = AsyncMock(return_value=1)
    v.fetch_ledger = AsyncMock(return_value=None)
    return LedgerCache(v, maxsize=20, fold_interval_secs=600)


class TestLedgerCacheHealth:
    def test_health_empty_cache(self) -> None:
        h = _make_cache().health()
        assert h["cache_size"] == 0
        assert h["pending_usage"] == 0
        assert h["usage_folds"] == 0
        assert h["usage_fold_running"] is False

    @pytest.mark.asyncio
    async def test_health_counts_patrons_with_pending_usage(self) -> None:
        cache = _make_cache()
        cache.note_usage("a", "tool")
        cache.note_usage("a", "tool")
        cache.note_usage("b", "tool")
        assert cache.health()["pending_usage"] == 2
        assert await cache.fold_usage() == 2
        h = cache.health()
        assert h["pending_usage"] == 0
        assert h["usage_folds"] == 2

    @pytest.mark.asyncio
    async def test_health_reports_the_fold_loop(self) -> None:
        cache = _make_cache()
        await cache.start_usage_fold()
        assert cache.health()["usage_fold_running"] is True
        await cache.stop()
        assert cache.health()["usage_fold_running"] is False


class TestCheckBalanceCacheHealth:
    @pytest.mark.asyncio
    async def test_check_balance_includes_cache_health(self) -> None:
        from tollbooth.tools.credits import check_balance_tool

        cache = _make_cache()
        await cache.get("test-user")
        result = await check_balance_tool(cache, "test-user")
        result["cache_health"] = cache.health()
        assert result["success"] is True
        assert result["cache_health"]["cache_size"] >= 0
        assert "usage_fold_running" in result["cache_health"]


class TestBTCPayStatusCacheHealth:
    @pytest.mark.asyncio
    async def test_btcpay_status_cache_health_present(self) -> None:
        from tollbooth.tools.credits import btcpay_status_tool

        cache = _make_cache()
        settings = MagicMock(
            btcpay_host="", btcpay_store_id="", btcpay_api_key="",
            btcpay_tier_config=None, btcpay_user_tiers=None,
        )
        result = await btcpay_status_tool(settings, None)
        result["cache_health"] = cache.health()
        assert result["cache_health"]["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_btcpay_status_cache_health_none_when_no_cache(self) -> None:
        from tollbooth.tools.credits import btcpay_status_tool

        settings = MagicMock(
            btcpay_host="", btcpay_store_id="", btcpay_api_key="",
            btcpay_tier_config=None, btcpay_user_tiers=None,
        )
        result = await btcpay_status_tool(settings, None)
        result["cache_health"] = None
        assert result["cache_health"] is None
