"""Tests for LedgerCache: LRU eviction, background flush, concurrency."""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock

from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll until predicate() is true or timeout elapses.

    Returns whether it became true. Bounded wait-until-condition — returns as
    soon as the condition holds (no fixed sleep guessing at flush timing).
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def _mock_vault(ledger_json: str | None = None, fail_store: bool = False):
    """Create a mock vault with fetch_ledger/store_ledger."""
    vault = AsyncMock()
    vault.fetch_ledger = AsyncMock(return_value=ledger_json)
    if fail_store:
        vault.store_ledger = AsyncMock(side_effect=Exception("vault write failed"))
    else:
        vault.store_ledger = AsyncMock(return_value="ledger-thought-id")
    return vault


def _funded_vault(balance: int = 500) -> AsyncMock:
    """Create a mock vault that returns a funded ledger."""
    ledger = UserLedger()
    ledger.credit_deposit(balance, "seed")
    return _mock_vault(ledger_json=ledger.to_json())


# ---------------------------------------------------------------------------
# Cache miss / hit
# ---------------------------------------------------------------------------


class TestLedgerCacheGetMiss:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_fresh_ledger(self) -> None:
        vault = _mock_vault(ledger_json=None)
        cache = LedgerCache(vault, maxsize=5)
        ledger = await cache.get("user1")
        assert ledger.balance_api_sats == 0
        vault.fetch_ledger.assert_called_once_with("user1")

    @pytest.mark.asyncio
    async def test_cache_miss_loads_from_vault(self) -> None:
        stored = UserLedger()
        stored.credit_deposit(500, "seed")
        vault = _mock_vault(ledger_json=stored.to_json())
        cache = LedgerCache(vault, maxsize=5)
        ledger = await cache.get("user1")
        assert ledger.balance_api_sats == 500

    @pytest.mark.asyncio
    async def test_cache_miss_vault_error_returns_fresh(self) -> None:
        vault = AsyncMock()
        vault.fetch_ledger = AsyncMock(side_effect=Exception("network error"))
        cache = LedgerCache(vault, maxsize=5)
        ledger = await cache.get("user1")
        assert ledger.balance_api_sats == 0


class TestLedgerCacheGetHit:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_same_object(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        ledger1 = await cache.get("user1")
        ledger2 = await cache.get("user1")
        assert ledger1 is ledger2
        # Vault should only be called once
        vault.fetch_ledger.assert_called_once()

    @pytest.mark.asyncio
    async def test_mutations_visible_on_cache_hit(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        ledger = await cache.get("user1")
        ledger.credit_deposit(999, "test")
        cache.mark_dirty("user1")
        ledger2 = await cache.get("user1")
        assert ledger2.balance_api_sats == 999


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLedgerCacheEviction:
    @pytest.mark.asyncio
    async def test_eviction_at_capacity(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=2)
        await cache.get("user1")
        await cache.get("user2")
        await cache.get("user3")  # should evict user1
        assert cache.size == 2
        # user1 was evicted, next access should reload from vault
        vault.fetch_ledger.reset_mock()
        await cache.get("user1")
        vault.fetch_ledger.assert_called_with("user1")

    @pytest.mark.asyncio
    async def test_eviction_flushes_dirty_entry(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=2)
        ledger1 = await cache.get("user1")
        ledger1.credit_deposit(42, "test")
        cache.mark_dirty("user1")
        await cache.get("user2")
        await cache.get("user3")  # evicts user1 (fire-and-forget flush)
        await asyncio.sleep(0)  # let background task run
        vault.store_ledger.assert_called_once()
        args = vault.store_ledger.call_args[0]
        assert args[0] == "user1"

    @pytest.mark.asyncio
    async def test_eviction_does_not_flush_clean_entry(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=2)
        await cache.get("user1")  # clean
        await cache.get("user2")
        await cache.get("user3")  # evicts user1 (clean)
        vault.store_ledger.assert_not_called()

    @pytest.mark.asyncio
    async def test_lru_order_access_refreshes(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=2)
        await cache.get("user1")
        await cache.get("user2")
        await cache.get("user1")   # refresh user1 — user2 is now LRU
        await cache.get("user3")   # evicts user2
        assert cache.size == 2
        # user1 should still be cached (no vault call)
        vault.fetch_ledger.reset_mock()
        await cache.get("user1")
        vault.fetch_ledger.assert_not_called()


# ---------------------------------------------------------------------------
# Dirty tracking and flush
# ---------------------------------------------------------------------------


class TestLedgerCacheFlush:
    @pytest.mark.asyncio
    async def test_mark_dirty_and_flush(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        ledger = await cache.get("user1")
        ledger.credit_deposit(100, "test")
        cache.mark_dirty("user1")
        # dirty_count should be 1
        assert cache._entries["user1"].dirty_count == 1
        count = await cache.flush_dirty()
        assert count == 1
        vault.store_ledger.assert_called_once()
        # After flush, dirty_count should be reset
        assert cache._entries["user1"].dirty_count == 0

    @pytest.mark.asyncio
    async def test_flush_skips_clean_entries(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")  # clean
        count = await cache.flush_dirty()
        assert count == 0
        vault.store_ledger.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_clears_dirty_flag(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        cache.mark_dirty("user1")
        await cache.flush_dirty()
        # Second flush should be a no-op
        vault.store_ledger.reset_mock()
        count = await cache.flush_dirty()
        assert count == 0
        vault.store_ledger.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_failure_keeps_dirty(self) -> None:
        vault = _mock_vault(fail_store=True)
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        cache.mark_dirty("user1")
        count = await cache.flush_dirty()
        assert count == 0  # failed, entry still dirty
        # dirty_count should be preserved on failure
        assert cache._entries["user1"].dirty_count == 1
        # Retry should attempt again
        vault.store_ledger.reset_mock()
        vault.store_ledger = AsyncMock(return_value="ok")
        count = await cache.flush_dirty()
        assert count == 1

    @pytest.mark.asyncio
    async def test_flush_all(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        await cache.get("user2")
        cache.mark_dirty("user1")
        cache.mark_dirty("user2")
        count = await cache.flush_all()
        assert count == 2

    @pytest.mark.asyncio
    async def test_mark_dirty_nonexistent_noop(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        cache.mark_dirty("ghost")  # should not raise

    @pytest.mark.asyncio
    async def test_mark_dirty_increments_count(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        cache.mark_dirty("user1")
        cache.mark_dirty("user1")
        cache.mark_dirty("user1")
        assert cache._entries["user1"].dirty_count == 3


# ---------------------------------------------------------------------------
# Snapshot all
# ---------------------------------------------------------------------------


class TestLedgerCacheSnapshotAll:
    @pytest.mark.asyncio
    async def test_snapshot_all_iterates_entries(self) -> None:
        vault = _mock_vault()
        vault.snapshot_ledger = AsyncMock(return_value="snap-id")
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        await cache.get("user2")
        count = await cache.snapshot_all("2026-02-16T12:00:00Z")
        assert count == 2
        assert vault.snapshot_ledger.call_count == 2

    @pytest.mark.asyncio
    async def test_snapshot_all_empty_cache(self) -> None:
        vault = _mock_vault()
        vault.snapshot_ledger = AsyncMock(return_value="snap-id")
        cache = LedgerCache(vault, maxsize=5)
        count = await cache.snapshot_all("2026-02-16T12:00:00Z")
        assert count == 0
        vault.snapshot_ledger.assert_not_called()

    @pytest.mark.asyncio
    async def test_snapshot_all_skips_failures(self) -> None:
        vault = _mock_vault()
        call_count = 0

        async def snapshot_side_effect(user_id, ledger_json, ts):
            nonlocal call_count
            call_count += 1
            if user_id == "user1":
                raise Exception("vault error")
            return "snap-id"

        vault.snapshot_ledger = AsyncMock(side_effect=snapshot_side_effect)
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        await cache.get("user2")
        count = await cache.snapshot_all("2026-02-16T12:00:00Z")
        assert count == 1  # user1 failed, user2 succeeded
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_snapshot_all_counts_none_as_skipped(self) -> None:
        vault = _mock_vault()
        vault.snapshot_ledger = AsyncMock(return_value=None)
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        count = await cache.snapshot_all("2026-02-16T12:00:00Z")
        assert count == 0  # None means no ledger thought existed


# ---------------------------------------------------------------------------
# Background flush
# ---------------------------------------------------------------------------


class TestLedgerCacheBackgroundFlush:
    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_interval_secs=1)
        await cache.start_background_flush()
        assert cache._flush_task is not None
        await cache.stop()
        assert cache._flush_task is None

    @pytest.mark.asyncio
    async def test_background_flush_writes_dirty(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_interval_secs=0.1)
        ledger = await cache.get("user1")
        ledger.credit_deposit(77, "test")
        cache.mark_dirty("user1")
        await cache.start_background_flush()
        flushed = await _wait_until(lambda: vault.store_ledger.called)
        await cache.stop()
        assert flushed, "background flush did not write within timeout"
        vault.store_ledger.assert_called()

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_interval_secs=999)
        await cache.start_background_flush()
        await cache.get("user1")
        cache.mark_dirty("user1")
        await cache.stop()  # should flush before returning
        vault.store_ledger.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_interval_secs=1)
        await cache.start_background_flush()
        task1 = cache._flush_task
        await cache.start_background_flush()
        assert cache._flush_task is task1  # same task, not duplicated
        await cache.stop()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestLedgerCacheConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_get_same_user(self) -> None:
        """Two concurrent gets for the same user should both see the same ledger."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)

        results = await asyncio.gather(
            cache.get("user1"),
            cache.get("user1"),
        )
        assert results[0] is results[1]

    @pytest.mark.asyncio
    async def test_concurrent_get_different_users(self) -> None:
        """Concurrent gets for different users should not interfere."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)

        results = await asyncio.gather(
            cache.get("user1"),
            cache.get("user2"),
        )
        assert results[0] is not results[1]
        assert cache.size == 2


# ---------------------------------------------------------------------------
# Flush retry
# ---------------------------------------------------------------------------


class TestLedgerCacheFlushRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_transient_failure(self) -> None:
        """Flush succeeds on second attempt after transient vault error."""
        vault = _mock_vault()
        call_count = 0

        async def store_side_effect(user_id, ledger_json):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("transient error")
            return "ok"

        vault.store_ledger = AsyncMock(side_effect=store_side_effect)
        cache = LedgerCache(vault, maxsize=5, flush_retries=1, flush_retry_delay=0.01)
        await cache.get("user1")
        cache.mark_dirty("user1")

        result = await cache.flush_user("user1")

        assert result is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted_returns_false(self) -> None:
        """All retry attempts fail -> returns False, entry stays dirty."""
        vault = _mock_vault(fail_store=True)
        cache = LedgerCache(vault, maxsize=5, flush_retries=2, flush_retry_delay=0.01)
        await cache.get("user1")
        cache.mark_dirty("user1")

        result = await cache.flush_user("user1")

        assert result is False
        assert cache._entries["user1"].dirty is True
        # 1 initial + 2 retries = 3 total attempts
        assert vault.store_ledger.call_count == 3

    @pytest.mark.asyncio
    async def test_zero_retries_single_attempt(self) -> None:
        """flush_retries=0 means exactly one attempt, no retries."""
        vault = _mock_vault(fail_store=True)
        cache = LedgerCache(vault, maxsize=5, flush_retries=0, flush_retry_delay=0.01)
        await cache.get("user1")
        cache.mark_dirty("user1")

        result = await cache.flush_user("user1")

        assert result is False
        assert vault.store_ledger.call_count == 1

    @pytest.mark.asyncio
    async def test_default_config_has_retry(self) -> None:
        """Default LedgerCache has flush_retries=1 visible in health()."""
        vault = _mock_vault()
        cache = LedgerCache(vault)
        health = cache.health()
        assert health["flush_retries"] == 1
        assert health["flush_retry_delay"] == 2.0


# ---------------------------------------------------------------------------
# Size property
# ---------------------------------------------------------------------------


class TestLedgerCacheSize:
    @pytest.mark.asyncio
    async def test_size_empty(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_size_after_gets(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        await cache.get("user1")
        await cache.get("user2")
        assert cache.size == 2


# ---------------------------------------------------------------------------
# Flush-due (per-entry triggers)
# ---------------------------------------------------------------------------


class TestFlushDue:
    @pytest.mark.asyncio
    async def test_flush_due_after_count_threshold(self) -> None:
        """dirty_count >= N triggers flush_due."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=3)
        await cache.get("user1")
        cache.mark_dirty("user1")
        cache.mark_dirty("user1")
        assert not cache.flush_due("user1")  # 2 < 3
        cache.mark_dirty("user1")
        assert cache.flush_due("user1")  # 3 >= 3

    @pytest.mark.asyncio
    async def test_flush_not_due_below_threshold(self) -> None:
        """dirty_count < N doesn't trigger flush."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=10)
        await cache.get("user1")
        cache.mark_dirty("user1")
        assert not cache.flush_due("user1")

    @pytest.mark.asyncio
    async def test_flush_due_after_staleness(self) -> None:
        """Time > T since last flush triggers flush_due."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_staleness_secs=0.05)
        await cache.get("user1")
        cache.mark_dirty("user1")
        assert not cache.flush_due("user1")  # just created, not stale yet
        # Simulate time passing by backdating last_flush_time
        cache._entries["user1"].last_flush_time = time.monotonic() - 0.1
        assert cache.flush_due("user1")

    @pytest.mark.asyncio
    async def test_flush_due_requires_dirty(self) -> None:
        """Clean entries never trigger flush_due."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=1)
        await cache.get("user1")
        assert not cache.flush_due("user1")

    @pytest.mark.asyncio
    async def test_flush_due_resets_after_flush(self) -> None:
        """dirty_count resets to 0 after a successful flush."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=2)
        await cache.get("user1")
        cache.mark_dirty("user1")
        cache.mark_dirty("user1")
        assert cache.flush_due("user1")
        await cache.flush_user("user1")
        assert not cache.flush_due("user1")
        assert cache._entries["user1"].dirty_count == 0

    @pytest.mark.asyncio
    async def test_flush_due_nonexistent_user(self) -> None:
        """flush_due for unknown user returns False."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        assert not cache.flush_due("ghost")

    @pytest.mark.asyncio
    async def test_get_triggers_flush_at_threshold(self) -> None:
        """get() auto-flushes when the entry hits the batch threshold."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=3)
        await cache.get("user1")
        cache.mark_dirty("user1")
        cache.mark_dirty("user1")
        cache.mark_dirty("user1")  # dirty_count = 3 = batch_size
        vault.store_ledger.assert_not_called()
        # Next get() should trigger the fire-and-forget flush
        await cache.get("user1")
        await asyncio.sleep(0)  # let background task run
        vault.store_ledger.assert_called_once()
        assert cache._entries["user1"].dirty_count == 0

    @pytest.mark.asyncio
    async def test_get_triggers_flush_on_staleness(self) -> None:
        """get() auto-flushes when entry is stale."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5, flush_staleness_secs=0.05)
        await cache.get("user1")
        cache.mark_dirty("user1")
        # Backdate last_flush_time to simulate staleness
        cache._entries["user1"].last_flush_time = time.monotonic() - 0.1
        vault.store_ledger.assert_not_called()
        await cache.get("user1")
        await asyncio.sleep(0)  # let fire-and-forget flush run
        vault.store_ledger.assert_called_once()


# ---------------------------------------------------------------------------
# Debit method
# ---------------------------------------------------------------------------


class TestDebitMethod:
    @pytest.mark.asyncio
    async def test_debit_succeeds_with_balance(self) -> None:
        """debit() returns True and ledger is debited."""
        vault = _funded_vault(500)
        cache = LedgerCache(vault, maxsize=5)
        result = await cache.debit("user1", "search_thoughts", 10)
        assert result is True
        ledger = await cache.get("user1")
        assert ledger.balance_api_sats == 490
        assert cache._entries["user1"].dirty is True

    @pytest.mark.asyncio
    async def test_debit_fails_insufficient_balance(self) -> None:
        """debit() returns False when balance is insufficient, no state change."""
        vault = _mock_vault()  # empty ledger, 0 balance
        cache = LedgerCache(vault, maxsize=5)
        result = await cache.debit("user1", "search_thoughts", 10)
        assert result is False
        # Entry should not be dirty (debit didn't happen)
        assert not cache._entries["user1"].dirty

    @pytest.mark.asyncio
    async def test_debit_triggers_flush_at_threshold(self) -> None:
        """After N debits via debit(), vault.store_ledger is called on next get()."""
        vault = _funded_vault(500)
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=3)
        await cache.debit("user1", "tool_a", 1)
        await cache.debit("user1", "tool_b", 1)
        await cache.debit("user1", "tool_c", 1)  # dirty_count = 3
        vault.store_ledger.assert_not_called()
        # Next get() triggers the fire-and-forget flush
        await cache.get("user1")
        await asyncio.sleep(0)  # let background task run
        vault.store_ledger.assert_called_once()

    @pytest.mark.asyncio
    async def test_debit_no_flush_below_threshold(self) -> None:
        """Below N debits, no vault write happens."""
        vault = _funded_vault(500)
        cache = LedgerCache(vault, maxsize=5, flush_batch_size=10)
        await cache.debit("user1", "tool_a", 1)
        await cache.debit("user1", "tool_b", 1)
        # Access entry again — should NOT flush (2 < 10)
        await cache.get("user1")
        vault.store_ledger.assert_not_called()


# ---------------------------------------------------------------------------
# Write-through credit
# ---------------------------------------------------------------------------


class TestWriteThroughCredit:
    @pytest.mark.asyncio
    async def test_write_through_credit_flushes_immediately(self) -> None:
        """write_through_credit() calls vault.store_ledger right away."""
        vault = _mock_vault()
        cache = LedgerCache(vault, maxsize=5)
        ledger = await cache.get("user1")
        ledger.credit_deposit(1000, "invoice-123")
        result = await cache.write_through_credit("user1")
        assert result is True
        vault.store_ledger.assert_called_once()
        # Entry should be clean after write-through
        assert not cache._entries["user1"].dirty

    @pytest.mark.asyncio
    async def test_write_through_credit_returns_false_on_failure(self) -> None:
        """write_through_credit() returns False when vault write fails."""
        vault = _mock_vault(fail_store=True)
        cache = LedgerCache(vault, maxsize=5, flush_retries=0)
        await cache.get("user1")
        result = await cache.write_through_credit("user1")
        assert result is False
        # Entry should still be dirty
        assert cache._entries["user1"].dirty is True


# ---------------------------------------------------------------------------
# Health metrics
# ---------------------------------------------------------------------------


class TestLedgerCacheHealth:
    def test_health_includes_flush_config(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(
            vault,
            flush_batch_size=5,
            flush_staleness_secs=60.0,
        )
        health = cache.health()
        assert health["flush_batch_size"] == 5
        assert health["flush_staleness_secs"] == 60.0
        assert "last_flush_check_age_secs" not in health

    def test_health_default_values(self) -> None:
        vault = _mock_vault()
        cache = LedgerCache(vault)
        health = cache.health()
        assert health["flush_batch_size"] == 10
        assert health["flush_staleness_secs"] == 120.0
        assert health["flush_retries"] == 1
        assert health["flush_retry_delay"] == 2.0
