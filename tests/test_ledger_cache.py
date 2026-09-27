"""LedgerCache: one write path, versions bound to snapshots, reads that never write.

The store is faked honestly: it hands out a version with every read and
accepts a write only at that exact version. That is the property the old
per-user version cache broke — and the property every test here leans on.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.vault_backend import (
    LedgerUnavailableError,
    LedgerVersionConflict,
    LedgerWriteError,
)

# ---------------------------------------------------------------------------
# An honest store
# ---------------------------------------------------------------------------


class HonestVault:
    """One row per user; a write lands only at the version the writer read."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, int]] = {}
        self.reads = 0
        self.writes = 0
        self.refused = 0
        self.log: list[tuple[str, int | None]] = []  # (user, expected_version) per write

    async def fetch_ledger(self, user_id: str) -> tuple[str, int] | None:
        self.reads += 1
        return self.rows.get(user_id)

    async def store_ledger(self, user_id: str, ledger_json: str, expected_version: int | None) -> int:
        self.writes += 1
        self.log.append((user_id, expected_version))
        current = self.rows.get(user_id)
        if expected_version is None:
            if current is not None:
                self.refused += 1
                raise LedgerVersionConflict("row appeared")
            self.rows[user_id] = (ledger_json, 1)
            return 1
        if current is None or current[1] != expected_version:
            self.refused += 1
            raise LedgerVersionConflict(f"had v{expected_version}")
        self.rows[user_id] = (ledger_json, expected_version + 1)
        return expected_version + 1

    async def snapshot_ledger(self, user_id: str, ledger_json: str, timestamp: str) -> str | None:
        return "snap"

    # test helpers
    def seed(self, user_id: str, ledger: UserLedger, version: int = 1) -> None:
        self.rows[user_id] = (ledger.to_json(), version)

    def ledger(self, user_id: str) -> UserLedger:
        return UserLedger.from_json(self.rows[user_id][0])

    def version(self, user_id: str) -> int:
        return self.rows[user_id][1]


def _funded(balance: int = 500) -> HonestVault:
    v = HonestVault()
    led = UserLedger()
    led.credit_deposit(balance, "seed")
    v.seed("u1", led)
    return v


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


class TestGet:
    @pytest.mark.asyncio
    async def test_miss_on_a_new_patron_is_an_empty_ledger(self) -> None:
        cache = LedgerCache(HonestVault())
        led = await cache.get("u1")
        assert led.balance_api_sats == 0
        assert not getattr(led, "_vault_unavailable", False)

    @pytest.mark.asyncio
    async def test_miss_loads_the_stored_ledger(self) -> None:
        cache = LedgerCache(_funded(500))
        assert (await cache.get("u1")).balance_api_sats == 500

    @pytest.mark.asyncio
    async def test_hit_returns_the_same_snapshot_without_a_read(self) -> None:
        vault = _funded()
        cache = LedgerCache(vault)
        a = await cache.get("u1")
        b = await cache.get("u1")
        assert a is b
        assert vault.reads == 1

    @pytest.mark.asyncio
    async def test_an_unreadable_store_yields_a_flagged_uncached_empty_ledger(self) -> None:
        vault = AsyncMock()
        vault.fetch_ledger = AsyncMock(side_effect=Exception("cold"))
        cache = LedgerCache(vault)
        led = await cache.get("u1")
        assert led.balance_api_sats == 0
        assert getattr(led, "_vault_unavailable", False) is True
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_reads_never_write(self) -> None:
        vault = _funded()
        cache = LedgerCache(vault)
        for _ in range(5):
            await cache.get("u1")
        await cache.get_fresh("u1")
        assert vault.writes == 0

    @pytest.mark.asyncio
    async def test_get_fresh_reloads_and_replaces_the_snapshot(self) -> None:
        vault = _funded(500)
        cache = LedgerCache(vault)
        stale = await cache.get("u1")
        richer = UserLedger()
        richer.credit_deposit(900, "elsewhere")
        vault.seed("u1", richer, version=2)
        fresh = await cache.get_fresh("u1")
        assert fresh.balance_api_sats == 900
        assert fresh is not stale
        assert (await cache.get("u1")) is fresh


class TestEviction:
    @pytest.mark.asyncio
    async def test_capacity_evicts_the_least_recently_used(self) -> None:
        cache = LedgerCache(HonestVault(), maxsize=2)
        await cache.get("a")
        await cache.get("b")
        await cache.get("a")  # refresh a
        await cache.get("c")  # evicts b
        assert set(cache._entries) == {"a", "c"}
        assert cache.size == 2

    @pytest.mark.asyncio
    async def test_eviction_writes_nothing(self) -> None:
        vault = HonestVault()
        cache = LedgerCache(vault, maxsize=1)
        await cache.get("a")
        await cache.get("b")
        assert vault.writes == 0


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_gets_of_one_patron_read_the_store_once(self) -> None:
        vault = _funded()
        cache = LedgerCache(vault)
        results = await asyncio.gather(*(cache.get("u1") for _ in range(10)))
        assert all(r is results[0] for r in results)
        assert vault.reads == 1


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------


class TestMutate:
    @pytest.mark.asyncio
    async def test_writes_at_the_version_it_read(self) -> None:
        vault = _funded()
        vault.rows["u1"] = (vault.rows["u1"][0], 7)
        cache = LedgerCache(vault)
        await cache.credit("u1", 100, "inv")
        assert vault.log == [("u1", 7)]
        assert vault.version("u1") == 8

    @pytest.mark.asyncio
    async def test_a_new_patron_is_inserted_not_updated(self) -> None:
        vault = HonestVault()
        cache = LedgerCache(vault)
        await cache.credit("u1", 100, "inv")
        assert vault.log == [("u1", None)]
        assert vault.version("u1") == 1

    @pytest.mark.asyncio
    async def test_the_snapshot_served_afterwards_is_what_was_written(self) -> None:
        vault = _funded(500)
        cache = LedgerCache(vault)
        await cache.credit("u1", 100, "inv")
        assert (await cache.get("u1")).balance_api_sats == 600
        assert vault.reads == 1, "the post-write snapshot is served without another read"

    @pytest.mark.asyncio
    async def test_fn_returning_false_writes_nothing_and_installs_the_fresh_read(self) -> None:
        vault = _funded(5)
        cache = LedgerCache(vault)
        old = await cache.get("u1")
        richer = UserLedger()
        richer.credit_deposit(5, "seed")
        richer.credit_deposit(50, "later")
        vault.seed("u1", richer, version=2)
        ok = await cache.debit("u1", "tool", 100)  # still short
        assert ok is False
        assert vault.writes == 0
        served = await cache.get("u1")
        assert served is not old and served.balance_api_sats == 55

    @pytest.mark.asyncio
    async def test_an_unreadable_store_raises_and_never_writes(self) -> None:
        vault = HonestVault()
        vault.fetch_ledger = AsyncMock(side_effect=Exception("cold"))  # type: ignore[method-assign]
        cache = LedgerCache(vault)
        with pytest.raises(LedgerUnavailableError):
            await cache.credit("u1", 100, "inv")
        assert vault.writes == 0

    @pytest.mark.asyncio
    async def test_a_lost_race_is_replayed_on_the_winners_state(self) -> None:
        vault = _funded(500)
        cache = LedgerCache(vault)
        real_store = vault.store_ledger
        raced = {"done": False}

        async def store_with_a_rival(user_id, ledger_json, expected_version):
            if not raced["done"]:
                raced["done"] = True
                rival = vault.ledger(user_id)
                rival.credit_deposit(250, "rival")
                vault.rows[user_id] = (rival.to_json(), vault.version(user_id) + 1)
            return await real_store(user_id, ledger_json, expected_version)

        vault.store_ledger = store_with_a_rival  # type: ignore[method-assign]
        await cache.credit("u1", 100, "ours")
        final = vault.ledger("u1")
        assert final.balance_api_sats == 850, "both writers' credits survive"
        assert vault.refused == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_raise(self) -> None:
        vault = _funded()
        vault.store_ledger = AsyncMock(side_effect=LedgerVersionConflict("always"))  # type: ignore[method-assign]
        cache = LedgerCache(vault)
        with pytest.raises(LedgerWriteError):
            await cache.credit("u1", 1, "inv", )

    @pytest.mark.asyncio
    async def test_debit_succeeds_and_is_written_through(self) -> None:
        vault = _funded(100)
        cache = LedgerCache(vault)
        assert await cache.debit("u1", "tool", 30) is True
        assert vault.ledger("u1").balance_api_sats == 70
        assert vault.ledger("u1").history["tool"].calls == 1


# ---------------------------------------------------------------------------
# Usage accounting as deltas
# ---------------------------------------------------------------------------


class TestUsage:
    @pytest.mark.asyncio
    async def test_a_free_call_writes_nothing_by_itself(self) -> None:
        vault = _funded()
        cache = LedgerCache(vault)
        for _ in range(20):
            cache.note_usage("u1", "match_state")
        assert vault.writes == 0
        assert cache.pending_usage == 1

    @pytest.mark.asyncio
    async def test_the_next_mutate_carries_the_counters_along(self) -> None:
        vault = _funded(500)
        cache = LedgerCache(vault)
        cache.note_usage("u1", "match_state")
        cache.note_usage("u1", "match_state")
        cache.note_usage("u1", "my_bee")
        await cache.credit("u1", 100, "inv")
        stored = vault.ledger("u1")
        assert stored.history["match_state"].calls == 2
        assert stored.history["my_bee"].calls == 1
        assert stored.balance_api_sats == 600
        assert vault.writes == 1
        assert cache.pending_usage == 0

    @pytest.mark.asyncio
    async def test_fold_writes_pending_counters_onto_fresh_state(self) -> None:
        vault = _funded(500)
        cache = LedgerCache(vault)
        cache.note_usage("u1", "match_state")
        folded = await cache.fold_usage()
        assert folded == 1
        stored = vault.ledger("u1")
        assert stored.history["match_state"].calls == 1
        assert stored.balance_api_sats == 500
        assert cache.pending_usage == 0

    @pytest.mark.asyncio
    async def test_a_fold_that_cannot_read_keeps_its_counters(self) -> None:
        vault = HonestVault()
        vault.fetch_ledger = AsyncMock(side_effect=Exception("cold"))  # type: ignore[method-assign]
        cache = LedgerCache(vault)
        cache.note_usage("u1", "tool")
        assert await cache.fold_usage() == 0
        assert cache.pending_usage == 1
        assert cache._usage["u1"] == {"tool": 1}

    @pytest.mark.asyncio
    async def test_a_refused_debit_keeps_the_counters_for_a_real_write(self) -> None:
        vault = _funded(5)
        cache = LedgerCache(vault)
        cache.note_usage("u1", "tool")
        assert await cache.debit("u1", "tool", 100) is False
        assert vault.writes == 0
        assert cache._usage["u1"] == {"tool": 1}

    @pytest.mark.asyncio
    async def test_counters_never_touch_tranches(self) -> None:
        """The persistence-boundary assertion: a fold changes counters only."""
        vault = _funded(500)
        before = vault.ledger("u1").to_json()
        cache = LedgerCache(vault)
        cache.note_usage("u1", "tool")
        await cache.fold_usage()
        after = vault.ledger("u1")
        import json
        b, a = json.loads(before), json.loads(after.to_json())
        for key in ("tranches", "total_deposited_api_sats", "total_consumed_api_sats",
                    "total_expired_api_sats", "pending_invoices", "credited_invoices", "invoices"):
            assert a[key] == b[key], key

    @pytest.mark.asyncio
    async def test_the_fold_loop_starts_stops_and_writes_what_is_pending(self) -> None:
        vault = _funded()
        cache = LedgerCache(vault, fold_interval_secs=3600)
        await cache.start_usage_fold()
        await cache.start_usage_fold()  # idempotent
        assert cache.health()["usage_fold_running"] is True
        cache.note_usage("u1", "tool")
        await cache.stop()
        assert cache.health()["usage_fold_running"] is False
        assert vault.ledger("u1").history["tool"].calls == 1


class TestHealth:
    def test_health_names_the_fold_not_a_flush(self) -> None:
        h = LedgerCache(HonestVault()).health()
        assert set(h) == {"cache_size", "pending_usage", "usage_folds", "fold_interval_secs", "usage_fold_running"}
        assert h["fold_interval_secs"] == 60
