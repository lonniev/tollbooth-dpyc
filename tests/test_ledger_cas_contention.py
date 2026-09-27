"""What happens to money when two writers hit the same ledger at once.

The definitive store refuses to blind-overwrite: ``store_ledger`` lands only
at the version the writer read its snapshot at, and raises
``LedgerVersionConflict`` otherwise. The other half of the protocol is the
caller re-reading and re-applying, which ``mutate()`` does.

Two incidents shaped these tests. 2026-08-01, eXcalibur: two Uvicorn workers
produced a stream of ``Failed to flush ledger to vault`` that read as a Neon
outage and was contention. 2026-09-27, Bee's Knees: a settled 1,000-sat
top-up was credited, acknowledged by DM, and erased within the hour — a
background flush holding a stale snapshot wrote it under the version a fresher
writer had just produced, because the version was read from a per-user cache
rather than carried with the snapshot. There is no such flush any more; the
last test here is that interleaving, and it now cannot lose.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.vault_backend import LedgerVersionConflict


class _HonestVault:
    """One row, one version; a write lands only at the version that was read.

    ``rival`` is a ledger another writer commits BETWEEN our read and our
    write — published at the moment our first write arrives, which is what
    makes it a race and not merely a rejected write.
    """

    def __init__(self, *, rival: UserLedger | None = None):
        self.stored: str | None = None
        self.version = 0
        self.rival = rival
        self.writes = 0
        self.reads = 0
        self.expected_seen: list[int | None] = []

    async def fetch_ledger(self, user_id: str) -> tuple[str, int] | None:
        self.reads += 1
        return None if self.stored is None else (self.stored, self.version)

    async def store_ledger(self, user_id: str, ledger_json: str, expected_version: int | None) -> int:
        self.writes += 1
        self.expected_seen.append(expected_version)
        if self.rival is not None:
            self.stored = self.rival.to_json()
            self.version += 1
            self.rival = None
        have = None if self.stored is None else self.version
        if expected_version != have:
            raise LedgerVersionConflict(f"expected v{expected_version}, row is v{have}")
        self.stored = ledger_json
        self.version += 1
        return self.version

    async def snapshot_ledger(self, user_id: str, ledger_json: str, timestamp: str) -> str | None:
        return None


def _cache(vault: _HonestVault) -> LedgerCache:
    return LedgerCache(vault, maxsize=20, fold_interval_secs=600)


class TestMutateSurvivesContention:
    @pytest.mark.asyncio
    async def test_a_credit_is_reapplied_onto_the_winners_state(self):
        rival = UserLedger()
        rival.credit_deposit(500, "rival-invoice")
        vault = _HonestVault(rival=rival)
        cache = _cache(vault)

        def _credit_ours(led: UserLedger) -> int:
            led.credit_deposit(300, "our-invoice")
            return 300

        granted = await cache.mutate("user-1", _credit_ours)

        assert granted == 300
        assert vault.writes == 2, "the first write lost the race and was retried"
        assert vault.expected_seen == [None, 1], "the retry wrote at the version it re-read"
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 800, "both credits survived"
        assert {"rival-invoice", "our-invoice"} <= set(final.credited_invoices)

    @pytest.mark.asyncio
    async def test_an_idempotency_guard_still_sees_fresh_state_after_a_conflict(self):
        rival = UserLedger()
        rival.credit_deposit(500, "invoice-42")  # rival settled the SAME invoice
        vault = _HonestVault(rival=rival)
        cache = _cache(vault)

        def _settle(led: UserLedger) -> int:
            if "invoice-42" in led.credited_invoices:
                return 0
            led.credit_deposit(500, "invoice-42")
            return 500

        granted = await cache.mutate("user-1", _settle)

        assert granted == 0, "re-applied settlement must notice the rival's credit"
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 500, "credited once, not twice"


class TestRestoreIsRetrySafe:
    @pytest.mark.asyncio
    async def test_a_rival_crediting_mid_restore_does_not_double_credit(self):
        from tollbooth.tools.credits import restore_credits_tool

        rival = UserLedger()
        rival.credit_deposit(1000, "inv-1")
        vault = _HonestVault(rival=rival)
        cache = _cache(vault)

        btcpay = AsyncMock()
        btcpay.get_invoice = AsyncMock(return_value={"id": "inv-1", "status": "Settled", "amount": "1000"})

        result = await restore_credits_tool(btcpay, cache, "user-1", "inv-1")

        assert result["success"] is True
        assert result["credits_granted"] == 0, "the rival's credit must be noticed"
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 1000, "one payment, one credit"
        assert len([t for t in final.tranches if t.invoice_id == "inv-1"]) == 1


class TestNothingWritesAStaleSnapshot:
    @pytest.mark.asyncio
    async def test_the_2026_09_27_interleaving_cannot_erase_a_credit(self):
        """Free calls arm usage counters; a settlement lands; then the counters
        are written. Under the old flush the counters' whole-ledger snapshot,
        taken before the settlement, went out under the settlement's version
        and erased it. Now the counters are deltas folded onto the fresh row."""
        vault = _HonestVault()
        cache = _cache(vault)
        # A patron with a row: the page polls, counters accumulate.
        await cache.credit("patron", 118, "seed")
        for _ in range(8):
            cache.note_usage("patron", "match_state")
        # check_payment settles a 1,000-sat invoice through mutate.
        def _settle(led: UserLedger) -> int:
            led.credit_deposit(1000, "5oKNGu3DKs5NHPFWsGBPPR")
            return 1000
        assert await cache.mutate("patron", _settle) == 1000
        # Whatever writes next carries counters, not a snapshot.
        assert await cache.fold_usage() == 0, "the settlement already carried them"
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 1118
        assert "5oKNGu3DKs5NHPFWsGBPPR" in final.credited_invoices
        assert final.history["match_state"].calls == 8

    @pytest.mark.asyncio
    async def test_counters_noted_after_a_settlement_fold_on_top_of_it(self):
        vault = _HonestVault()
        cache = _cache(vault)
        await cache.credit("patron", 1000, "inv")
        cache.note_usage("patron", "match_state")
        await cache.fold_usage()
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 1000
        assert final.history["match_state"].calls == 1
        assert vault.expected_seen == [None, 1], "every write carried the version it read"

    @pytest.mark.asyncio
    async def test_every_write_carries_the_version_of_its_own_read(self):
        """The invariant itself: the store sees, for each write, exactly the
        version handed out by the read that produced the snapshot."""
        vault = _HonestVault()
        cache = _cache(vault)
        for i in range(5):
            await cache.credit("patron", 1, f"inv-{i}")
            cache.note_usage("patron", "poll")
            await cache.fold_usage()
        assert vault.expected_seen == [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert vault.version == 10

    @pytest.mark.asyncio
    async def test_the_cache_has_no_way_to_write_a_snapshot(self):
        for gone in ("mark_dirty", "flush_user", "flush_dirty", "flush_all", "snapshot_all", "write_through_credit"):
            assert not hasattr(LedgerCache, gone), f"{gone} must not exist"
