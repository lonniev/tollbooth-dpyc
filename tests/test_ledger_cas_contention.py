"""What happens to money when two replicas write the same ledger at once.

The definitive store refuses to blind-overwrite: ``store_ledger`` compares the
version the writer last read and raises ``LedgerVersionConflict`` if the row has
moved on. That is correct, and it is only half a protocol — the other half is the
caller re-fetching and re-applying, which ``mutate()`` does and the old
``mark_dirty`` + ``flush_user`` path did not. That path retried the *same stale
snapshot*, which can never satisfy the CAS guard, then gave up and left the entry
dirty forever, unflushable for the life of the process.

Observed live 2026-08-01: two Uvicorn workers serving one eXcalibur deployment
produced a continuous stream of ``Failed to flush ledger to vault``, three
conflicts for one npub inside 65 ms. It read as a Neon outage. Neon was healthy —
a snippet write and read-back through the same database succeeded during the same
window. It was contention, and the swallowed exception is what disguised it.

So these tests assert two things:

1. **A money mutation survives losing a CAS race** — it is re-applied against the
   winner's state, and both writers' effects are present afterwards.
2. **A re-applied mutation does not double-credit.** ``credit_deposit`` is NOT
   idempotent; it appends a tranche every call. Every restore path therefore
   guards on the tranche already existing, and that guard is load-bearing
   precisely because the retry re-runs the function.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.vault_backend import LedgerVersionConflict


class _FakeVault:
    """A vault with a real version counter and honest CAS semantics.

    ``conflicts_before_success`` makes the next N writes lose the race, the way a
    second replica writing between our read and our write would.
    """

    def __init__(self, *, conflicts_before_success: int = 0):
        self.stored: str | None = None
        self.version = 0
        self.conflicts_left = conflicts_before_success
        self.writes = 0
        self.reads = 0
        # What a competing replica has already committed; served on re-fetch.
        self.competitor_json: str | None = None

    async def fetch_ledger(self, user_id: str) -> str | None:
        self.reads += 1
        return self.stored

    async def store_ledger(self, user_id: str, ledger_json: str) -> str:
        self.writes += 1
        if self.conflicts_left > 0:
            self.conflicts_left -= 1
            # Losing the race means someone else's write landed BETWEEN our read
            # and ours — so their state becomes the stored state, and that is
            # what our re-fetch must see. Publishing it here rather than up front
            # is what makes this a race and not merely a rejected write.
            if self.competitor_json is not None:
                self.stored = self.competitor_json
                self.version += 1
            raise LedgerVersionConflict(f"version conflict for {user_id}")
        self.stored = ledger_json
        self.version += 1
        return str(self.version)


def _cache(vault: _FakeVault) -> LedgerCache:
    return LedgerCache(vault, maxsize=20, flush_interval_secs=600)


# ---------------------------------------------------------------------------
# 1. A money mutation survives a lost race
# ---------------------------------------------------------------------------


class TestMutateSurvivesContention:
    @pytest.mark.asyncio
    async def test_a_credit_is_reapplied_onto_the_winners_state(self):
        """The classic lost-update: we credit while another replica credits.

        Neither may be dropped. Ours is re-applied against THEIR committed state,
        so the final ledger carries both tranches.
        """
        vault = _FakeVault(conflicts_before_success=1)
        # The competitor got there first with 500 sats.
        rival = UserLedger()
        rival.credit_deposit(500, "rival-invoice")
        vault.competitor_json = rival.to_json()

        cache = _cache(vault)

        def _credit_ours(led: UserLedger) -> int:
            led.credit_deposit(300, "our-invoice")
            return 300

        granted = await cache.mutate("user-1", _credit_ours)

        assert granted == 300
        assert vault.writes == 2, "the first write lost the race and was retried"
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 800, "both credits survived"
        assert "rival-invoice" in final.credited_invoices
        assert "our-invoice" in final.credited_invoices

    @pytest.mark.asyncio
    async def test_an_idempotency_guard_still_sees_fresh_state_after_a_conflict(self):
        """The guard must run against the WINNER's ledger, not our stale copy.

        This is why settlement checks `credited_invoices` inside the mutation: if
        the rival already credited this very invoice, the re-applied attempt has
        to notice and decline, or the patron is credited twice for one payment.
        """
        vault = _FakeVault(conflicts_before_success=1)
        rival = UserLedger()
        rival.credit_deposit(500, "invoice-42")  # rival settled the SAME invoice
        vault.competitor_json = rival.to_json()

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
        """The real ``restore_credits_tool`` against a real lost race.

        ``restore_credits_tool`` guards on ``credited_invoices`` up front, but
        that read happens BEFORE the write. A rival crediting the same invoice in
        between slips past it — and because ``mutate()`` re-applies the function
        on conflict, an unguarded restore would then mint a second tranche for
        one payment. The in-mutation tranche check is what closes that window.
        """
        from tollbooth.tools.credits import restore_credits_tool

        rival = UserLedger()
        rival.credit_deposit(1000, "inv-1")  # the rival restored it first

        vault = _FakeVault(conflicts_before_success=1)
        vault.competitor_json = rival.to_json()
        cache = _cache(vault)

        btcpay = AsyncMock()
        btcpay.get_invoice = AsyncMock(return_value={
            "id": "inv-1", "status": "Settled", "amount": "1000",
        })

        result = await restore_credits_tool(btcpay, cache, "user-1", "inv-1")

        assert result["success"] is True
        assert result["credits_granted"] == 0, "the rival's credit must be noticed"
        final = UserLedger.from_json(vault.stored)
        assert final.balance_api_sats == 1000, "one payment, one credit"
        assert len([t for t in final.tranches if t.invoice_id == "inv-1"]) == 1


# ---------------------------------------------------------------------------
# 2. A lost race no longer strands a cache entry forever
# ---------------------------------------------------------------------------


class TestFlushNoLongerStrands:
    @pytest.mark.asyncio
    async def test_a_conflicted_flush_adopts_fresh_state_instead_of_looping(self):
        """The old path retried the same stale snapshot — which the CAS guard can
        never accept — and left the entry dirty for the life of the process.

        Now the conflict is recognized: the newer stored state is adopted and the
        entry stops being a permanently-unflushable zombie.
        """
        vault = _FakeVault(conflicts_before_success=99)  # every write loses
        winner = UserLedger()
        winner.credit_deposit(700, "winner-invoice")
        vault.competitor_json = winner.to_json()

        cache = _cache(vault)
        await cache.get("user-1")
        cache.mark_dirty("user-1")
        assert cache.dirty_count == 1

        flushed = await cache.flush_user("user-1")

        assert flushed is False, "honest: this replica's counters did not persist"
        assert cache.dirty_count == 0, "entry is no longer a stuck zombie"
        # And the adopted state is the winner's, not an empty overwrite.
        adopted = await cache.get("user-1")
        assert adopted.balance_api_sats == 700

    @pytest.mark.asyncio
    async def test_a_conflicted_flush_does_not_burn_its_retries(self):
        """Retrying a CAS conflict is provably useless — the version we hold is
        one we will never hold again. It must not cost a retry budget or a sleep."""
        vault = _FakeVault(conflicts_before_success=99)
        vault.competitor_json = UserLedger().to_json()

        cache = _cache(vault)
        await cache.get("user-1")
        cache.mark_dirty("user-1")

        await cache.flush_user("user-1")

        assert vault.writes == 1, "one attempt, then adopt — no pointless retries"

    @pytest.mark.asyncio
    async def test_a_non_conflict_failure_still_retries(self):
        """Only CAS conflicts short-circuit. A genuine transient error — a dropped
        connection — is still worth a second attempt."""
        vault = _FakeVault()
        attempts = {"n": 0}

        async def _flaky(user_id, ledger_json):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("connection reset")
            vault.stored = ledger_json
            return "1"

        vault.store_ledger = _flaky  # type: ignore[assignment]
        cache = LedgerCache(vault, maxsize=20, flush_interval_secs=600, flush_retry_delay=0)
        await cache.get("user-1")
        cache.mark_dirty("user-1")

        flushed = await cache.flush_user("user-1")

        assert flushed is True
        assert attempts["n"] == 2, "a transient failure earns its retry"
