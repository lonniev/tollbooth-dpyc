"""Demo: UserLedger lifecycle with NeonVault persistence.

Shows how credits flow through the system:
  1. Ensure schema (idempotent table creation)
  2. Create a fresh ledger → credit it → persist via NeonVault
  3. Fetch it back → verify round-trip fidelity
  4. Debit (tool call) → persist → fetch → verify FIFO draw
  5. LedgerCache integration (write-behind flush)
  6. Snapshot (audit journal entry)
"""

import asyncio
import json
import os
from datetime import UTC

from tollbooth import LedgerCache, NeonVault, UserLedger

DATABASE_URL = os.environ.get(
    "NEON_DATABASE_URL",
    "postgresql://neondb_owner:npg_jsoldJREFh15"
    "@ep-twilight-thunder-ainublqx-pooler.c-4.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require&channel_binding=require",
)

DEMO_USER = "npub1_demo_user_feb23"


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main() -> None:
    vault = NeonVault(database_url=DATABASE_URL)

    try:
        # ── Step 1: Schema ───────────────────────────────────────
        banner("1. Ensure schema (CREATE TABLE IF NOT EXISTS)")
        await vault.ensure_schema()
        print("  Tables: balances, transactions + 2 indexes ✓")

        # ── Step 2: Fresh ledger → credit → store ────────────────
        banner("2. Create ledger, credit 500 api_sats, persist")
        ledger = UserLedger()
        print(f"  Fresh ledger balance: {ledger.balance_api_sats}")

        ledger.credit_deposit(api_sats=500, invoice_id="demo-inv-001", ttl_seconds=86400)
        print(f"  After credit_deposit(500): balance={ledger.balance_api_sats}")
        print(f"  Tranches: {len(ledger.tranches)}")
        print(f"    [0] original={ledger.tranches[0].original_sats}, "
              f"remaining={ledger.tranches[0].remaining_sats}, "
              f"expires_at={ledger.tranches[0].expires_at}")

        ledger_json = ledger.to_json()
        version = await vault.store_ledger(DEMO_USER, ledger_json)
        print(f"  Stored to Neon → version={version}")

        # ── Step 3: Fetch back → verify ──────────────────────────
        banner("3. Fetch from Neon → deserialize → verify")
        fetched_json = await vault.fetch_ledger(DEMO_USER)
        fetched = UserLedger.from_json(fetched_json)
        print(f"  Fetched balance: {fetched.balance_api_sats}")
        print(f"  Tranches: {len(fetched.tranches)}")
        print(f"  total_deposited: {fetched.total_deposited_api_sats}")
        assert fetched.balance_api_sats == 500, "Round-trip mismatch!"
        print("  Round-trip ✓")

        # ── Step 4: Debit (simulate tool call) → persist ─────────
        banner("4. Debit 3 api_sats (tool call), then 12 more")
        ok = fetched.debit("search_thoughts", 3)
        print(f"  debit('search_thoughts', 3) → {ok}, balance={fetched.balance_api_sats}")

        ok = fetched.debit("create_thought", 12)
        print(f"  debit('create_thought', 12) → {ok}, balance={fetched.balance_api_sats}")

        version = await vault.store_ledger(DEMO_USER, fetched.to_json())
        print(f"  Stored to Neon → version={version}")

        # Fetch again to prove persistence
        refetched_json = await vault.fetch_ledger(DEMO_USER)
        refetched = UserLedger.from_json(refetched_json)
        print(f"  Re-fetched balance: {refetched.balance_api_sats}")
        print(f"  total_consumed: {refetched.total_consumed_api_sats}")
        print(f"  History: {json.dumps({t: u.to_dict() for t, u in refetched.history.items()}, indent=4)}")
        assert refetched.balance_api_sats == 485

        # ── Step 5: LedgerCache integration ──────────────────────
        banner("5. LedgerCache — in-memory hot path with write-behind")
        cache = LedgerCache(vault=vault, maxsize=10, flush_interval_secs=5)

        # get() → cache miss → loads from Neon
        cached_ledger = await cache.get(DEMO_USER)
        print(f"  cache.get() → balance={cached_ledger.balance_api_sats} (loaded from Neon)")
        print(f"  cache.size={cache.size}, dirty={cache.dirty_count}")

        # Debit via cache
        cached_ledger.debit("brain_query", 10)
        cache.mark_dirty(DEMO_USER)
        print(f"  After debit(10) + mark_dirty → balance={cached_ledger.balance_api_sats}")
        print(f"  cache.dirty={cache.dirty_count} (not yet flushed)")

        # Explicit flush
        flushed = await cache.flush_dirty()
        print(f"  flush_dirty() → wrote {flushed} ledger(s)")
        print(f"  cache.dirty={cache.dirty_count} (clean now)")

        # ── Step 6: Snapshot (audit journal) ─────────────────────
        banner("6. Snapshot — append-only audit entry in transactions table")
        from datetime import datetime
        ts = datetime.now(UTC).isoformat()
        snap_id = await vault.snapshot_ledger(DEMO_USER, cached_ledger.to_json(), ts)
        print(f"  Snapshot recorded → transaction id={snap_id}")

        # ── Summary ──────────────────────────────────────────────
        banner("Summary")
        final_json = await vault.fetch_ledger(DEMO_USER)
        final = UserLedger.from_json(final_json)
        print(f"  Final balance: {final.balance_api_sats} api_sats")
        print(f"  Deposited:     {final.total_deposited_api_sats}")
        print(f"  Consumed:      {final.total_consumed_api_sats}")
        print(f"  Tranches:      {len(final.tranches)}")
        print("  History:")
        for tool, u in final.history.items():
            print(f"    {tool}: {u.calls} calls, {u.api_sats} api_sats")

        await cache.stop()
        print("\n  Done ✓")

    finally:
        await vault.close()


if __name__ == "__main__":
    asyncio.run(main())
