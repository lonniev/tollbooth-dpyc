"""Mock a realistic ledger, persist to Neon, fetch back, show both sides."""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta

from tollbooth import UserLedger, InvoiceRecord, Tranche, NeonVault

DATABASE_URL = os.environ.get(
    "NEON_DATABASE_URL",
    "postgresql://neondb_owner:npg_jsoldJREFh15"
    "@ep-twilight-thunder-ainublqx-pooler.c-4.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require&channel_binding=require",
)

DEMO_USER = "npub1_mock_realistic_feb23"


def build_mock_ledger() -> UserLedger:
    """Build a ledger that looks like a real user with a few days of history."""
    now = datetime.now(timezone.utc)

    ledger = UserLedger()

    # Two paid deposits — one older (partially consumed), one fresh
    ledger.tranches = [
        Tranche(
            granted_at=(now - timedelta(days=5)).isoformat(),
            original_sats=1000,
            remaining_sats=220,
            invoice_id="inv_A1b2C3d4",
            expires_at=(now + timedelta(days=25)).isoformat(),
        ),
        Tranche(
            granted_at=(now - timedelta(hours=6)).isoformat(),
            original_sats=5000,
            remaining_sats=5000,
            invoice_id="inv_X9y8Z7w6",
            expires_at=(now + timedelta(days=30)).isoformat(),
        ),
    ]

    # Seed deposit as a never-expiring tranche
    ledger.tranches.insert(0, Tranche(
        granted_at=(now - timedelta(days=14)).isoformat(),
        original_sats=200,
        remaining_sats=0,  # fully consumed
        invoice_id="seed:registration",
        expires_at=None,
    ))

    ledger.total_deposited_api_sats = 6200
    ledger.total_consumed_api_sats = 980
    ledger.total_expired_api_sats = 0
    ledger.last_deposit_at = (now - timedelta(hours=6)).strftime("%Y-%m-%d")

    # Invoice records
    ledger.invoices = {
        "inv_A1b2C3d4": InvoiceRecord(
            invoice_id="inv_A1b2C3d4",
            amount_sats=1000,
            api_sats_credited=1000,
            multiplier=1,
            status="Settled",
            created_at=(now - timedelta(days=5)).isoformat(),
            settled_at=(now - timedelta(days=5, hours=-1)).isoformat(),
            btcpay_status="Settled",
        ),
        "inv_X9y8Z7w6": InvoiceRecord(
            invoice_id="inv_X9y8Z7w6",
            amount_sats=5000,
            api_sats_credited=5000,
            multiplier=1,
            status="Settled",
            created_at=(now - timedelta(hours=6)).isoformat(),
            settled_at=(now - timedelta(hours=5, minutes=58)).isoformat(),
            btcpay_status="Settled",
        ),
    }
    ledger.credited_invoices = ["inv_A1b2C3d4", "inv_X9y8Z7w6"]

    # Tool usage history
    from tollbooth import ToolUsage
    ledger.history = {
        "brain_query":     ToolUsage(calls=42, api_sats=420),
        "search_thoughts": ToolUsage(calls=85, api_sats=85),
        "create_thought":  ToolUsage(calls=15, api_sats=75),
        "get_thought":     ToolUsage(calls=120, api_sats=120),
        "get_note":        ToolUsage(calls=60, api_sats=60),
        "create_link":     ToolUsage(calls=22, api_sats=110),
        "update_thought":  ToolUsage(calls=8, api_sats=40),
        "get_thought_graph": ToolUsage(calls=35, api_sats=35),
        "check_balance":   ToolUsage(calls=12, api_sats=0),
        "session_status":  ToolUsage(calls=5, api_sats=0),
    }

    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    ledger.daily_log = {
        yesterday: {
            "brain_query":     ToolUsage(calls=18, api_sats=180),
            "get_thought":     ToolUsage(calls=40, api_sats=40),
            "search_thoughts": ToolUsage(calls=30, api_sats=30),
        },
        today: {
            "brain_query":     ToolUsage(calls=7, api_sats=70),
            "get_thought":     ToolUsage(calls=12, api_sats=12),
            "create_thought":  ToolUsage(calls=3, api_sats=15),
        },
    }

    return ledger


async def main() -> None:
    vault = NeonVault(database_url=DATABASE_URL)

    try:
        await vault.ensure_schema()

        # Build and persist
        ledger = build_mock_ledger()
        ledger_json = ledger.to_json()

        print("STORED JSON")
        print("-" * 60)
        print(ledger_json)

        await vault.store_ledger(DEMO_USER, ledger_json)

        # Fetch back
        fetched_json = await vault.fetch_ledger(DEMO_USER)
        fetched = UserLedger.from_json(fetched_json)

        print("\n\nFETCHED JSON")
        print("-" * 60)
        print(fetched_json)

        print("\n\nDESERIALIZED LEDGER")
        print("-" * 60)
        print(f"  balance_api_sats:         {fetched.balance_api_sats}")
        print(f"  total_deposited_api_sats: {fetched.total_deposited_api_sats}")
        print(f"  total_consumed_api_sats:  {fetched.total_consumed_api_sats}")
        print(f"  total_expired_api_sats:   {fetched.total_expired_api_sats}")
        print(f"  tranches:                 {len(fetched.tranches)}")
        for i, t in enumerate(fetched.tranches):
            print(f"    [{i}] original={t.original_sats:>5}, "
                  f"remaining={t.remaining_sats:>5}, "
                  f"expires={'never' if t.expires_at is None else t.expires_at[:19]}")
        print(f"  invoices:                 {len(fetched.invoices)}")
        for iid, rec in fetched.invoices.items():
            print(f"    {iid}: {rec.status}, {rec.api_sats_credited} api_sats")
        print(f"  history ({len(fetched.history)} tools):")
        for tool, u in sorted(fetched.history.items(), key=lambda x: -x[1].api_sats):
            print(f"    {tool:25s} {u.calls:>4} calls  {u.api_sats:>5} api_sats")
        print(f"  daily_log days:           {sorted(fetched.daily_log.keys())}")

        # Fidelity check
        print("\n\nFIDELITY CHECK")
        print("-" * 60)
        stored = json.loads(ledger_json)
        fetched_parsed = json.loads(fetched_json)
        match = stored == fetched_parsed
        print(f"  JSON round-trip identical: {match}")
        if not match:
            # Show diffs
            for key in set(stored) | set(fetched_parsed):
                if stored.get(key) != fetched_parsed.get(key):
                    print(f"  DIFF in '{key}':")
                    print(f"    stored:  {json.dumps(stored.get(key))[:120]}")
                    print(f"    fetched: {json.dumps(fetched_parsed.get(key))[:120]}")

    finally:
        await vault.close()


if __name__ == "__main__":
    asyncio.run(main())
