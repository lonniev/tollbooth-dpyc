"""Quick timing comparison: NeonVault store + fetch latency."""

import asyncio
import os
import time

from tollbooth import UserLedger, NeonVault

DATABASE_URL = os.environ.get(
    "NEON_DATABASE_URL",
    "postgresql://neondb_owner:npg_jsoldJREFh15"
    "@ep-twilight-thunder-ainublqx-pooler.c-4.us-east-1.aws.neon.tech"
    "/neondb?sslmode=require&channel_binding=require",
)

USER = "npub1_timing_test"
ROUNDS = 10


async def main() -> None:
    vault = NeonVault(database_url=DATABASE_URL)

    try:
        await vault.ensure_schema()

        ledger = UserLedger()
        ledger.credit_deposit(1000, "timing-test", ttl_seconds=3600)

        # Warm up connection pool
        await vault.store_ledger(USER, ledger.to_json())
        await vault.fetch_ledger(USER)

        # Time store_ledger
        store_times = []
        for i in range(ROUNDS):
            ledger.debit("timing_tool", 1)
            blob = ledger.to_json()
            t0 = time.perf_counter()
            await vault.store_ledger(USER, blob)
            store_times.append((time.perf_counter() - t0) * 1000)

        # Time fetch_ledger
        fetch_times = []
        for i in range(ROUNDS):
            t0 = time.perf_counter()
            await vault.fetch_ledger(USER)
            fetch_times.append((time.perf_counter() - t0) * 1000)

        print(f"NeonVault latency ({ROUNDS} rounds, ms)")
        print(f"{'':>15s}  {'min':>6s}  {'avg':>6s}  {'max':>6s}")
        print(f"{'store_ledger':>15s}  {min(store_times):6.1f}  "
              f"{sum(store_times)/len(store_times):6.1f}  {max(store_times):6.1f}")
        print(f"{'fetch_ledger':>15s}  {min(fetch_times):6.1f}  "
              f"{sum(fetch_times)/len(fetch_times):6.1f}  {max(fetch_times):6.1f}")

    finally:
        await vault.close()


if __name__ == "__main__":
    asyncio.run(main())
