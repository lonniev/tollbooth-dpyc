"""Demo: Nostr audit events via AuditedVault decorator.

Shows how AuditedVault wraps a mock vault backend and publishes kind 30078
audit events to configured Nostr relays on every write.

Setup:
    pip install -e ".[nostr]"
    export TOLLBOOTH_NOSTR_OPERATOR_NSEC=nsec1...
    export TOLLBOOTH_NOSTR_RELAYS=wss://relay.damus.io,wss://nos.lol

Run:
    python demos/demo_nostr_audit.py

After running, verify events on a relay:
    {"kinds": [30078], "#t": ["tollbooth-audit"]}
"""

import asyncio
import json
import os
import sys
from typing import Any
import time

from tollbooth import UserLedger
from tollbooth.nostr_audit import AuditedVault, NostrAuditPublisher


DEMO_USER = "npub1_demo_audit_feb23"


class MockVault:
    """In-memory vault for demo purposes."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._version: int = 0

    async def store_ledger(self, user_id: str, ledger_json: str) -> str:
        self._store[user_id] = ledger_json
        self._version += 1
        return str(self._version)

    async def fetch_ledger(self, user_id: str) -> str | None:
        return self._store.get(user_id)

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str,
    ) -> str | None:
        await self.store_ledger(user_id, ledger_json)
        return f"snap-{self._version}"


def banner(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def main() -> None:
    nsec = os.environ.get("TOLLBOOTH_NOSTR_OPERATOR_NSEC", "").strip()
    relays_str = os.environ.get("TOLLBOOTH_NOSTR_RELAYS", "").strip()

    if not nsec or not relays_str:
        print(
            "Error: Set TOLLBOOTH_NOSTR_OPERATOR_NSEC and "
            "TOLLBOOTH_NOSTR_RELAYS environment variables.",
            file=sys.stderr,
        )
        print(
            "\nExample:\n"
            "  export TOLLBOOTH_NOSTR_OPERATOR_NSEC=nsec1...\n"
            "  export TOLLBOOTH_NOSTR_RELAYS=wss://relay.damus.io,wss://nos.lol",
            file=sys.stderr,
        )
        sys.exit(1)

    relays = [r.strip() for r in relays_str.split(",") if r.strip()]

    # ── Step 1: Create publisher + AuditedVault ──────────────────
    banner("1. Create NostrAuditPublisher + AuditedVault")
    publisher = NostrAuditPublisher(
        operator_nsec=nsec,
        relays=relays,
        enabled=True,
    )
    if not publisher.enabled:
        print("  Publisher disabled — check nsec and pynostr installation.")
        sys.exit(1)

    inner_vault = MockVault()
    vault = AuditedVault(inner_vault, publisher)
    print(f"  Publisher enabled: {publisher.enabled}")
    print(f"  Relays: {relays}")
    print(f"  Inner vault: MockVault (in-memory)")

    # ── Step 2: Create ledger and credit ────────────────────────
    banner("2. Create ledger, credit 1000 api_sats")
    ledger = UserLedger()
    ledger.credit_deposit(api_sats=1000, invoice_id="demo-inv-001")
    print(f"  Balance: {ledger.balance_api_sats}")

    # ── Step 3: Store via AuditedVault (triggers audit event) ───
    banner("3. Store ledger → triggers Nostr audit event (flush)")
    version = await vault.store_ledger(DEMO_USER, ledger.to_json())
    print(f"  Stored → version={version}")
    print(f"  Audit event published (fire-and-forget)")

    # Small delay for thread to complete
    await asyncio.sleep(2)

    # ── Step 4: Debit and store again ───────────────────────────
    banner("4. Debit 50 api_sats → store again")
    ledger.debit("brain_query", 50)
    print(f"  Balance after debit: {ledger.balance_api_sats}")
    version = await vault.store_ledger(DEMO_USER, ledger.to_json())
    print(f"  Stored → version={version}")
    print(f"  Second audit event published")

    await asyncio.sleep(2)

    # ── Step 5: Snapshot (triggers audit event) ─────────────────
    banner("5. Snapshot → triggers Nostr audit event (snapshot)")
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    snap_id = await vault.snapshot_ledger(DEMO_USER, ledger.to_json(), ts)
    print(f"  Snapshot → id={snap_id}")
    print(f"  Snapshot audit event published")

    await asyncio.sleep(2)

    # ── Step 6: Verify on relay ─────────────────────────────────
    banner("6. Verify — subscribe to relay for audit events")
    try:
        from websocket import create_connection

        relay = relays[0]
        sub_filter = json.dumps(
            ["REQ", "demo-verify", {"kinds": [30078], "#t": ["tollbooth-audit"], "limit": 5}]
        )
        print(f"  Connecting to {relay}...")
        sslopt: dict[str, Any] = {}
        ws = create_connection(relay, timeout=10, sslopt=sslopt)
        try:
            ws.send(sub_filter)
            found = 0
            for _ in range(10):  # Read up to 10 messages
                resp = ws.recv()
                msg = json.loads(resp)
                if msg[0] == "EVENT":
                    event = msg[2]
                    content = json.loads(event.get("content", "{}"))
                    print(
                        f"  Found event: balance={content.get('balance')}, "
                        f"type={content.get('event_type')}, "
                        f"user={content.get('user_id', '')[:20]}"
                    )
                    found += 1
                elif msg[0] == "EOSE":
                    break
            if found == 0:
                print("  No audit events found (relay may take a moment to index)")
            else:
                print(f"  Found {found} audit event(s) on relay")
        finally:
            ws.close()
    except ImportError:
        print("  websocket-client not installed — skipping relay verification")
    except Exception as e:
        print(f"  Relay verification failed (non-fatal): {e}")

    # ── Summary ─────────────────────────────────────────────────
    banner("Summary")
    print(f"  3 audit events published (2 flushes + 1 snapshot)")
    print(f"  Final balance: {ledger.balance_api_sats} api_sats")
    print(f"  Verify with filter:")
    print(f'    {{"kinds": [30078], "#t": ["tollbooth-audit"]}}')
    print(f"\n  Done ✓")


if __name__ == "__main__":
    asyncio.run(main())
