"""Durable store for operator Neon-402 alerts.

When an operator's database exhausts its provider (Neon) compute/storage
quota, the operator's runtime reports it MCP-to-MCP to its Authority — the
Authority provisions and is responsible for the books, so it must learn the
instant they lock rather than from a patron complaint. Each report is
recorded here, keyed by the operator's npub, durable in the Authority's own
Neon vault (a DIFFERENT project from the operator's, so it is reachable even
while the operator's books are dark).

The row is a latest-state record, not an append log: a re-report from the
same operator refreshes ``last_seen_at`` and bumps ``seen_count`` so a
flapping outage doesn't bloat the table. The Authority owner clears a row
once capacity is restored.

Persistence reuses the Authority's ``NeonVault`` HTTP SQL helper
(``_execute``) and schema-prefix helper (``_t``); Neon returns affected-row
counts under the camelCase key ``rowCount``.
"""

from __future__ import annotations

from typing import Any

TABLE = "operator_neon_alerts"


async def ensure_schema(vault: Any) -> None:
    """Create the operator_neon_alerts table if absent."""
    await vault._execute(
        f"CREATE TABLE IF NOT EXISTS {vault._t(TABLE)} ("
        "  operator_npub TEXT PRIMARY KEY,"
        "  detail TEXT NOT NULL DEFAULT '',"
        "  seen_count INTEGER NOT NULL DEFAULT 1,"
        "  first_seen_at TIMESTAMPTZ DEFAULT now(),"
        "  last_seen_at TIMESTAMPTZ DEFAULT now()"
        ")"
    )


async def record(vault: Any, operator_npub: str, detail: str = "") -> None:
    """Record (or refresh) a Neon-402 alert for *operator_npub*.

    Idempotent on operator_npub: a re-report keeps the first-seen timestamp,
    refreshes the detail + last-seen, and increments the count.
    """
    await vault._execute(
        f"INSERT INTO {vault._t(TABLE)} "
        "(operator_npub, detail, seen_count, first_seen_at, last_seen_at) "
        "VALUES ($1, $2, 1, now(), now()) "
        "ON CONFLICT (operator_npub) DO UPDATE SET "
        "  detail = EXCLUDED.detail,"
        f"  seen_count = {vault._t(TABLE)}.seen_count + 1,"
        "  last_seen_at = now()",
        [operator_npub, detail[:500]],
    )


async def list_all(vault: Any) -> list[dict[str, Any]]:
    """Return all outstanding Neon-402 alerts, most-recently-seen first."""
    result = await vault._execute(
        f"SELECT operator_npub, detail, seen_count, first_seen_at, last_seen_at "
        f"FROM {vault._t(TABLE)} ORDER BY last_seen_at DESC",
    )
    return [dict(r) for r in (result.get("rows") or [])]


async def clear(vault: Any, operator_npub: str) -> bool:
    """Clear the alert for *operator_npub* (capacity restored). Returns hit."""
    result = await vault._execute(
        f"DELETE FROM {vault._t(TABLE)} WHERE operator_npub = $1",
        [operator_npub],
    )
    return (result.get("rowCount", 0) or 0) > 0
