"""Durable store of what operators ask an Authority for: to be adopted, or to
change their registry entry.

Each request is one operator's own act, proven on its own connection; the
Authority owner decides later, on their own time. A request is a row keyed by
the operator's npub in its kind's table, durable in the Authority's Neon vault
rather than process memory, so concurrent requests and serverless restarts are
both fine.

Persistence reuses the Authority's bootstrapped ``NeonVault`` via its HTTP SQL
helper (``_execute``) and schema-prefix helper (``_t``). The Neon HTTP API
ignores ``search_path`` (hence ``_t``) and returns affected-row counts under the
camelCase key ``rowCount``.
"""

from __future__ import annotations

from typing import Any

# What an operator may ask for, and the table each kind of request lives in.
ADOPTION = "adoption"
UPDATE = "update"
TABLES = {
    ADOPTION: "operator_adoption_requests",
    UPDATE: "operator_update_requests",
}

# Lifecycle of a request.
PENDING = "pending"
REJECTED = "rejected"
PROVISIONED = "provisioned"  # an adoption, approved
APPLIED = "applied"          # an update, approved
CLOSED = "closed"            # the operator left, or the Authority disowned it

# A pending request that no owner acts on eventually lapses.
DEFAULT_TTL_SECONDS = 7 * 24 * 3600

_COLUMNS = "operator_npub, service_url, display_name, status, note, requested_at, decided_at, expires_at"


def _table(vault: Any, kind: str) -> str:
    if kind not in TABLES:
        raise ValueError(f"Unknown request kind {kind!r}; expected one of {sorted(TABLES)}.")
    return str(vault._t(TABLES[kind]))


async def ensure_schema(vault: Any) -> None:
    """Create the request tables if absent, and bring an older adoption table
    up to the shared shape."""
    for kind in TABLES:
        await vault._execute(
            f"CREATE TABLE IF NOT EXISTS {_table(vault, kind)} ("
            "  operator_npub TEXT PRIMARY KEY,"
            "  service_url TEXT NOT NULL DEFAULT '',"
            "  display_name TEXT NOT NULL DEFAULT '',"
            f"  status TEXT NOT NULL DEFAULT '{PENDING}',"
            "  note TEXT NOT NULL DEFAULT '',"
            "  requested_at TIMESTAMPTZ DEFAULT now(),"
            "  decided_at TIMESTAMPTZ,"
            f"  expires_at TIMESTAMPTZ DEFAULT now() + interval '{DEFAULT_TTL_SECONDS} seconds'"
            ")"
        )
    adoptions = _table(vault, ADOPTION)
    await vault._execute(
        f"ALTER TABLE {adoptions} "
        "ADD COLUMN IF NOT EXISTS display_name TEXT NOT NULL DEFAULT '', "
        "DROP COLUMN IF EXISTS dpop_token_hash"
    )


async def upsert_pending(
    vault: Any,
    kind: str,
    operator_npub: str,
    *,
    service_url: str = "",
    display_name: str = "",
    note: str = "",
) -> None:
    """Record (or refresh) a pending request.

    Idempotent on operator_npub: a repeat resets the row to ``pending`` with
    the new fields and a fresh TTL, so a rejected operator can ask again and
    one that re-sends is not duplicated.
    """
    await vault._execute(
        f"INSERT INTO {_table(vault, kind)} "
        "(operator_npub, service_url, display_name, status, note, "
        " requested_at, decided_at, expires_at) "
        f"VALUES ($1, $2, $3, '{PENDING}', $4, now(), NULL, "
        f"        now() + interval '{DEFAULT_TTL_SECONDS} seconds') "
        "ON CONFLICT (operator_npub) DO UPDATE SET "
        "  service_url = EXCLUDED.service_url,"
        "  display_name = EXCLUDED.display_name,"
        f"  status = '{PENDING}',"
        "  note = EXCLUDED.note,"
        "  requested_at = now(),"
        "  decided_at = NULL,"
        f"  expires_at = now() + interval '{DEFAULT_TTL_SECONDS} seconds'",
        [operator_npub, service_url, display_name, note],
    )


async def get(vault: Any, kind: str, operator_npub: str) -> dict[str, Any] | None:
    """Return *operator_npub*'s request of *kind*, or None."""
    result = await vault._execute(
        f"SELECT {_COLUMNS} FROM {_table(vault, kind)} WHERE operator_npub = $1",
        [operator_npub],
    )
    rows = result.get("rows") or []
    return dict(rows[0]) if rows else None


async def list_pending(vault: Any) -> list[dict[str, Any]]:
    """Every non-expired pending request of every kind, newest first, each
    naming its ``kind``."""
    pending: list[dict[str, Any]] = []
    for kind in TABLES:
        result = await vault._execute(
            f"SELECT {_COLUMNS} FROM {_table(vault, kind)} "
            f"WHERE status = '{PENDING}' AND expires_at > now()",
        )
        pending += [{**dict(r), "kind": kind} for r in (result.get("rows") or [])]
    return sorted(pending, key=lambda r: str(r.get("requested_at") or ""), reverse=True)


async def mark(vault: Any, kind: str, operator_npub: str, status: str) -> bool:
    """Move *operator_npub*'s request of *kind* to *status*, stamping
    decided_at. Returns whether there was one."""
    result = await vault._execute(
        f"UPDATE {_table(vault, kind)} SET status = $2, decided_at = now() "
        "WHERE operator_npub = $1",
        [operator_npub, status],
    )
    # Neon HTTP API returns affected rows under camelCase "rowCount".
    return (result.get("rowCount", 0) or 0) > 0


async def prune_expired(vault: Any) -> int:
    """Delete lapsed pending requests of every kind. Returns the number removed."""
    removed = 0
    for kind in TABLES:
        result = await vault._execute(
            f"DELETE FROM {_table(vault, kind)} "
            f"WHERE status = '{PENDING}' AND expires_at <= now()",
        )
        removed += result.get("rowCount", 0) or 0
    return removed
