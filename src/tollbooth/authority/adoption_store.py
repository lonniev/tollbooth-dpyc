"""Durable store for deferred operator-adoption requests.

When an operator asks an Authority to adopt it, the request is recorded
here as a ``pending`` row keyed by the operator's npub — durable in the
Authority's Neon vault, not in process memory. This is deliberate: the
Authority-onboarding precedent kept its challenge in an in-memory
singleton (one-at-a-time, lost on a Horizon cold start). Operator adoption
must tolerate concurrent requests and serverless restarts, so its pending
state lives in Postgres and the owner reviews it on their own time.

Persistence reuses the Authority's bootstrapped ``NeonVault`` via its HTTP
SQL helper (``_execute``) and schema-prefix helper (``_t``). The Neon HTTP
API ignores ``search_path`` (hence ``_t``) and returns affected-row counts
under the camelCase key ``rowCount``.
"""

from __future__ import annotations

from typing import Any

TABLE = "operator_adoption_requests"

# Lifecycle of a request.
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
PROVISIONED = "provisioned"

# A pending request that no owner acts on eventually lapses.
DEFAULT_TTL_SECONDS = 7 * 24 * 3600


async def ensure_schema(vault: Any) -> None:
    """Create the operator_adoption_requests table if absent."""
    await vault._execute(
        f"CREATE TABLE IF NOT EXISTS {vault._t(TABLE)} ("
        "  operator_npub TEXT PRIMARY KEY,"
        "  service_url TEXT NOT NULL DEFAULT '',"
        "  status TEXT NOT NULL DEFAULT 'pending',"
        "  dpop_token_hash TEXT NOT NULL DEFAULT '',"
        "  note TEXT NOT NULL DEFAULT '',"
        "  requested_at TIMESTAMPTZ DEFAULT now(),"
        "  decided_at TIMESTAMPTZ,"
        f"  expires_at TIMESTAMPTZ DEFAULT now() + interval '{DEFAULT_TTL_SECONDS} seconds'"
        ")"
    )


async def upsert_pending(
    vault: Any,
    operator_npub: str,
    service_url: str,
    *,
    note: str = "",
    dpop_token_hash: str = "",
) -> None:
    """Record (or refresh) a pending adoption request.

    Idempotent on operator_npub: a re-request resets the row to ``pending``
    with a fresh TTL, so a previously-rejected operator can re-apply and an
    operator that re-sends is not duplicated.
    """
    await vault._execute(
        f"INSERT INTO {vault._t(TABLE)} "
        "(operator_npub, service_url, status, dpop_token_hash, note, "
        " requested_at, decided_at, expires_at) "
        f"VALUES ($1, $2, '{PENDING}', $3, $4, now(), NULL, "
        f"        now() + interval '{DEFAULT_TTL_SECONDS} seconds') "
        "ON CONFLICT (operator_npub) DO UPDATE SET "
        "  service_url = EXCLUDED.service_url,"
        f"  status = '{PENDING}',"
        "  dpop_token_hash = EXCLUDED.dpop_token_hash,"
        "  note = EXCLUDED.note,"
        "  requested_at = now(),"
        "  decided_at = NULL,"
        f"  expires_at = now() + interval '{DEFAULT_TTL_SECONDS} seconds'",
        [operator_npub, service_url, dpop_token_hash, note],
    )


async def get(vault: Any, operator_npub: str) -> dict[str, Any] | None:
    """Return the adoption request row for *operator_npub*, or None."""
    result = await vault._execute(
        f"SELECT operator_npub, service_url, status, note, "
        f"requested_at, decided_at, expires_at "
        f"FROM {vault._t(TABLE)} WHERE operator_npub = $1",
        [operator_npub],
    )
    rows = result.get("rows") or []
    return dict(rows[0]) if rows else None


async def list_pending(vault: Any) -> list[dict[str, Any]]:
    """Return all non-expired pending requests, newest first."""
    result = await vault._execute(
        f"SELECT operator_npub, service_url, note, requested_at, expires_at "
        f"FROM {vault._t(TABLE)} "
        f"WHERE status = '{PENDING}' AND expires_at > now() "
        "ORDER BY requested_at DESC",
    )
    return [dict(r) for r in (result.get("rows") or [])]


async def mark(vault: Any, operator_npub: str, status: str) -> bool:
    """Transition a request to *status*, stamping decided_at. Returns hit."""
    result = await vault._execute(
        f"UPDATE {vault._t(TABLE)} SET status = $2, decided_at = now() "
        f"WHERE operator_npub = $1",
        [operator_npub, status],
    )
    # Neon HTTP API returns affected rows under camelCase "rowCount".
    return (result.get("rowCount", 0) or 0) > 0


async def prune_expired(vault: Any) -> int:
    """Delete lapsed pending requests. Returns the number removed."""
    result = await vault._execute(
        f"DELETE FROM {vault._t(TABLE)} "
        f"WHERE status = '{PENDING}' AND expires_at <= now()",
    )
    return result.get("rowCount", 0) or 0
