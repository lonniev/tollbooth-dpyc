"""Claim-check async jobs — persistence for slow Operator tools.

Some Operator tools know their work is slow (an LLM round-trip, a web-search
augmented generation). Such a tool returns like any normal MCP tool — it just
returns a *claim check* instead of the end item. The actual work runs as a
concurrent asyncio task in the Operator's process, and the result persists in
the Operator's Neon db. A companion tool (defined by the Operator) redeems the
claim for the result. Nothing here is exposed on the wire by the wheel — this
module is library code consumed via ``OperatorRuntime.start_async_job`` /
``fetch_async_job``.

The tool itself codes how long it may take (``max_runtime_seconds``, also the
staleness threshold for re-claiming a job orphaned by a serverless container
recycle) and how long a finished result is kept (``result_ttl_seconds``).

The work runs either in-process (an asyncio task) or on a detached executor
(see ``tollbooth/async_executor.py``); in the detached case ``run_handle`` holds
the executor's handle so a later fetch can poll it for the result.

The result is an opaque JSON blob: either the output content itself (e.g. the
LLM response) or state the Operator can use to find the output elsewhere
(e.g. ``{"entry_id": ...}``).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Lifecycle: pending → running → done | error.
# A 'running' row whose started_at is older than its own max_runtime_seconds
# is presumed orphaned (container recycled mid-job) and may be re-claimed.

_ROW_FIELDS = (
    "claim, npub, kind, tool_id, params, status, attempts, "
    "max_runtime_seconds, expected_seconds, result_ttl_seconds, result, error, run_handle, "
    "EXTRACT(EPOCH FROM (now() - created_at)) AS elapsed_seconds, "
    "(status = 'running' AND started_at IS NOT NULL AND "
    " started_at < now() - make_interval(secs => max_runtime_seconds)) AS stalled, "
    "(expires_at IS NOT NULL AND expires_at < now()) AS expired"
)

# Polling cadence advised to an agent redeeming a claim check. A constant 3s
# tick hammers a long job with dozens of idle round-trips; naive exponential
# backoff (wait 1s, 2s, 4s, …) is worse — it tells you to wait *longest* right
# when the result is most imminent, since the longer a job has already run the
# CLOSER it is to done, not farther. So we count DOWN toward the deadline by
# which the job is guaranteed resolved (its declared max_runtime): the advised
# interval holds at a steady ceiling through the bulk of the run (bounding how
# long a finished result waits to be noticed) and TIGHTENS toward a floor in the
# home stretch. (The web frontend has its own client-side backoff and ignores
# this field; this is the advice an AI agent honors.)
_POLL_FLOOR_SECONDS = 5
_POLL_CEILING_SECONDS = 30
_POLL_GROWTH = 0.5  # next delay ≈ half the time still expected to remain

# When a caller declares a real time budget (expected_seconds > 0 — e.g. an
# author who sized and is paying ad valorem for a dynamic block), trust it:
# sleep most of it up front rather than polling through the middle. The first
# wait is ~75% of the budget, then each subsequent wait is ~75% of what's left,
# so it converges geometrically as the finish nears. One single hop is never
# advised longer than this absolute cap (the floor still applies near/after the
# budget). This is distinct from the no-estimate path, where max_runtime is a
# safety ceiling, not a prediction, so a long first wait would be wrong.
_POLL_BUDGET_GROWTH = 0.75
_POLL_BUDGET_CEILING_SECONDS = 300


def poll_backoff_seconds(
    elapsed_seconds: float,
    max_runtime_seconds: int,
    expected_seconds: int = 0,
) -> int:
    """Advise the next ``poll_after_seconds`` for a still-running async job.

    Counts down toward the finish rather than up from the start, so the interval
    SHRINKS as the job nears done (the opposite of exponential backoff). Two
    regimes, selected by whether the caller declared a time budget:

    * ``expected_seconds > 0`` — a real author-declared budget: first wait ≈75%
      of it, then ≈75% of the remaining each poll (geometric tightening).
    * otherwise — ``max_runtime_seconds`` is only a safety ceiling: hold at a
      steady ceiling (~max/10) through the bulk and tighten to a floor near the
      deadline.

    Pure function; both regimes are floored so a finished result is never left
    waiting longer than necessary.
    """
    if expected_seconds > 0:
        remaining = expected_seconds - elapsed_seconds
        suggested = remaining * _POLL_BUDGET_GROWTH
        return int(max(_POLL_FLOOR_SECONDS, min(suggested, _POLL_BUDGET_CEILING_SECONDS)))
    ceiling = 15
    if max_runtime_seconds > 0:
        ceiling = max(_POLL_FLOOR_SECONDS, min(max_runtime_seconds // 10, _POLL_CEILING_SECONDS))
    remaining = max_runtime_seconds - elapsed_seconds
    suggested = remaining * _POLL_GROWTH
    return int(max(_POLL_FLOOR_SECONDS, min(suggested, ceiling)))


def _as_dict(value: Any) -> dict[str, Any]:
    """JSON values cross the Neon HTTP boundary untyped — cast here."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("t", "true", "1")


def _row_to_job(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Neon row at the deserialization boundary."""
    return {
        "claim": str(row.get("claim", "")),
        "npub": str(row.get("npub", "")),
        "kind": str(row.get("kind", "")),
        "tool_id": str(row.get("tool_id", "")),
        "params": _as_dict(row.get("params")),
        "status": str(row.get("status", "")),
        "attempts": int(row.get("attempts") or 0),
        "max_runtime_seconds": int(row.get("max_runtime_seconds") or 0),
        "expected_seconds": int(row.get("expected_seconds") or 0),
        "result_ttl_seconds": int(row.get("result_ttl_seconds") or 0),
        "result": _as_dict(row.get("result")),
        "error": str(row.get("error") or ""),
        "run_handle": str(row.get("run_handle") or ""),
        "elapsed_seconds": float(row.get("elapsed_seconds") or 0.0),
        "stalled": _as_bool(row.get("stalled")),
        "expired": _as_bool(row.get("expired")),
    }


def valid_claim(claim: str) -> bool:
    """Claims arrive as adversarial tool arguments — validate before SQL."""
    try:
        uuid.UUID(str(claim))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


class AsyncJobStore:
    """Neon-backed store for claim-check jobs.

    Thin wrapper over the vault's HTTP SQL API — one instance per call site
    is fine; all state lives in Postgres.
    """

    def __init__(self, vault: Any) -> None:
        self._vault = vault

    def _t(self, table: str) -> str:
        return self._vault._t(table)

    async def create(
        self,
        *,
        npub: str,
        kind: str,
        tool_id: str,
        params: dict[str, Any],
        max_runtime_seconds: int,
        result_ttl_seconds: int,
        expected_seconds: int = 0,
    ) -> str:
        """Persist a new pending job; return its claim check."""
        result = await self._vault._execute(
            f"INSERT INTO {self._t('async_jobs')} "
            "(npub, kind, tool_id, params, max_runtime_seconds, result_ttl_seconds, expected_seconds) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7) RETURNING claim",
            [
                npub,
                kind,
                tool_id,
                json.dumps(params),
                int(max_runtime_seconds),
                int(result_ttl_seconds),
                int(expected_seconds),
            ],
        )
        return str(result["rows"][0]["claim"])

    async def claim_for_run(self, claim: str) -> dict[str, Any] | None:
        """Atomically take ownership of a runnable job.

        Claimable: 'pending', or 'running' past its own max_runtime
        (orphaned by a recycled container). Returns the claimed row, or
        None when another container owns it / it already finished.
        """
        if not valid_claim(claim):
            return None
        result = await self._vault._execute(
            f"UPDATE {self._t('async_jobs')} "
            "SET status = 'running', attempts = attempts + 1, started_at = now() "
            "WHERE claim = $1 AND (status = 'pending' OR (status = 'running' "
            "AND started_at < now() - make_interval(secs => max_runtime_seconds))) "
            f"RETURNING {_ROW_FIELDS}",
            [claim],
        )
        # Neon's HTTP SQL API reports camelCase rowCount — `rowcount`
        # would silently report 0 and double-run every job.
        if int(result.get("rowCount") or 0) < 1:
            return None
        rows = result.get("rows") or []
        return _row_to_job(rows[0]) if rows else None

    async def complete(self, claim: str, result_blob: dict[str, Any]) -> None:
        """Mark done; result kept until its coded TTL elapses."""
        await self._vault._execute(
            f"UPDATE {self._t('async_jobs')} "
            "SET status = 'done', result = $2::jsonb, completed_at = now(), "
            "expires_at = now() + make_interval(secs => result_ttl_seconds) "
            "WHERE claim = $1",
            [claim, json.dumps(result_blob)],
        )

    async def fail(self, claim: str, error: str) -> None:
        """Mark terminally failed; the error state also expires on the TTL."""
        await self._vault._execute(
            f"UPDATE {self._t('async_jobs')} "
            "SET status = 'error', error = $2, completed_at = now(), "
            "expires_at = now() + make_interval(secs => result_ttl_seconds) "
            "WHERE claim = $1",
            [claim, error],
        )

    async def release(self, claim: str) -> None:
        """Return a running job to 'pending' for another attempt."""
        await self._vault._execute(
            f"UPDATE {self._t('async_jobs')} "
            "SET status = 'pending' WHERE claim = $1 AND status = 'running'",
            [claim],
        )

    async def set_run_handle(self, claim: str, handle: str) -> None:
        """Record a detached executor's handle (e.g. a Prefect flow-run id) so
        a later fetch can poll it for the result."""
        if not valid_claim(claim):
            return
        await self._vault._execute(
            f"UPDATE {self._t('async_jobs')} SET run_handle = $2 WHERE claim = $1",
            [claim, handle],
        )

    async def get(self, claim: str, npub: str) -> dict[str, Any] | None:
        """Fetch a job — claim AND npub must match.

        A claim check alone never unlocks another patron's job.
        """
        if not valid_claim(claim):
            return None
        result = await self._vault._execute(
            f"SELECT {_ROW_FIELDS} FROM {self._t('async_jobs')} "
            "WHERE claim = $1 AND npub = $2",
            [claim, npub],
        )
        rows = result.get("rows") or []
        return _row_to_job(rows[0]) if rows else None

    async def purge_expired(self) -> int:
        """Drop expired results and long-abandoned open jobs."""
        result = await self._vault._execute(
            f"DELETE FROM {self._t('async_jobs')} "
            "WHERE (expires_at IS NOT NULL AND expires_at < now()) "
            "OR (status IN ('pending', 'running') "
            "AND created_at < now() - interval '24 hours')"
        )
        return int(result.get("rowCount") or 0)
