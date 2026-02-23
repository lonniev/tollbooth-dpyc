"""NeonVault — VaultBackend implementation using Neon serverless Postgres.

Self-contained: uses httpx to call Neon's SQL-over-HTTP API. No new
dependencies beyond what tollbooth-dpyc already requires. Provides ACID
ledger persistence with optimistic concurrency control and an append-only
transaction journal for audit.

Neon HTTP API:
- Endpoint: https://{host}/sql (derived from NEON_DATABASE_URL)
- Auth: Neon-Connection-String header with full connection string
- Request: {"query": "SELECT $1::text", "params": ["hello"]}
- Response: {"fields": [...], "rows": [{"text": "hello"}], "rowCount": 1, "command": "SELECT"}

Call ``ensure_schema()`` once at startup to create tables if they don't exist.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


class NeonQueryError(Exception):
    """Raised when a Neon SQL query returns an error."""


class NeonVault:
    """Vault persistence via Neon serverless Postgres HTTP API.

    Implements the tollbooth ``VaultBackend`` protocol:

    - ``store_ledger(user_id, ledger_json) -> str``
    - ``fetch_ledger(user_id) -> str | None``
    - ``snapshot_ledger(user_id, ledger_json, timestamp) -> str | None``

    Uses optimistic concurrency control via a ``version`` column in the
    ``balances`` table. Stores the full ``UserLedger.to_json()`` blob
    and maintains an append-only ``transactions`` journal for snapshots.

    Configuration:

    - ``database_url``: Standard Postgres connection string
      (``postgres://user:pass@ep-xxx.region.aws.neon.tech/dbname``)
    - ``http_endpoint``: Optional explicit HTTP endpoint URL. If not
      provided, derived from ``database_url`` host as ``https://{host}/sql``.
    """

    def __init__(
        self,
        database_url: str,
        http_endpoint: str | None = None,
    ) -> None:
        parsed = urlparse(database_url)

        if http_endpoint:
            self._endpoint = http_endpoint.rstrip("/")
        else:
            self._endpoint = f"https://{parsed.hostname}/sql"

        self._client = httpx.AsyncClient(
            headers={
                "Neon-Connection-String": database_url,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._version_cache: dict[str, int] = {}

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # -- SQL helpers ---------------------------------------------------------

    async def _execute(
        self,
        query: str,
        params: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single SQL statement via Neon HTTP API.

        Returns the result dict with ``rows``, ``rowCount``, ``command``, etc.
        Raises ``NeonQueryError`` on SQL errors, ``httpx.HTTPStatusError``
        on transport errors.
        """
        body = {"query": query, "params": params or []}
        resp = await self._client.post(self._endpoint, json=body)
        resp.raise_for_status()
        data = resp.json()

        # Neon returns SQL errors in the response body with a "message" field
        if isinstance(data, dict) and "message" in data and "rows" not in data:
            raise NeonQueryError(data["message"])

        return data

    # -- VaultBackend protocol -----------------------------------------------

    async def store_ledger(self, user_id: str, ledger_json: str) -> str:
        """UPSERT ledger JSON into ``balances`` table with version increment.

        Uses optimistic concurrency: if a version is cached from a prior
        ``fetch_ledger``, issues an UPDATE with a version guard. On version
        conflict (0 rows affected), falls through to a full UPSERT.

        Returns the new version as a string.
        """
        cached_version = self._version_cache.get(user_id)

        if cached_version is not None:
            result = await self._execute(
                "UPDATE balances "
                "SET ledger_json = $1, version = version + 1, last_flush = now() "
                "WHERE npub = $2 AND version = $3 "
                "RETURNING version",
                [ledger_json, user_id, cached_version],
            )
            rows = result.get("rows", [])
            if rows:
                new_version = rows[0]["version"]
                self._version_cache[user_id] = new_version
                return str(new_version)

            logger.info(
                "Version conflict for %s (cached v%d), falling through to upsert.",
                user_id,
                cached_version,
            )

        # Full UPSERT — handles both first-time inserts and conflict recovery
        result = await self._execute(
            "INSERT INTO balances (npub, ledger_json, version, last_flush, created_at) "
            "VALUES ($1, $2, 1, now(), now()) "
            "ON CONFLICT (npub) DO UPDATE "
            "SET ledger_json = EXCLUDED.ledger_json, "
            "    version = balances.version + 1, "
            "    last_flush = now() "
            "RETURNING version",
            [user_id, ledger_json],
        )
        rows = result.get("rows", [])
        if rows:
            new_version = rows[0]["version"]
            self._version_cache[user_id] = new_version
            return str(new_version)

        raise NeonQueryError("UPSERT returned no rows")

    async def fetch_ledger(self, user_id: str) -> str | None:
        """Fetch the current ledger JSON for a user.

        Returns the ledger JSON string, or ``None`` if no record exists.
        Also caches the version for subsequent optimistic updates.
        """
        result = await self._execute(
            "SELECT ledger_json, version FROM balances WHERE npub = $1",
            [user_id],
        )
        rows = result.get("rows", [])
        if not rows:
            return None

        ledger_json = rows[0]["ledger_json"]
        version = rows[0]["version"]
        self._version_cache[user_id] = version
        return ledger_json

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str,
    ) -> str | None:
        """Store a timestamped snapshot in the transactions journal.

        First updates ``balances`` via ``store_ledger``, then inserts a
        ``snapshot`` record into the ``transactions`` table. Returns the
        transaction ID as a string, or ``None`` if the journal insert fails.
        """
        await self.store_ledger(user_id, ledger_json)

        try:
            balance = self._extract_balance(ledger_json)
            result = await self._execute(
                "INSERT INTO transactions "
                "(npub, tx_type, amount_api_sats, detail, balance_after, created_at) "
                "VALUES ($1, 'snapshot', 0, $2, $3, $4::timestamptz) "
                "RETURNING id",
                [user_id, f"Snapshot at {timestamp}", balance, timestamp],
            )
            rows = result.get("rows", [])
            if rows:
                return str(rows[0]["id"])
        except (NeonQueryError, httpx.HTTPError) as e:
            logger.warning("Failed to record snapshot for %s: %s", user_id, e)

        return None

    # -- Schema management ---------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the ``balances`` and ``transactions`` tables if they don't exist.

        Safe to call on every startup — uses ``IF NOT EXISTS``.
        """
        await self._execute(
            "CREATE TABLE IF NOT EXISTS balances ("
            "    npub TEXT PRIMARY KEY,"
            "    ledger_json TEXT NOT NULL,"
            "    version INTEGER NOT NULL DEFAULT 1,"
            "    last_flush TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        await self._execute(
            "CREATE TABLE IF NOT EXISTS transactions ("
            "    id BIGSERIAL PRIMARY KEY,"
            "    npub TEXT NOT NULL,"
            "    tx_type TEXT NOT NULL,"
            "    amount_api_sats INTEGER NOT NULL,"
            "    tool_name TEXT,"
            "    invoice_id TEXT,"
            "    detail TEXT,"
            "    balance_after INTEGER NOT NULL,"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_npub "
            "ON transactions(npub)"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_created "
            "ON transactions(created_at)"
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _extract_balance(ledger_json: str) -> int:
        """Extract the balance from a ledger JSON string.

        Sums ``remaining_sats`` across all tranches. Returns 0 on parse error.
        """
        try:
            obj = json.loads(ledger_json)
            return sum(t.get("remaining_sats", 0) for t in obj.get("tranches", []))
        except (json.JSONDecodeError, TypeError, AttributeError):
            return 0
