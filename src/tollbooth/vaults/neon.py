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
import re
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
        encryption_nsec_hex: str | None = None,
    ) -> None:
        parsed = urlparse(database_url)

        # Keep the pooler endpoint — Neon HTTP SQL API requires it.
        # The direct (non-pooler) endpoint only supports Postgres wire protocol.
        hostname = parsed.hostname or ""

        if http_endpoint:
            self._endpoint = http_endpoint.rstrip("/")
        else:
            self._endpoint = f"https://{hostname}/sql"

        # Resolve operator schema prefix for schema-qualified queries.
        # The Neon HTTP SQL API doesn't honor the options search_path from
        # the connection string, so ALL table references must be explicit.
        self._schema_prefix = ""
        try:
            from urllib.parse import parse_qs as _pqs
            _params = _pqs(parsed.query)
            _options = _params.get("options", [""])[0]
            if "search_path=" in _options:
                _sp = _options.split("search_path=", 1)[1].split("&")[0].split()[0]
                _first = _sp.split(",")[0].strip()
                if _first and _first != "public":
                    if not re.match(r"^[a-z][a-z0-9_]*$", _first):
                        raise ValueError(f"Unsafe schema name in search_path: {_first!r}")
                    self._schema_prefix = f"{_first}."
                    logger.info("Neon: schema prefix = %s", self._schema_prefix)
        except ValueError:
            raise
        except Exception as exc:
            logger.debug("Schema prefix parsing skipped (non-fatal): %s", exc)
        self._client = httpx.AsyncClient(
            headers={
                "Neon-Connection-String": database_url,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._version_cache: dict[str, int] = {}

        # Field encryption — if nsec provided, all stored values are AES-256-GCM encrypted.
        # Without nsec, vault operates in plaintext mode (backward compatible).
        self._cipher = None
        if encryption_nsec_hex:
            from tollbooth.vault_encryption import VaultCipher
            self._cipher = VaultCipher(nsec_hex=encryption_nsec_hex)

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt if cipher is configured, otherwise passthrough."""
        return self._cipher.encrypt(plaintext) if self._cipher else plaintext

    def _decrypt(self, value: str) -> str:
        """Decrypt if cipher is configured. Handles migration from plaintext."""
        if not self._cipher:
            return value
        if self._cipher.is_encrypted(value):
            return self._cipher.decrypt(value)
        return value  # Legacy plaintext — return as-is

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    def _t(self, table: str) -> str:
        """Return schema-qualified table name."""
        return f"{self._schema_prefix}{table}"

    # -- SQL helpers ---------------------------------------------------------

    async def _execute(
        self,
        query: str,
        params: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a single SQL statement via Neon HTTP API.

        Returns the result dict with ``rows``, ``rowCount``, ``command``, etc.
        Raises ``NeonQueryError`` on SQL errors with the Neon-supplied message,
        including 4xx HTTP responses (Neon's REST gateway returns 400 with a
        SQL error message in the body for things like missing relations or
        permission denied). Raises ``httpx.HTTPStatusError`` only on 5xx or
        bodyless 4xx — anything where Neon didn't tell us why.
        """
        body = {"query": query, "params": params or []}
        resp = await self._client.post(self._endpoint, json=body)

        # Read the body before raise_for_status so 4xx error messages from
        # Neon (which arrive as `{"message": "..."}` in a 400 body) surface
        # to the caller instead of being lost behind an opaque
        # "Client error '400 Bad Request'". Previously the
        # raise_for_status() short-circuit prevented anyone from learning
        # whether the failure was "relation does not exist", "permission
        # denied", or a connection-level rejection.
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = None
            if isinstance(err_body, dict) and err_body.get("message"):
                raise NeonQueryError(
                    f"Neon HTTP {resp.status_code}: {err_body['message']} "
                    f"(query={query[:120]}…)"
                )
            # Body wasn't JSON or didn't have a message — fall through to
            # raise_for_status so callers still see the HTTP error.
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
        ledger_json = self._encrypt(ledger_json)
        cached_version = self._version_cache.get(user_id)

        if cached_version is not None:
            result = await self._execute(
                f"UPDATE {self._t('balances')} "
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
            f"INSERT INTO {self._t('balances')}(npub, ledger_json, version, last_flush, created_at) "
            "VALUES ($1, $2, 1, now(), now()) "
            "ON CONFLICT (npub) DO UPDATE "
            f"SET ledger_json = EXCLUDED.ledger_json, "
            f"    version = {self._t('balances')}.version + 1, "
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
            f"SELECT ledger_json, version FROM {self._t('balances')} WHERE npub = $1",
            [user_id],
        )
        rows = result.get("rows", [])
        if not rows:
            return None

        ledger_json = rows[0]["ledger_json"]
        version = rows[0]["version"]
        self._version_cache[user_id] = version
        return self._decrypt(ledger_json)

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
                f"INSERT INTO {self._t('transactions')} "
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

        When the connection uses a per-operator schema (search_path=op_xxx,public),
        tables must be created in the operator's schema explicitly. Otherwise
        CREATE TABLE IF NOT EXISTS sees the table in ``public`` and skips,
        leaving the operator's schema without its own tables.

        The per-operator role typically OWNS the schema (Authority transfers
        ownership at provisioning time) but does NOT have CREATE on the
        database itself. That means ``CREATE SCHEMA IF NOT EXISTS`` raises
        ``permission denied for database`` even when the schema already
        exists — Postgres checks the privilege before the IF NOT EXISTS
        short-circuit. So: probe ``pg_namespace`` first, and only attempt
        CREATE SCHEMA when the schema is genuinely missing.
        """
        # Ensure the operator's schema exists if we have one
        if self._schema_prefix:
            schema_name = self._schema_prefix.rstrip(".")
            idx_prefix = f"{schema_name}_"
            # Probe for existence — operator role can SELECT pg_namespace
            # even when it can't CREATE on the database.
            exists_result = await self._execute(
                "SELECT 1 FROM pg_namespace WHERE nspname = $1",
                [schema_name],
            )
            if not exists_result.get("rows"):
                # Genuinely missing — attempt to create (will succeed for
                # privileged roles, fail loud for unprivileged ones).
                await self._execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        else:
            idx_prefix = ""

        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('balances')} ("
            "    npub TEXT PRIMARY KEY,"
            "    ledger_json TEXT NOT NULL,"
            "    version INTEGER NOT NULL DEFAULT 1,"
            "    last_flush TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('transactions')} ("
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
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_transactions_npub "
            f"ON {self._t('transactions')}(npub)"
        )
        await self._execute(
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_transactions_created "
            f"ON {self._t('transactions')}(created_at)"
        )
        # -- Anchors table (OTS Bitcoin anchoring) --
        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('anchors')} ("
            "    id BIGSERIAL PRIMARY KEY,"
            "    root_hash TEXT NOT NULL UNIQUE,"
            "    leaf_count INTEGER NOT NULL,"
            "    status TEXT NOT NULL DEFAULT 'pending',"
            "    ots_receipts_json TEXT,"
            "    snapshot_json TEXT NOT NULL,"
            "    leaf_hashes_json TEXT NOT NULL,"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    confirmed_at TIMESTAMPTZ"
            ")"
        )
        await self._execute(
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_anchors_created "
            f"ON {self._t('anchors')}(created_at)"
        )
        await self._execute(
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_anchors_status "
            f"ON {self._t('anchors')}(status)"
        )
        # -- Global demand counters (surge pricing) --
        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('tool_demand')} ("
            "    tool_name TEXT NOT NULL,"
            "    window_key TEXT NOT NULL,"
            "    count INTEGER NOT NULL DEFAULT 0,"
            "    PRIMARY KEY (tool_name, window_key)"
            ")"
        )
        # -- Authority configuration (curator npub, onboarding state) --
        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('authority_config')} ("
            "    key TEXT PRIMARY KEY,"
            "    value TEXT NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        )
        # -- Operator pricing models (runtime-configurable tool pricing) --
        await self._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('operator_pricing_models')} ("
            "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "    operator TEXT NOT NULL,"
            "    name TEXT NOT NULL,"
            "    model_json JSONB NOT NULL,"
            "    is_active BOOLEAN DEFAULT false,"
            "    created_at TIMESTAMPTZ DEFAULT now(),"
            "    updated_at TIMESTAMPTZ DEFAULT now()"
            ")"
        )
        await self._execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {idx_prefix}one_active_per_operator "
            f"ON {self._t('operator_pricing_models')} (operator) WHERE is_active = true"
        )
        await self._execute(
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_pricing_models_operator "
            f"ON {self._t('operator_pricing_models')} (operator)"
        )

    # -- Global demand counters (surge pricing) --------------------------------

    async def get_demand(self, tool_name: str, window_key: str) -> int:
        """Read the global demand count for a tool in a time window.

        Returns 0 on miss or any error — callers get base pricing
        when demand data is unavailable.
        """
        try:
            result = await self._execute(
                f"SELECT count FROM {self._t('tool_demand')} "
                "WHERE tool_name = $1 AND window_key = $2",
                [tool_name, window_key],
            )
            rows = result.get("rows", [])
            return int(rows[0]["count"]) if rows else 0
        except Exception:
            logger.debug("get_demand failed for %s/%s", tool_name, window_key)
            return 0

    async def increment_demand(self, tool_name: str, window_key: str) -> None:
        """Atomically increment the demand counter (fire-and-forget safe).

        Designed to be called via ``asyncio.create_task()`` — errors are
        logged but never propagated.
        """
        try:
            await self._execute(
                f"INSERT INTO {self._t('tool_demand')} (tool_name, window_key, count) "
                "VALUES ($1, $2, 1) "
                "ON CONFLICT (tool_name, window_key) "
                f"DO UPDATE SET count = {self._t('tool_demand')}.count + 1",
                [tool_name, window_key],
            )
        except Exception:
            logger.debug(
                "increment_demand failed for %s/%s", tool_name, window_key,
            )

    # -- Anchor operations ---------------------------------------------------

    async def fetch_all_balances(self) -> list[tuple[str, str]]:
        """Fetch all (npub, ledger_json) pairs, sorted by npub.

        Used by the OTS anchoring system to build a Merkle tree of all
        ledger balances.
        """
        result = await self._execute(
            f"SELECT npub, ledger_json FROM {self._t('balances')} ORDER BY npub"
        )
        rows = result.get("rows", [])
        return [(row["npub"], row["ledger_json"]) for row in rows]

    async def store_anchor(
        self,
        root_hash: str,
        leaf_count: int,
        status: str,
        ots_receipts_json: str | None,
        snapshot_json: str,
        leaf_hashes_json: str,
        created_at: str,
    ) -> str:
        """Store an anchor record. Returns the anchor ID as a string."""
        result = await self._execute(
            f"INSERT INTO {self._t('anchors')} "
            "(root_hash, leaf_count, status, ots_receipts_json, "
            " snapshot_json, leaf_hashes_json, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz) "
            "RETURNING id",
            [root_hash, leaf_count, status, ots_receipts_json,
             snapshot_json, leaf_hashes_json, created_at],
        )
        rows = result.get("rows", [])
        if rows:
            return str(rows[0]["id"])
        raise NeonQueryError("INSERT anchor returned no rows")

    async def fetch_anchor(self, anchor_id: str) -> dict[str, Any] | None:
        """Fetch a single anchor record by ID."""
        result = await self._execute(
            f"SELECT id, root_hash, leaf_count, status, ots_receipts_json, "
            f"snapshot_json, leaf_hashes_json, created_at, confirmed_at "
            f"FROM {self._t('anchors')} WHERE id = $1",
            [int(anchor_id)],
        )
        rows = result.get("rows", [])
        return rows[0] if rows else None

    async def list_anchors(
        self,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent anchor records, optionally filtered by status."""
        if status:
            result = await self._execute(
                f"SELECT id, root_hash, leaf_count, status, ots_receipts_json, "
                f"created_at, confirmed_at "
                f"FROM {self._t('anchors')} WHERE status = $1 "
                "ORDER BY created_at DESC LIMIT $2",
                [status, limit],
            )
        else:
            result = await self._execute(
                f"SELECT id, root_hash, leaf_count, status, ots_receipts_json, "
                f"created_at, confirmed_at "
                f"FROM {self._t('anchors')} ORDER BY created_at DESC LIMIT $1",
                [limit],
            )
        return result.get("rows", [])

    async def update_anchor_status(
        self,
        anchor_id: str,
        status: str,
        confirmed_at: str | None = None,
    ) -> None:
        """Update an anchor's status (e.g., 'submitted' → 'confirmed')."""
        if confirmed_at:
            await self._execute(
                f"UPDATE {self._t('anchors')} SET status = $1, confirmed_at = $2::timestamptz "
                "WHERE id = $3",
                [status, confirmed_at, int(anchor_id)],
            )
        else:
            await self._execute(
                f"UPDATE {self._t('anchors')} SET status = $1 WHERE id = $2",
                [status, int(anchor_id)],
            )

    async def update_anchor_receipts(
        self,
        anchor_id: str,
        ots_receipts_json: str,
    ) -> None:
        """Update an anchor's OTS receipts (e.g., after upgrade)."""
        await self._execute(
            f"UPDATE {self._t('anchors')} SET ots_receipts_json = $1 WHERE id = $2",
            [ots_receipts_json, int(anchor_id)],
        )

    # -- Authority configuration -----------------------------------------------

    async def get_config(self, key: str) -> str | None:
        """Read a value from the ``authority_config`` table.

        Returns ``None`` if the key does not exist.
        """
        try:
            result = await self._execute(
                f"SELECT value FROM {self._t('authority_config')} WHERE key = $1",
                [key],
            )
            rows = result.get("rows", [])
            return rows[0]["value"] if rows else None
        except Exception:
            return None

    async def set_config(self, key: str, value: str) -> None:
        """Upsert a value into the ``authority_config`` table."""
        await self._execute(
            f"INSERT INTO {self._t('authority_config')} (key, value, updated_at) "
            "VALUES ($1, $2, now()) "
            "ON CONFLICT (key) DO UPDATE SET value = $2, updated_at = now()",
            [key, value],
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


class NeonCredentialVault:
    """CredentialVaultBackend backed by Neon serverless Postgres.

    Implements the ``CredentialVaultBackend`` protocol for encrypted
    credential persistence.  Shares the httpx client and ``_execute()``
    helper from a ``NeonVault`` instance — no new connections or config.

    Schema: ``credentials`` table with composite PK ``(service, npub)``.
    Call ``ensure_schema()`` at startup alongside ``NeonVault.ensure_schema()``.
    """

    def __init__(self, *, neon_vault: NeonVault) -> None:
        self._neon = neon_vault

    def _t(self, table: str) -> str:
        """Schema-qualified table name, delegated to the underlying NeonVault."""
        return self._neon._t(table)

    async def ensure_schema(self) -> None:
        """Create the ``credentials`` and ``session_bindings`` tables if they don't exist."""
        await self._neon._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('credentials')} ("
            "    service TEXT NOT NULL,"
            "    npub TEXT NOT NULL,"
            "    encrypted_blob TEXT NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    PRIMARY KEY (service, npub)"
            ")"
        )
        await self.ensure_session_bindings_schema()

    async def store_credentials(
        self, service: str, npub: str, encrypted_blob: str,
    ) -> None:
        """Store an encrypted credential blob. Overwrites existing."""
        await self._neon._execute(
            f"INSERT INTO {self._t('credentials')} (service, npub, encrypted_blob, updated_at) "
            "VALUES ($1, $2, $3, now()) "
            "ON CONFLICT (service, npub) DO UPDATE "
            "SET encrypted_blob = EXCLUDED.encrypted_blob, "
            "    updated_at = now()",
            [service, npub, encrypted_blob],
        )

    async def fetch_credentials(
        self, service: str, npub: str,
    ) -> str | None:
        """Fetch an encrypted credential blob. Returns None if not found."""
        result = await self._neon._execute(
            f"SELECT encrypted_blob FROM {self._t('credentials')} "
            "WHERE service = $1 AND npub = $2",
            [service, npub],
        )
        rows = result.get("rows", [])
        return rows[0]["encrypted_blob"] if rows else None

    async def delete_credentials(
        self, service: str, npub: str,
    ) -> bool:
        """Delete stored credentials. Returns True if found and deleted."""
        result = await self._neon._execute(
            f"DELETE FROM {self._t('credentials')} WHERE service = $1 AND npub = $2",
            [service, npub],
        )
        return (result.get("rowCount", 0) or 0) > 0

    # -- SessionBindingBackend implementation --------------------------------

    async def ensure_session_bindings_schema(self) -> None:
        """Create the ``session_bindings`` table if it doesn't exist."""
        await self._neon._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('session_bindings')} ("
            "    caller_id TEXT NOT NULL,"
            "    service TEXT NOT NULL,"
            "    npub TEXT NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    PRIMARY KEY (caller_id, service)"
            ")"
        )

    async def store_session_binding(
        self, caller_id: str, service: str, npub: str,
    ) -> None:
        """Persist a session binding (upserts on conflict)."""
        await self._neon._execute(
            f"INSERT INTO {self._t('session_bindings')} (caller_id, service, npub, updated_at) "
            "VALUES ($1, $2, $3, now()) "
            "ON CONFLICT (caller_id, service) DO UPDATE "
            "SET npub = EXCLUDED.npub, "
            "    updated_at = now()",
            [caller_id, service, npub],
        )

    async def fetch_session_binding(
        self, caller_id: str, service: str,
    ) -> str | None:
        """Look up the npub for a caller+service pair."""
        result = await self._neon._execute(
            f"SELECT npub FROM {self._t('session_bindings')} "
            "WHERE caller_id = $1 AND service = $2",
            [caller_id, service],
        )
        rows = result.get("rows", [])
        return rows[0]["npub"] if rows else None

    async def delete_session_binding(
        self, caller_id: str, service: str,
    ) -> bool:
        """Remove a session binding. Returns True if found and deleted."""
        result = await self._neon._execute(
            f"DELETE FROM {self._t('session_bindings')} "
            "WHERE caller_id = $1 AND service = $2",
            [caller_id, service],
        )
        return (result.get("rowCount", 0) or 0) > 0
