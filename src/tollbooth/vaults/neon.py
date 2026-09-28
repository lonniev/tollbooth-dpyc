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

from tollbooth.vault_backend import LedgerVersionConflict

logger = logging.getLogger(__name__)


class NeonQueryError(Exception):
    """Raised when a Neon SQL query returns an error.

    Carries the Postgres SQLSTATE in ``code`` when Neon supplied one
    (e.g. ``42501`` permission denied, ``42P01`` undefined table), so
    callers can distinguish permanent misconfiguration from transient
    connectivity trouble. ``code`` is ``""`` when Neon gave no SQLSTATE.

    ``status`` carries the HTTP status of the Neon REST gateway response
    when the failure was an HTTP-level one (``0`` for SQL errors returned
    in a 200 body). This is how the persistence *provider*'s own signals —
    notably ``402`` when the project has exhausted its compute/storage
    quota — reach the classifier, since a billing 402 carries no SQLSTATE.
    """

    def __init__(self, message: str, code: str = "", status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


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
        except Exception as exc:  # noqa: BLE001
            logger.debug("Schema prefix parsing skipped (non-fatal): %s", exc)
        self._client = httpx.AsyncClient(
            headers={
                "Neon-Connection-String": database_url,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        self._endpoint_host = hostname

        # Field encryption — if nsec provided, all stored values are AES-256-GCM encrypted.
        # Without nsec, vault operates in plaintext mode. Every production caller
        # supplies a key (operator via Authority bootstrap, self-provisioning
        # actor via its own nsec); a keyless vault means financial/PII data would
        # land in a plaintext column, so make that choice loud rather than silent.
        self._cipher = None
        if encryption_nsec_hex:
            from tollbooth.vault_encryption import VaultCipher
            self._cipher = VaultCipher(nsec_hex=encryption_nsec_hex)
        else:
            logger.warning(
                "NeonVault constructed WITHOUT encryption_nsec_hex — stored "
                "values (including balances) will be PLAINTEXT. This is only "
                "acceptable for tests; production callers must pass a key."
            )

    @property
    def endpoint_host(self) -> str:
        """The Neon hostname this vault connects to — e.g.
        ``ep-cool-name-a1b2c3d4-pooler.us-east-2.aws.neon.tech``. Used to
        identify which control-plane project is THIS Authority's own, so the
        proactive watch can show just that project instead of every project the
        org key can see.
        """
        return self._endpoint_host

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt if cipher is configured, otherwise passthrough."""
        return self._cipher.encrypt(plaintext) if self._cipher else plaintext

    def _decrypt(self, value: str) -> str:
        """Decrypt if cipher is configured. Bridges legacy plaintext rows.

        A configured cipher that reads a value which isn't ciphertext means a
        pre-encryption (plaintext) row — it's returned as-is so the read
        succeeds and gets re-encrypted on its next write. This bridge is only
        safe while such rows still exist; the warning makes the shrinking
        legacy population observable so the fallthrough can be removed once it
        stops firing. Not a silent downgrade.
        """
        if not self._cipher:
            return value
        if self._cipher.is_encrypted(value):
            return self._cipher.decrypt(value)
        logger.warning(
            "NeonVault read a PLAINTEXT value under an encrypting cipher — "
            "legacy pre-encryption row; it will migrate to ciphertext on next "
            "write. Investigate if this persists after a full write cycle."
        )
        return value  # Legacy plaintext — return as-is (migrates on next write)

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
            except Exception:  # noqa: BLE001
                err_body = None
            if isinstance(err_body, dict) and err_body.get("message"):
                raise NeonQueryError(
                    f"Neon HTTP {resp.status_code}: {err_body['message']} "
                    f"(query={query[:120]}…)",
                    code=str(err_body.get("code") or ""),
                    status=resp.status_code,
                )
            # Body wasn't JSON or didn't have a message — fall through to
            # raise_for_status so callers still see the HTTP error.
            resp.raise_for_status()

        data = resp.json()

        # Neon returns SQL errors in the response body with a "message" field
        if isinstance(data, dict) and "message" in data and "rows" not in data:
            raise NeonQueryError(data["message"], code=str(data.get("code") or ""))

        return data

    async def _execute_batch(self, queries: list[str]) -> list[dict[str, Any]]:
        """Run several statements in ONE request and ONE transaction.

        Neon's HTTP endpoint takes ``{"queries": [{"query", "params"}, …]}``
        and answers ``{"results": […]}`` (the batch the serverless driver's
        ``transaction()`` sends). Used for schema preparation, where one
        concern's tables land together or not at all.
        """
        body = {"queries": [{"query": q, "params": []} for q in queries]}
        resp = await self._client.post(self._endpoint, json=body)
        if resp.status_code >= 400:
            try:
                err_body = resp.json()
            except Exception:  # noqa: BLE001
                err_body = None
            if isinstance(err_body, dict) and err_body.get("message"):
                raise NeonQueryError(
                    f"Neon HTTP {resp.status_code}: {err_body['message']} "
                    f"(batch of {len(queries)}, first={queries[0][:80] if queries else ''}…)",
                    code=str(err_body.get("code") or ""),
                    status=resp.status_code,
                )
            resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "message" in data and "results" not in data:
            raise NeonQueryError(data["message"], code=str(data.get("code") or ""))
        results = data.get("results", []) if isinstance(data, dict) else []
        return list(results)

    # -- VaultBackend protocol -----------------------------------------------

    async def store_ledger(
        self, user_id: str, ledger_json: str, expected_version: int | None,
    ) -> int:
        """Write ``ledger_json`` only if ``balances`` still holds ``expected_version``.

        ``None`` means the writer read no row: insert one, and treat "a row
        appeared meanwhile" as a conflict. Otherwise a guarded UPDATE lands only
        when the version the writer read is the version on the row; anything
        else raises ``LedgerVersionConflict`` and the caller re-reads. The
        version is the one that came back with the snapshot — this class keeps
        no memory of versions, so a writer cannot borrow a fresher one.

        Returns the new version.
        """
        ledger_json = self._encrypt(ledger_json)

        if expected_version is None:
            result = await self._execute(
                f"INSERT INTO {self._t('balances')}(npub, ledger_json, version, last_flush, created_at) "
                "VALUES ($1, $2, 1, now(), now()) "
                "ON CONFLICT (npub) DO NOTHING "
                "RETURNING version",
                [user_id, ledger_json],
            )
            rows = result.get("rows", [])
            if rows:
                return int(rows[0]["version"])
            raise LedgerVersionConflict(
                f"ledger for {user_id[:20]} appeared before this insert — refetch required"
            )

        result = await self._execute(
            f"UPDATE {self._t('balances')} "
            "SET ledger_json = $1, version = version + 1, last_flush = now() "
            "WHERE npub = $2 AND version = $3 "
            "RETURNING version",
            [ledger_json, user_id, expected_version],
        )
        rows = result.get("rows", [])
        if rows:
            return int(rows[0]["version"])
        logger.info(
            "Ledger CAS conflict for %s (wrote at v%d) — caller must re-fetch.",
            user_id[:20], expected_version,
        )
        raise LedgerVersionConflict(
            f"ledger version conflict for {user_id[:20]} (had v{expected_version})"
        )

    async def fetch_ledger(self, user_id: str) -> tuple[str, int] | None:
        """The current ledger JSON and the version it was read at, or ``None``."""
        result = await self._execute(
            f"SELECT ledger_json, version FROM {self._t('balances')} WHERE npub = $1",
            [user_id],
        )
        rows = result.get("rows", [])
        if not rows:
            return None
        return self._decrypt(rows[0]["ledger_json"]), int(rows[0]["version"])

    async def snapshot_ledger(
        self, user_id: str, ledger_json: str, timestamp: str,
    ) -> str | None:
        """Append a timestamped copy to the ``transactions`` journal.

        Never touches the live ``balances`` row — that is ``store_ledger``'s
        job, under its version guard. Returns the journal row id, or ``None``
        if the insert fails.
        """
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

    async def ensure_schema(self) -> list[dict[str, Any]]:
        """Make sure this database is prepared for the SDK's schema version.

        One read of the ``tollbooth_schema`` breadcrumb when it already is (the
        usual cold start); otherwise each concern's tables in one compound
        request apiece, then the crumb. See ``tollbooth.vaults.schema``.
        """
        from tollbooth.vaults.schema import prepare_schema

        return await prepare_schema(self)

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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
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

    Schema: ``credentials`` table with composite PK ``(service, npub)`` and
    ``session_bindings``; prepared by ``tollbooth.vaults.schema`` with every
    other concern (``credential_schema_statements`` below).
    """

    def __init__(self, *, neon_vault: NeonVault) -> None:
        self._neon = neon_vault

    def _t(self, table: str) -> str:
        """Schema-qualified table name, delegated to the underlying NeonVault."""
        return self._neon._t(table)

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


# ---------------------------------------------------------------------------
# DDL owned by this module — assembled by ``tollbooth.vaults.schema``.
# Changing any statement here means bumping ``schema.SCHEMA_VERSION``.
# ---------------------------------------------------------------------------


def ledger_schema_statements(t: Any, idx: str) -> list[str]:
    """Balances, the transaction journal, demand counters and actor config."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t('balances')} ("
            "    npub TEXT PRIMARY KEY,"
            "    ledger_json TEXT NOT NULL,"
            "    version INTEGER NOT NULL DEFAULT 1,"
            "    last_flush TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        ),
        (
            f"CREATE TABLE IF NOT EXISTS {t('transactions')} ("
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
        ),
        f"CREATE INDEX IF NOT EXISTS {idx}idx_transactions_npub ON {t('transactions')}(npub)",
        f"CREATE INDEX IF NOT EXISTS {idx}idx_transactions_created ON {t('transactions')}(created_at)",
        # Global demand counters (surge pricing).
        (
            f"CREATE TABLE IF NOT EXISTS {t('tool_demand')} ("
            "    tool_name TEXT NOT NULL,"
            "    window_key TEXT NOT NULL,"
            "    count INTEGER NOT NULL DEFAULT 0,"
            "    PRIMARY KEY (tool_name, window_key)"
            ")"
        ),
        # Actor configuration (curator npub, proven-npub cache, onboarding state).
        (
            f"CREATE TABLE IF NOT EXISTS {t('authority_config')} ("
            "    key TEXT PRIMARY KEY,"
            "    value TEXT NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        ),
    ]


def notarization_schema_statements(t: Any, idx: str) -> list[str]:
    """OTS Bitcoin anchors of the ledger's Merkle root."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t('anchors')} ("
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
        ),
        f"CREATE INDEX IF NOT EXISTS {idx}idx_anchors_created ON {t('anchors')}(created_at)",
        f"CREATE INDEX IF NOT EXISTS {idx}idx_anchors_status ON {t('anchors')}(status)",
    ]


def credential_schema_statements(t: Any, idx: str) -> list[str]:
    """Secure Courier credentials and session bindings."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t('credentials')} ("
            "    service TEXT NOT NULL,"
            "    npub TEXT NOT NULL,"
            "    encrypted_blob TEXT NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    PRIMARY KEY (service, npub)"
            ")"
        ),
        (
            f"CREATE TABLE IF NOT EXISTS {t('session_bindings')} ("
            "    caller_id TEXT NOT NULL,"
            "    service TEXT NOT NULL,"
            "    npub TEXT NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    PRIMARY KEY (caller_id, service)"
            ")"
        ),
    ]
