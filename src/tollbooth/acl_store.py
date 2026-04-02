"""Tool ACL store — CRUD for signed ACL events in Neon Postgres.

Follows the PricingModelStore composition pattern: wraps a NeonVault
and shares its ``_execute()`` helper.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolAclStore:
    """CRUD for ``tool_acls`` table via Neon HTTP API.

    Usage::

        store = ToolAclStore(neon_vault=vault)
        await store.ensure_schema()

        await store.store_acl("npub1op...", "set_pricing_model", signed_event_json)
        acl = await store.fetch_acl("npub1op...", "set_pricing_model")
    """

    def __init__(self, *, neon_vault: Any) -> None:
        self._neon = neon_vault

    def _t(self, table: str) -> str:
        """Schema-qualified table name via the underlying NeonVault."""
        return self._neon._t(table)

    async def ensure_schema(self) -> None:
        """Create the ``tool_acls`` table and indexes."""
        await self._neon._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('tool_acls')} ("
            "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "    operator TEXT NOT NULL,"
            "    tool_pattern TEXT NOT NULL,"
            "    signed_event_json JSONB NOT NULL,"
            "    created_at TIMESTAMPTZ DEFAULT now(),"
            "    updated_at TIMESTAMPTZ DEFAULT now(),"
            "    UNIQUE (operator, tool_pattern)"
            ")"
        )
        await self._neon._execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_acls_operator "
            f"ON {self._t('tool_acls')} (operator)"
        )

    async def store_acl(
        self,
        operator: str,
        tool_pattern: str,
        signed_event_json: str,
    ) -> str:
        """Store or update a signed ACL event. Returns the row ID."""
        result = await self._neon._execute(
            f"INSERT INTO {self._t('tool_acls')} (operator, tool_pattern, signed_event_json) "
            "VALUES ($1, $2, $3::jsonb) "
            "ON CONFLICT (operator, tool_pattern) DO UPDATE "
            "SET signed_event_json = EXCLUDED.signed_event_json, "
            "    updated_at = now() "
            "RETURNING id",
            [operator, tool_pattern, signed_event_json],
        )
        rows = result.get("rows", [])
        if not rows:
            raise RuntimeError("UPSERT tool_acls returned no rows")
        return str(rows[0]["id"])

    async def fetch_acl(
        self,
        operator: str,
        tool_pattern: str,
    ) -> str | None:
        """Fetch the signed ACL event JSON for exact tool_pattern match.

        Returns the signed_event_json string, or ``None`` if not found.
        """
        result = await self._neon._execute(
            f"SELECT signed_event_json FROM {self._t('tool_acls')} "
            "WHERE operator = $1 AND tool_pattern = $2",
            [operator, tool_pattern],
        )
        rows = result.get("rows", [])
        if not rows:
            return None
        raw = rows[0]["signed_event_json"]
        # Neon may return JSONB as a dict or as a string
        if isinstance(raw, dict):
            import json
            return json.dumps(raw)
        return raw

    async def fetch_all_acls(self, operator: str) -> list[dict]:
        """List all ACLs for an operator.

        Returns ``[{tool_pattern, signed_event_json, updated_at}]``.
        """
        result = await self._neon._execute(
            f"SELECT tool_pattern, signed_event_json, updated_at "
            f"FROM {self._t('tool_acls')} WHERE operator = $1 "
            "ORDER BY tool_pattern",
            [operator],
        )
        return result.get("rows", [])

    async def delete_acl(self, operator: str, tool_pattern: str) -> bool:
        """Delete an ACL. Returns ``True`` if found and deleted."""
        result = await self._neon._execute(
            f"DELETE FROM {self._t('tool_acls')} "
            "WHERE operator = $1 AND tool_pattern = $2",
            [operator, tool_pattern],
        )
        return (result.get("rowCount", 0) or 0) > 0
