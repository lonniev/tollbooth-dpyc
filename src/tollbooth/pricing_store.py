"""Pricing model store — CRUD for operator pricing models in Neon Postgres.

Follows the NeonCredentialVault composition pattern: wraps a NeonVault
and shares its ``_execute()`` helper.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tollbooth.pricing_model import PricingModel

logger = logging.getLogger(__name__)


class PricingModelStore:
    """CRUD for ``operator_pricing_models`` table via Neon HTTP API.

    Usage::

        store = PricingModelStore(neon_vault=vault)
        await store.ensure_schema()

        model = PricingModel(operator="npub1abc", name="Launch Pricing", ...)
        model_id = await store.create_model(model)
        await store.activate_model(model_id, operator="npub1abc")
    """

    def __init__(self, *, neon_vault: Any) -> None:
        self._neon = neon_vault

    async def ensure_schema(self) -> None:
        """Create the ``operator_pricing_models`` table and indexes."""
        await self._neon._execute(
            "CREATE TABLE IF NOT EXISTS operator_pricing_models ("
            "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "    operator TEXT NOT NULL,"
            "    name TEXT NOT NULL,"
            "    model_json JSONB NOT NULL,"
            "    is_active BOOLEAN DEFAULT false,"
            "    created_at TIMESTAMPTZ DEFAULT now(),"
            "    updated_at TIMESTAMPTZ DEFAULT now()"
            ")"
        )
        await self._neon._execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_operator "
            "ON operator_pricing_models (operator) WHERE is_active = true"
        )
        await self._neon._execute(
            "CREATE INDEX IF NOT EXISTS idx_pricing_models_operator "
            "ON operator_pricing_models (operator)"
        )

    async def fetch_active_model(self, operator: str) -> PricingModel | None:
        """Return the active pricing model for an operator, or ``None``."""
        result = await self._neon._execute(
            "SELECT id, operator, name, model_json, is_active "
            "FROM operator_pricing_models "
            "WHERE operator = $1 AND is_active = true "
            "LIMIT 1",
            [operator],
        )
        rows = result.get("rows", [])
        if not rows:
            return None
        return PricingModel.from_row(rows[0])

    async def list_models(self, operator: str) -> list[PricingModel]:
        """List all pricing models for an operator (newest first)."""
        result = await self._neon._execute(
            "SELECT id, operator, name, model_json, is_active "
            "FROM operator_pricing_models "
            "WHERE operator = $1 "
            "ORDER BY created_at DESC",
            [operator],
        )
        rows = result.get("rows", [])
        return [PricingModel.from_row(row) for row in rows]

    async def create_model(self, model: PricingModel) -> str:
        """Insert a new pricing model. Returns the UUID as a string."""
        result = await self._neon._execute(
            "INSERT INTO operator_pricing_models "
            "(operator, name, model_json, is_active) "
            "VALUES ($1, $2, $3::jsonb, false) "
            "RETURNING id",
            [model.operator, model.name, model.to_json()],
        )
        rows = result.get("rows", [])
        if not rows:
            raise RuntimeError("INSERT pricing model returned no rows")
        return str(rows[0]["id"])

    async def update_model(self, model_id: str, model_json: str) -> None:
        """Replace the ``model_json`` blob for an existing model."""
        await self._neon._execute(
            "UPDATE operator_pricing_models "
            "SET model_json = $1::jsonb, updated_at = now() "
            "WHERE id = $2::uuid",
            [model_json, model_id],
        )

    async def activate_model(self, model_id: str, operator: str) -> None:
        """Deactivate the current active model and activate the given one.

        Two UPDATEs: deactivate all for operator, then activate the target.
        """
        await self._neon._execute(
            "UPDATE operator_pricing_models "
            "SET is_active = false, updated_at = now() "
            "WHERE operator = $1 AND is_active = true",
            [operator],
        )
        await self._neon._execute(
            "UPDATE operator_pricing_models "
            "SET is_active = true, updated_at = now() "
            "WHERE id = $1::uuid AND operator = $2",
            [model_id, operator],
        )

    async def delete_model(self, model_id: str) -> bool:
        """Delete a pricing model. Refuses to delete an active model.

        Returns ``True`` if deleted, ``False`` if not found or active.
        """
        # Check if active first
        check = await self._neon._execute(
            "SELECT is_active FROM operator_pricing_models WHERE id = $1::uuid",
            [model_id],
        )
        rows = check.get("rows", [])
        if not rows:
            return False
        if rows[0].get("is_active"):
            logger.warning("Refusing to delete active pricing model %s", model_id)
            return False

        result = await self._neon._execute(
            "DELETE FROM operator_pricing_models WHERE id = $1::uuid",
            [model_id],
        )
        return (result.get("rowCount", 0) or 0) > 0
