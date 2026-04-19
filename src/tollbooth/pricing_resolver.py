"""Pricing resolver — runtime cost lookup with TTL cache.

The resolver is the integration point MCP servers use to get dynamic tool
pricing.  It checks the active pricing model from Neon, caches it in
memory, and returns 0 on cache miss or Neon failure.

All lookups are by ``tool_id`` (UUID), not tool name.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from tollbooth.constraints.config import load_constraints
from tollbooth.constraints.engine import ConstraintEngine
from tollbooth.pricing_model import PricingModel

logger = logging.getLogger(__name__)


class PricingResolver:
    """Runtime resolver for tool costs and constraint engines.

    Parameters
    ----------
    store:
        A ``PricingModelStore`` instance (or anything with
        ``fetch_active_model(operator) -> PricingModel | None``).
    operator:
        The operator identifier (npub) whose active model to resolve.
    cache_ttl:
        Time-to-live in seconds for the in-memory cache (default 300).
    """

    def __init__(
        self,
        *,
        store: Any,
        operator: str,
        cache_ttl: float = 300.0,
    ) -> None:
        self._store = store
        self._operator = operator
        self._cache_ttl = cache_ttl

        self._cached_model: PricingModel | None = None
        self._cached_cost_map: dict[str, int] | None = None
        self._cached_tool_ids: set[str] | None = None
        self._cached_priced_map: dict[str, bool] | None = None
        self._cached_engine: ConstraintEngine | None = None
        self._neon_available: bool = False
        # Initialize to negative infinity so the first _is_stale() always
        # returns True — even if time.monotonic() is small (fresh CI runner).
        self._cache_ts: float = -self._cache_ttl - 1.0

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._cache_ts) > self._cache_ttl

    @property
    def neon_available(self) -> bool:
        """True if we have successfully reached Neon at least once."""
        return self._neon_available

    async def _ensure_fresh(self) -> None:
        """Refresh the cache if stale.  Neon failure denies all paid tools."""
        if not self._is_stale():
            return

        try:
            model = await self._store.fetch_active_model(self._operator)
            self._neon_available = True
            self._cached_model = model
            if model is not None:
                self._cached_cost_map = model.tool_cost_map()
                self._cached_tool_ids = model.tool_id_set()
                self._cached_priced_map = model.tool_priced_map()
                constraint_cfg = model.to_constraint_config()
                if constraint_cfg is not None:
                    self._cached_engine = load_constraints(constraint_cfg)
                else:
                    self._cached_engine = None
            else:
                self._cached_cost_map = None
                self._cached_tool_ids = None
                self._cached_priced_map = None
                self._cached_engine = None
            self._cache_ts = time.monotonic()
        except Exception as exc:
            logger.warning(
                "Pricing model load failed for %s (%s: %s) — paid tools denied until retry",
                self._operator, type(exc).__name__, exc,
            )
            self._neon_available = False
            self._cached_model = None
            self._cached_cost_map = None
            self._cached_tool_ids = None
            self._cached_priced_map = None
            self._cached_engine = None
            # Retry on next call — don't cache the failure
            self._cache_ts = -self._cache_ttl - 1.0

    async def get_cost(self, tool_id: str) -> int:
        """Return the cost for *tool_id*.

        Returns 0 if the tool is not in the active model.
        """
        await self._ensure_fresh()
        if self._cached_cost_map is not None and tool_id in self._cached_cost_map:
            return self._cached_cost_map[tool_id]
        return 0

    async def is_priced(self, tool_id: str) -> bool:
        """Return True if the operator has explicitly set a price for this tool.

        False means TBD — the tool exists in the model but hasn't been priced yet.
        """
        await self._ensure_fresh()
        if self._cached_priced_map is not None:
            return self._cached_priced_map.get(tool_id, False)
        return False

    async def has_tool(self, tool_id: str) -> bool:
        """Return True if *tool_id* has an explicit entry in the pricing model."""
        await self._ensure_fresh()
        if self._cached_tool_ids is not None:
            return tool_id in self._cached_tool_ids
        return False

    async def get_tool_pricing(self, tool_id: str) -> "ToolPricing":
        """Return a ToolPricing for *tool_id*, supporting ad valorem pricing.

        Returns a free ToolPricing if the tool is not in the active model.
        """
        from tollbooth.pricing import ToolPricing

        await self._ensure_fresh()
        if self._cached_model is not None:
            for tp in self._cached_model.tools:
                if tp.tool_id == tool_id:
                    return tp.to_tool_pricing()
        return ToolPricing(fixed=0)

    async def get_constraint_engine(self) -> ConstraintEngine | None:
        """Return the constraint engine from the active model's pipeline.

        Returns ``None`` if no active model or no pipeline defined.
        """
        await self._ensure_fresh()
        return self._cached_engine

    def refresh(self) -> None:
        """Force cache reset — call after Pricing Studio activates a model."""
        self._cache_ts = -self._cache_ttl - 1.0
