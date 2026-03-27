"""Pricing resolver — runtime cost lookup with TTL cache and graceful fallback.

The resolver is the integration point MCP servers use to get dynamic tool
pricing.  It checks the active pricing model from Neon, caches it in
memory, and falls back to a static dict on cache miss or Neon failure.
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
    fallback_costs:
        Static ``{tool_name: cost}`` dict used when no active model
        exists or Neon is unreachable.
    cache_ttl:
        Time-to-live in seconds for the in-memory cache (default 300).
    """

    def __init__(
        self,
        *,
        store: Any,
        operator: str,
        fallback_costs: dict[str, int] | None = None,
        cache_ttl: float = 300.0,
    ) -> None:
        self._store = store
        self._operator = operator
        self._fallback_costs = fallback_costs or {}
        self._cache_ttl = cache_ttl

        self._cached_model: PricingModel | None = None
        self._cached_cost_map: dict[str, int] | None = None
        self._cached_engine: ConstraintEngine | None = None
        self._cache_ts: float = 0.0

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._cache_ts) > self._cache_ttl

    async def _ensure_fresh(self) -> None:
        """Refresh the cache if stale.  Neon failure degrades gracefully."""
        if not self._is_stale():
            return

        try:
            model = await self._store.fetch_active_model(self._operator)
            self._cached_model = model
            if model is not None:
                self._cached_cost_map = model.tool_cost_map()
                constraint_cfg = model.to_constraint_config()
                if constraint_cfg is not None:
                    self._cached_engine = load_constraints(constraint_cfg)
                else:
                    self._cached_engine = None
            else:
                self._cached_cost_map = None
                self._cached_engine = None
            self._cache_ts = time.monotonic()
        except Exception:
            logger.warning(
                "Failed to refresh pricing model for %s; using cached/fallback",
                self._operator,
                exc_info=True,
            )
            # Keep existing cache (stale > nothing)
            if self._cache_ts == 0.0:
                # First call ever failed — mark as attempted so we don't
                # hammer Neon on every request
                self._cache_ts = time.monotonic()

    async def get_cost(self, tool_name: str) -> int:
        """Return the cost for *tool_name*.

        Resolution order: active model → fallback static dict → 0.
        """
        await self._ensure_fresh()
        if self._cached_cost_map is not None and tool_name in self._cached_cost_map:
            return self._cached_cost_map[tool_name]
        return self._fallback_costs.get(tool_name, 0)

    async def get_tool_pricing(self, tool_name: str) -> "ToolPricing":
        """Return a ToolPricing for *tool_name*, supporting ad valorem pricing.

        Resolution order: active model's ToolPrice → fallback flat cost → free.
        """
        from tollbooth.pricing import ToolPricing

        await self._ensure_fresh()
        if self._cached_model is not None:
            for tp in self._cached_model.tools:
                if tp.tool_name == tool_name:
                    return tp.to_tool_pricing()
        flat = self._fallback_costs.get(tool_name, 0)
        return ToolPricing(fixed=flat)

    async def get_constraint_engine(self) -> ConstraintEngine | None:
        """Return the constraint engine from the active model's pipeline.

        Returns ``None`` if no active model or no pipeline defined.
        """
        await self._ensure_fresh()
        return self._cached_engine

    def refresh(self) -> None:
        """Force cache reset — call after Pricing Studio activates a model."""
        self._cache_ts = 0.0
