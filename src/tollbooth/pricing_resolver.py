"""Pricing resolver — runtime cost lookup with TTL cache.

The resolver is the integration point MCP servers use to get dynamic tool
pricing.  It checks the active pricing model from Neon, caches it in
memory, and returns 0 on cache miss or Neon failure.

All lookups are by ``tool_id`` (UUID), not tool name.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tollbooth.pricing import ToolPricing

from tollbooth.pricing_model import PipelineStep, PricingModel

logger = logging.getLogger(__name__)

# Postgres SQLSTATE classes that will never resolve by retrying — the
# operator's database needs repair, not patience. Class 28 = invalid
# authorization, class 3D = invalid catalog, class 42 = syntax errors and
# access-rule violations (42501 permission denied, 42P01 undefined table).
_PERMANENT_SQLSTATE_CLASSES = ("28", "3D", "42")


def _is_permanent_sql_error(exc: Exception) -> bool:
    """True when *exc* is a NeonQueryError carrying a permanent SQLSTATE."""
    code = getattr(exc, "code", "")
    return bool(code) and code[:2] in _PERMANENT_SQLSTATE_CLASSES


def _is_quota_error(exc: Exception) -> bool:
    """True when *exc* is the persistence provider (Neon) refusing with HTTP
    402 — the database has exhausted its compute/storage quota. Non-transient
    and distinct from a SQL fault: the books are locked for billing, not code."""
    return getattr(exc, "status", 0) == 402


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
        self._neon_available: bool = False
        self._last_error: Exception | None = None
        # Initialize to negative infinity so the first _is_stale() always
        # returns True — even if time.monotonic() is small (fresh CI runner).
        self._cache_ts: float = -self._cache_ttl - 1.0

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._cache_ts) > self._cache_ttl

    @property
    def neon_available(self) -> bool:
        """True if we have successfully reached Neon at least once."""
        return self._neon_available

    @property
    def last_error_permanent(self) -> bool:
        """True when the most recent load failure carries a permanent SQLSTATE.

        Permanent means retrying cannot succeed (permission denied, missing
        relation, auth failure) — the operator must repair the database.
        Meaningful only while ``neon_available`` is False.
        """
        return self._last_error is not None and _is_permanent_sql_error(self._last_error)

    @property
    def last_error_quota(self) -> bool:
        """True when the most recent load failure was Neon HTTP 402 — the
        database's compute/storage quota is exhausted (the books are locked
        for billing). Non-transient; the Authority that provisions this
        database must upgrade the plan or wait for the quota to reset.
        Meaningful only while ``neon_available`` is False.
        """
        return self._last_error is not None and _is_quota_error(self._last_error)

    @property
    def last_error_summary(self) -> str:
        """One-line summary of the most recent load failure, '' if none.

        Safe to surface to callers: SQL error messages from Neon name the
        table and SQLSTATE but never carry credentials.
        """
        if self._last_error is None:
            return ""
        return f"{type(self._last_error).__name__}: {self._last_error}"

    _HYDRATION_RETRIES = 3
    _HYDRATION_BACKOFF = 2.0  # seconds between retries

    async def _ensure_fresh(self) -> None:
        """Refresh the cache if stale.

        On cold start, retries up to 3 times with backoff to absorb
        the Neon connection warm-up delay rather than rejecting the
        caller's first request.
        """
        if not self._is_stale():
            return

        import asyncio
        last_exc: Exception | None = None

        for attempt in range(self._HYDRATION_RETRIES):
            try:
                model = await self._store.fetch_active_model(self._operator)
                self._neon_available = True
                self._last_error = None
                self._cached_model = model
                if model is not None:
                    self._cached_cost_map = model.tool_cost_map()
                    self._cached_tool_ids = model.tool_id_set()
                    self._cached_priced_map = model.tool_priced_map()
                else:
                    self._cached_cost_map = None
                    self._cached_tool_ids = None
                    self._cached_priced_map = None
                self._cache_ts = time.monotonic()
                if attempt > 0:
                    logger.info(
                        "Pricing model loaded for %s on attempt %d",
                        self._operator, attempt + 1,
                    )
                return
            except Exception as exc:
                last_exc = exc
                if _is_permanent_sql_error(exc) or _is_quota_error(exc):
                    # Permission denied / missing relation, or a provider 402
                    # (quota exhausted) — retrying with backoff cannot help;
                    # record and report immediately rather than burning the
                    # cold-start budget on a condition only a human can clear.
                    break
                if attempt < self._HYDRATION_RETRIES - 1:
                    logger.debug(
                        "Pricing model load attempt %d failed for %s (%s) — retrying in %.0fs",
                        attempt + 1, self._operator, exc, self._HYDRATION_BACKOFF,
                    )
                    await asyncio.sleep(self._HYDRATION_BACKOFF)

        logger.warning(
            "Pricing model load failed for %s (%s, %s: %s)",
            self._operator,
            "permanent SQL error" if (last_exc is not None and _is_permanent_sql_error(last_exc)) else f"{self._HYDRATION_RETRIES} attempts",
            type(last_exc).__name__, last_exc,
        )
        self._last_error = last_exc
        self._neon_available = False
        self._cached_model = None
        self._cached_cost_map = None
        self._cached_tool_ids = None
        self._cached_priced_map = None
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

    async def get_chain(self, tool_id: str) -> list[PipelineStep]:
        """Return the ordered constraint chain for *tool_id*.

        Empty list if the tool has no chain (or no entry in the model).
        Returns raw ``PipelineStep`` objects — the caller is responsible
        for instantiating ``ToolConstraint`` instances from them.
        """
        await self._ensure_fresh()
        if self._cached_model is None:
            return []
        return self._cached_model.chain_for(tool_id)

    def refresh(self) -> None:
        """Force cache reset — call after Pricing Studio activates a model."""
        self._cache_ts = -self._cache_ttl - 1.0
