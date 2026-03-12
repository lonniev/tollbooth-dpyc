"""Tests for PricingResolver — cache TTL, fallback, graceful failure."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from tollbooth.pricing_model import PipelineStep, PricingModel, ToolPrice
from tollbooth.pricing_resolver import PricingResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _active_model() -> PricingModel:
    return PricingModel(
        model_id="uuid-1",
        operator="npub1op",
        name="Active",
        is_active=True,
        tools=[
            ToolPrice(tool_name="search", price_sats=3),
            ToolPrice(tool_name="create", price_sats=7),
        ],
        pipeline=[
            PipelineStep(id="s1", type="free_trial", params={"first_n_free": 5}),
        ],
    )


class _MockStore:
    """Minimal mock that exposes fetch_active_model."""

    def __init__(self, model: PricingModel | None = None, *, fail: bool = False):
        self._model = model
        self._fail = fail
        self.call_count = 0

    async def fetch_active_model(self, operator: str) -> PricingModel | None:
        self.call_count += 1
        if self._fail:
            raise RuntimeError("Neon down")
        return self._model


# ---------------------------------------------------------------------------
# get_cost
# ---------------------------------------------------------------------------


class TestGetCost:
    @pytest.mark.asyncio
    async def test_returns_active_model_cost(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(
            store=store, operator="npub1op", fallback_costs={"search": 99},
        )
        cost = await resolver.get_cost("search")
        assert cost == 3  # from model, not fallback

    @pytest.mark.asyncio
    async def test_falls_back_to_static_dict(self) -> None:
        store = _MockStore(model=None)
        resolver = PricingResolver(
            store=store, operator="npub1op", fallback_costs={"search": 99},
        )
        cost = await resolver.get_cost("search")
        assert cost == 99

    @pytest.mark.asyncio
    async def test_falls_back_to_zero(self) -> None:
        store = _MockStore(model=None)
        resolver = PricingResolver(store=store, operator="npub1op")
        cost = await resolver.get_cost("unknown_tool")
        assert cost == 0

    @pytest.mark.asyncio
    async def test_unknown_tool_with_active_model(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(
            store=store, operator="npub1op", fallback_costs={"other": 42},
        )
        cost = await resolver.get_cost("other")
        assert cost == 42  # not in model, but in fallback


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCache:
    @pytest.mark.asyncio
    async def test_caches_within_ttl(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(
            store=store, operator="npub1op", cache_ttl=60.0,
        )
        await resolver.get_cost("search")
        await resolver.get_cost("search")
        await resolver.get_cost("search")
        assert store.call_count == 1  # only one Neon call

    @pytest.mark.asyncio
    async def test_refresh_resets_cache(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(
            store=store, operator="npub1op", cache_ttl=60.0,
        )
        await resolver.get_cost("search")
        assert store.call_count == 1

        resolver.refresh()
        await resolver.get_cost("search")
        assert store.call_count == 2

    @pytest.mark.asyncio
    async def test_stale_cache_triggers_refresh(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(
            store=store, operator="npub1op", cache_ttl=0.01,
        )
        await resolver.get_cost("search")
        assert store.call_count == 1

        # Wait for cache to expire
        time.sleep(0.02)
        await resolver.get_cost("search")
        assert store.call_count == 2


# ---------------------------------------------------------------------------
# Graceful Neon failure
# ---------------------------------------------------------------------------


class TestGracefulFailure:
    @pytest.mark.asyncio
    async def test_neon_failure_uses_fallback(self) -> None:
        store = _MockStore(fail=True)
        resolver = PricingResolver(
            store=store, operator="npub1op", fallback_costs={"search": 50},
        )
        cost = await resolver.get_cost("search")
        assert cost == 50  # fallback

    @pytest.mark.asyncio
    async def test_neon_failure_after_cache_keeps_stale(self) -> None:
        """If Neon was reachable before but fails now, use stale cache."""
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(
            store=store, operator="npub1op", cache_ttl=0.01,
        )
        # First call succeeds and caches
        cost = await resolver.get_cost("search")
        assert cost == 3

        # Make store fail and expire cache
        store._fail = True
        time.sleep(0.02)

        # Should use stale cache
        cost = await resolver.get_cost("search")
        assert cost == 3


# ---------------------------------------------------------------------------
# get_constraint_engine
# ---------------------------------------------------------------------------


class TestGetConstraintEngine:
    @pytest.mark.asyncio
    async def test_returns_engine_from_pipeline(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op")
        engine = await resolver.get_constraint_engine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_pipeline(self) -> None:
        model = PricingModel(
            model_id="uuid-2", operator="npub1op", name="NoPipeline",
            is_active=True, tools=[], pipeline=[],
        )
        store = _MockStore(model=model)
        resolver = PricingResolver(store=store, operator="npub1op")
        engine = await resolver.get_constraint_engine()
        assert engine is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_model(self) -> None:
        store = _MockStore(model=None)
        resolver = PricingResolver(store=store, operator="npub1op")
        engine = await resolver.get_constraint_engine()
        assert engine is None
