"""Tests for PricingResolver — cache TTL, has_tool, graceful failure."""

from __future__ import annotations

import time

import pytest

from tollbooth.tool_identity import capability_uuid
from tollbooth.pricing_model import PipelineStep, PricingModel, ToolPrice
from tollbooth.pricing_resolver import PricingResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEARCH_ID = capability_uuid("search")
CREATE_ID = capability_uuid("create")
CERTIFY_ID = capability_uuid("certify_credits")


def _active_model() -> PricingModel:
    return PricingModel(
        model_id="uuid-1",
        operator="npub1op",
        name="Active",
        is_active=True,
        tools=[
            ToolPrice(
                tool_id=SEARCH_ID,
                tool_name="search",
                price_sats=3,
                chain=[
                    PipelineStep(id="s1", type="free_trial", params={"first_n_free": 5}),
                ],
            ),
            ToolPrice(tool_id=CREATE_ID, tool_name="create", price_sats=7),
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
        resolver = PricingResolver(store=store, operator="npub1op")
        cost = await resolver.get_cost(SEARCH_ID)
        assert cost == 3

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_model(self) -> None:
        store = _MockStore(model=None)
        resolver = PricingResolver(store=store, operator="npub1op")
        cost = await resolver.get_cost(SEARCH_ID)
        assert cost == 0

    @pytest.mark.asyncio
    async def test_returns_zero_for_unknown_tool(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op")
        cost = await resolver.get_cost(capability_uuid("nonexistent"))
        assert cost == 0


# ---------------------------------------------------------------------------
# has_tool
# ---------------------------------------------------------------------------


class TestHasTool:
    @pytest.mark.asyncio
    async def test_returns_true_for_known_tool(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op")
        assert await resolver.has_tool(SEARCH_ID) is True

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_tool(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op")
        assert await resolver.has_tool(capability_uuid("nonexistent")) is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_model(self) -> None:
        store = _MockStore(model=None)
        resolver = PricingResolver(store=store, operator="npub1op")
        assert await resolver.has_tool(SEARCH_ID) is False


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------


class TestCache:
    @pytest.mark.asyncio
    async def test_caches_within_ttl(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op", cache_ttl=60.0)
        await resolver.get_cost(SEARCH_ID)
        await resolver.get_cost(SEARCH_ID)
        await resolver.get_cost(SEARCH_ID)
        assert store.call_count == 1

    @pytest.mark.asyncio
    async def test_refresh_resets_cache(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op", cache_ttl=60.0)
        await resolver.get_cost(SEARCH_ID)
        assert store.call_count == 1

        resolver.refresh()
        await resolver.get_cost(SEARCH_ID)
        assert store.call_count == 2

    @pytest.mark.asyncio
    async def test_stale_cache_triggers_refresh(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op", cache_ttl=0.01)
        await resolver.get_cost(SEARCH_ID)
        assert store.call_count == 1

        time.sleep(0.02)
        await resolver.get_cost(SEARCH_ID)
        assert store.call_count == 2


# ---------------------------------------------------------------------------
# Graceful Neon failure
# ---------------------------------------------------------------------------


class TestNeonUnavailable:
    @pytest.mark.asyncio
    async def test_neon_failure_clears_cache_and_marks_unavailable(self) -> None:
        """When Neon is unreachable, cache is cleared — no stale fallback."""
        store = _MockStore(fail=True)
        resolver = PricingResolver(store=store, operator="npub1op")
        cost = await resolver.get_cost(SEARCH_ID)
        assert cost == 0
        assert resolver.neon_available is False

    @pytest.mark.asyncio
    async def test_neon_failure_after_cache_clears_stale(self) -> None:
        """Neon going down after a successful fetch clears the cache."""
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op", cache_ttl=0.01)
        cost = await resolver.get_cost(SEARCH_ID)
        assert cost == 3
        assert resolver.neon_available is True

        store._fail = True
        time.sleep(0.02)

        cost = await resolver.get_cost(SEARCH_ID)
        assert cost == 0  # cache cleared, no stale fallback
        assert resolver.neon_available is False

    @pytest.mark.asyncio
    async def test_neon_recovery_restores_availability(self) -> None:
        """After Neon recovers, the resolver picks up the model again."""
        store = _MockStore(fail=True)
        resolver = PricingResolver(store=store, operator="npub1op", cache_ttl=0.01)
        await resolver.get_cost(SEARCH_ID)
        assert resolver.neon_available is False

        store._fail = False
        store._model = _active_model()
        time.sleep(0.02)

        cost = await resolver.get_cost(SEARCH_ID)
        assert cost == 3
        assert resolver.neon_available is True


# ---------------------------------------------------------------------------
# get_chain — per-tool constraint chain lookup
# ---------------------------------------------------------------------------


class TestGetChain:
    @pytest.mark.asyncio
    async def test_returns_chain_for_tool_with_chain(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op")
        chain = await resolver.get_chain(SEARCH_ID)
        assert len(chain) == 1
        assert chain[0].type == "free_trial"

    @pytest.mark.asyncio
    async def test_returns_empty_for_tool_without_chain(self) -> None:
        store = _MockStore(model=_active_model())
        resolver = PricingResolver(store=store, operator="npub1op")
        chain = await resolver.get_chain(CREATE_ID)
        assert chain == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_model(self) -> None:
        store = _MockStore(model=None)
        resolver = PricingResolver(store=store, operator="npub1op")
        chain = await resolver.get_chain(SEARCH_ID)
        assert chain == []


# ---------------------------------------------------------------------------
# get_tool_pricing
# ---------------------------------------------------------------------------


def _model_with_percent() -> PricingModel:
    return PricingModel(
        model_id="uuid-2",
        operator="npub1op",
        name="WithPercent",
        is_active=True,
        tools=[
            ToolPrice(tool_id=SEARCH_ID, tool_name="search", price_sats=3),
            ToolPrice(
                tool_id=CERTIFY_ID, tool_name="certify_credits", price_sats=2,
                price_type="percent", price_formula="amount_sats", min_cost=10,
            ),
        ],
    )


class TestGetToolPricing:
    @pytest.mark.asyncio
    async def test_flat_tool(self) -> None:
        store = _MockStore(model=_model_with_percent())
        resolver = PricingResolver(store=store, operator="npub1op")
        pricing = await resolver.get_tool_pricing(SEARCH_ID)
        assert pricing.fixed == 3
        assert pricing.compute() == 3

    @pytest.mark.asyncio
    async def test_percent_tool(self) -> None:
        store = _MockStore(model=_model_with_percent())
        resolver = PricingResolver(store=store, operator="npub1op")
        pricing = await resolver.get_tool_pricing(CERTIFY_ID)
        assert pricing.rate_percent == 2.0
        assert pricing.compute(amount_sats=1000) == 20
        assert pricing.compute(amount_sats=100) == 10  # min_cost

    @pytest.mark.asyncio
    async def test_unknown_tool_defaults_free(self) -> None:
        store = _MockStore(model=_model_with_percent())
        resolver = PricingResolver(store=store, operator="npub1op")
        pricing = await resolver.get_tool_pricing(capability_uuid("nonexistent"))
        assert pricing.compute() == 0
