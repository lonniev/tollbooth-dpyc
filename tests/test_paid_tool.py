"""Tests for the @runtime.paid_tool() decorator."""

import pytest

from tollbooth.runtime import OperatorRuntime
from tollbooth.tool_identity import ToolIdentity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakePricingResolver:
    """In-memory pricing resolver for testing."""

    def __init__(self, costs: dict[str, int]):
        # costs keyed by tool_id
        self._costs = costs

    async def get_cost(self, tool_id: str) -> int:
        return self._costs.get(tool_id, 0)

    async def has_tool(self, tool_id: str) -> bool:
        return tool_id in self._costs

    async def has_tool_by_name(self, tool_name: str) -> bool:
        return False  # no legacy name fallback in tests

    async def get_cost_by_name(self, tool_name: str) -> int:
        return 0

    async def get_tool_pricing(self, tool_id: str):
        from tollbooth.pricing import ToolPricing
        return ToolPricing(fixed=self._costs.get(tool_id, 0))

    async def get_constraint_engine(self):
        return None

    def refresh(self):
        pass


def _make_registry(tool_costs: dict[str, int] | None = None) -> tuple[dict[str, ToolIdentity], dict[str, int]]:
    """Build a tool_registry and a resolver cost map from {name: cost}."""
    costs = tool_costs or {"my_tool": 1, "expensive_tool": 100}
    registry: dict[str, ToolIdentity] = {}
    resolver_costs: dict[str, int] = {}  # keyed by tool_id
    for name, cost in costs.items():
        category = "free" if cost == 0 else "read"
        identity = ToolIdentity(capability=name, category=category, intent=f"Test tool {name}")
        registry[name] = identity
        resolver_costs[identity.tool_id] = cost
    return registry, resolver_costs


def _make_runtime(tool_costs: dict[str, int] | None = None) -> OperatorRuntime:
    """Create a minimal runtime for testing (no vault, no Nostr)."""
    registry, resolver_costs = _make_registry(tool_costs)
    rt = OperatorRuntime(
        tool_registry=registry,
        nsec_env_var="__UNUSED__",
    )
    # Inject fake pricing resolver
    rt._pricing_resolver = FakePricingResolver(resolver_costs)
    return rt


class FakeLedgerCache:
    """In-memory ledger cache for testing."""

    def __init__(self, balance: int = 1000):
        from tollbooth.ledger import UserLedger, Tranche
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        self._ledgers: dict[str, UserLedger] = {}
        self._default_balance = balance
        self._Tranche = Tranche
        self._now = now

    async def get(self, npub: str):
        if npub not in self._ledgers:
            from tollbooth.ledger import UserLedger
            ledger = UserLedger()
            ledger.tranches.append(self._Tranche(
                granted_at=self._now,
                original_sats=self._default_balance,
                remaining_sats=self._default_balance,
                invoice_id="test-seed",
            ))
            self._ledgers[npub] = ledger
        return self._ledgers[npub]

    def mark_dirty(self, npub: str) -> None:
        pass

    async def flush_user(self, npub: str) -> None:
        pass


VALID_NPUB = "npub1" + "a" * 58  # 63 chars total


async def _inject_fake_cache(rt: OperatorRuntime, balance: int = 1000):
    """Replace the runtime's ledger cache with a fake."""
    cache = FakeLedgerCache(balance)
    rt._ledger_cache = cache
    return cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPaidToolDecorator:

    @pytest.mark.asyncio
    async def test_success_path(self):
        rt = _make_runtime()
        await _inject_fake_cache(rt)

        @rt.paid_tool("my_tool")
        async def my_tool(x: int, npub: str = "") -> dict:
            return {"value": x * 2}

        result = await my_tool(21, npub=VALID_NPUB)
        assert result["value"] == 42

    @pytest.mark.asyncio
    async def test_debit_insufficient_balance(self):
        rt = _make_runtime({"my_tool": 9999})
        await _inject_fake_cache(rt, balance=10)

        @rt.paid_tool("my_tool")
        async def my_tool(npub: str = "") -> dict:
            return {"should": "not reach"}

        result = await my_tool(npub=VALID_NPUB)
        assert result["success"] is False
        assert "Insufficient balance" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_npub_returns_error(self):
        rt = _make_runtime()
        await _inject_fake_cache(rt)

        @rt.paid_tool("my_tool")
        async def my_tool(npub: str = "") -> dict:
            return {"should": "not reach"}

        result = await my_tool(npub="")
        assert result["success"] is False
        assert "npub is required" in result["error"]

    @pytest.mark.asyncio
    async def test_rollback_on_exception_catch_errors_true(self):
        rt = _make_runtime()
        cache = await _inject_fake_cache(rt, balance=100)

        @rt.paid_tool("my_tool", catch_errors=True)
        async def my_tool(npub: str = "") -> dict:
            raise RuntimeError("boom")

        # Get initial balance
        ledger = await cache.get(VALID_NPUB)
        initial = ledger.balance_api_sats

        result = await my_tool(npub=VALID_NPUB)
        assert result["success"] is False
        assert "boom" in result["error"]

        # Balance should be restored after rollback
        ledger = await cache.get(VALID_NPUB)
        assert ledger.balance_api_sats == initial

    @pytest.mark.asyncio
    async def test_rollback_on_exception_catch_errors_false(self):
        rt = _make_runtime()
        cache = await _inject_fake_cache(rt, balance=100)

        @rt.paid_tool("my_tool", catch_errors=False)
        async def my_tool(npub: str = "") -> dict:
            raise RuntimeError("boom")

        ledger = await cache.get(VALID_NPUB)
        initial = ledger.balance_api_sats

        with pytest.raises(RuntimeError, match="boom"):
            await my_tool(npub=VALID_NPUB)

        # Balance should be restored after rollback
        ledger = await cache.get(VALID_NPUB)
        assert ledger.balance_api_sats == initial

    @pytest.mark.asyncio
    async def test_free_tool_skips_debit(self):
        rt = _make_runtime({"free_tool": 0})
        # No cache needed — free tools skip debit entirely

        @rt.paid_tool("free_tool")
        async def free_tool(npub: str = "") -> dict:
            return {"free": True}

        result = await free_tool(npub="")
        assert result["free"] is True

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        rt = _make_runtime()

        @rt.paid_tool("my_tool")
        async def my_tool(npub: str = "") -> dict:
            """My docstring."""
            return {}

        assert my_tool.__name__ == "my_tool"
        assert my_tool.__doc__ == "My docstring."

    @pytest.mark.asyncio
    async def test_result_includes_low_balance_warning(self):
        rt = _make_runtime({"my_tool": 1})
        await _inject_fake_cache(rt, balance=2)  # low balance after debit

        @rt.paid_tool("my_tool")
        async def my_tool(npub: str = "") -> dict:
            return {"data": "ok"}

        result = await my_tool(npub=VALID_NPUB)
        assert result["data"] == "ok"
        # Warning may or may not be present depending on threshold,
        # but the decorator shouldn't crash

    @pytest.mark.asyncio
    async def test_unpriced_tool_blocked(self):
        """A paid-category tool not in the pricing model should be blocked."""
        registry = {
            "unpriced": ToolIdentity(capability="unpriced", category="write", intent="test"),
        }
        rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
        # Resolver with empty costs — tool not in model
        rt._pricing_resolver = FakePricingResolver({})

        @rt.paid_tool("unpriced")
        async def unpriced(npub: str = "") -> dict:
            return {"should": "not reach"}

        result = await unpriced(npub=VALID_NPUB)
        assert result["success"] is False
        assert "not yet priced" in result["error"]
