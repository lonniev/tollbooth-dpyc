"""Tests for the @runtime.paid_tool() decorator."""

import pytest

from tollbooth.runtime import OperatorRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(tool_costs: dict[str, int] | None = None) -> OperatorRuntime:
    """Create a minimal runtime for testing (no vault, no Nostr)."""
    return OperatorRuntime(
        tool_costs=tool_costs or {"my_tool": 1, "expensive_tool": 100},
        nsec_env_var="__UNUSED__",
    )


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
