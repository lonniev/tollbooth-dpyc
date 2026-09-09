"""A refusal must give back what it took, and no more.

`rollback_debit` recomputed `pricing.compute(...)` — the BASE price, before any
constraint had run. That is correct only when nothing adjusted the fare, and
silently generous when something did: a patron on a full-discount coupon paid
nothing and was handed the list price back, so a refused call MINTED money.

Refusals are ordinary in a contended tool — a lost race, a rule that says no —
so this is a leak rather than a rounding error. It was found in a game where
every move is a fare and simulated players run on a 100%-off coupon.
"""

from __future__ import annotations

import pytest

from tollbooth import runtime as rt_mod
from tollbooth.runtime import OperatorRuntime

TOOL_ID = "6ddeebd4-4016-4423-8236-24159fc2ca64"
NPUB = "npub1m52k7yun9zhhdx586ev4a52rmvzhvrw7hgt3qkjlp082zx59qd8q680ydx"


class _Cache:
    """Just enough ledger to see what was handed back."""

    def __init__(self) -> None:
        self.credits: list[tuple[str, int, str]] = []

    async def credit(self, npub: str, sats: int, key: str) -> None:
        self.credits.append((npub, sats, key))


class _Identity:
    category = "write"


class _Pricing:
    """The LIST price — what the old code refunded."""

    def compute(self, **_: object) -> int:
        return 100


class _Resolver:
    async def get_tool_pricing(self, _tool_id: str) -> _Pricing:
        return _Pricing()


def _runtime(cache: _Cache) -> OperatorRuntime:
    rt = OperatorRuntime.__new__(OperatorRuntime)
    rt._tool_registry = {TOOL_ID: _Identity()}
    rt._mcp_names = {}

    async def ledger_cache() -> _Cache:
        return cache

    async def pricing_resolver() -> _Resolver:
        return _Resolver()

    rt.ledger_cache = ledger_cache          # type: ignore[method-assign]
    rt.pricing_resolver = pricing_resolver  # type: ignore[method-assign]
    rt.mcp_name_for = lambda _tid: "a_tool"  # type: ignore[method-assign]
    return rt


@pytest.fixture(autouse=True)
def _clean_context():
    token = rt_mod._CHARGED.set(None)
    yield
    rt_mod._CHARGED.reset(token)


async def test_a_discounted_fare_is_refunded_not_the_list_price():
    """The bug, stated as a number: charged 0, refunded 100."""
    cache = _Cache()
    rt = _runtime(cache)
    await rt.rollback_debit(TOOL_ID, NPUB, tool_kwargs={}, charged=0)
    assert cache.credits == [], "nothing was taken, so nothing goes back"


async def test_a_partial_discount_comes_back_in_full_but_no_further():
    cache = _Cache()
    rt = _runtime(cache)
    await rt.rollback_debit(TOOL_ID, NPUB, tool_kwargs={}, charged=30)
    assert cache.credits == [(NPUB, 30, "rollback:a_tool")]


async def test_the_charge_recorded_for_this_call_is_used_when_none_is_passed():
    """`paid_tool` records it per task, so the refund needs no argument."""
    cache = _Cache()
    rt = _runtime(cache)
    rt_mod._CHARGED.set((TOOL_ID, 7))
    await rt.rollback_debit(TOOL_ID, NPUB, tool_kwargs={})
    assert cache.credits == [(NPUB, 7, "rollback:a_tool")]


async def test_a_record_belonging_to_another_tool_is_not_spent_on_this_one():
    """A context is per task and a task may serve more than one call.

    Trusting the record blindly would refund one tool's fare against another's,
    which is a different way of getting the number wrong.
    """
    cache = _Cache()
    rt = _runtime(cache)
    rt_mod._CHARGED.set(("some-other-tool", 7))
    await rt.rollback_debit(TOOL_ID, NPUB, tool_kwargs={})
    assert cache.credits == [(NPUB, 100, "rollback:a_tool")], "falls back to the list price"


async def test_with_nothing_recorded_the_list_price_is_the_honest_fallback():
    """A runner refunding in another process cannot know better.

    It is kept rather than made an error: not refunding at all would be worse
    than refunding approximately, and the log says which path was taken.
    """
    cache = _Cache()
    rt = _runtime(cache)
    await rt.rollback_debit(TOOL_ID, NPUB, tool_kwargs={})
    assert cache.credits == [(NPUB, 100, "rollback:a_tool")]
